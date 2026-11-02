import sys
import os.path as osp
import argparse
import logging
import math
import os
import shutil
import datetime
import gc
from packaging import version
import random
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import accelerate
from PIL import Image
from omegaconf import OmegaConf
import torch
import transformers
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils.torch_utils import is_compiled_module
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
)
from diffusers.utils.import_utils import is_xformers_available
from transformers import AutoTokenizer, PretrainedConfig

from utils.visualize import pil_draw_text, imglist2imgrid_with_tags
from utils.cv import visualize_rgba
from utils.torchcv import pad_rgb_torch
from utils.io_utils import flatten_dict, imglist2imgrid, json2dict, dict2json
from train.dataset_layerdiff import LayerDiffDataset3D
from utils.torch_utils import init_model_from_pretrained, tensor2img, seed_everything
from modules.layerdiffuse.vae import TransparentVAEDecoder, TransparentVAEEncoder, vae_encode
from modules.layerdiffuse.utils import patch_unet_convin, patch_transvae_sd
from modules.layerdiffuse.diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline
from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel

logger = get_logger(__name__, log_level="INFO")


def update_param_group_lr(param_group, lr):
    def _to_scalar(x):
        if isinstance(x, torch.Tensor) and x.dim() != 0:
            return x.squeeze()
        else:
            return x

    if isinstance(param_group["lr"], torch.Tensor):
        param_group["lr"].fill_(_to_scalar(lr))
        
    else:
        param_group["lr"] = lr

    if isinstance(param_group["initial_lr"], torch.Tensor):
        param_group["initial_lr"].fill_(_to_scalar(lr))
        
    else:
        param_group["initial_lr"] = lr


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoders, tokenizers, proportion_empty_prompts=0., is_train=True, batch_size=1):
    prompt_embeds_list = []

    captions = []
    for caption in prompt_batch:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    num_frames = len(prompt_batch) // batch_size
    for ii in range(batch_size):
        if random.random() < proportion_empty_prompts:
            captions[ii * num_frames: (ii + 1) * num_frames] = [''] * num_frames

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
                return_dict=False,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds[-1][-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return {"prompt_embeds": prompt_embeds.cpu(), "pooled_prompt_embeds": pooled_prompt_embeds.cpu()}


def generate_timestep_weights(args, num_timesteps):
    weights = torch.ones(num_timesteps)

    # Determine the indices to bias
    num_to_bias = int(args.timestep_bias_portion * num_timesteps)

    if args.timestep_bias_strategy == "later":
        bias_indices = slice(-num_to_bias, None)
    elif args.timestep_bias_strategy == "earlier":
        bias_indices = slice(0, num_to_bias)
    elif args.timestep_bias_strategy == "range":
        # Out of the possible 1000 timesteps, we might want to focus on eg. 200-500.
        range_begin = args.timestep_bias_begin
        range_end = args.timestep_bias_end
        if range_begin < 0:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide a beginning timestep greater or equal to zero."
            )
        if range_end > num_timesteps:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide an ending timestep smaller than the number of timesteps."
            )
        bias_indices = slice(range_begin, range_end)
    else:  # 'none' or any other string
        return weights
    if args.timestep_bias_multiplier <= 0:
        return ValueError(
            "The parameter --timestep_bias_multiplier is not intended to be used to disable the training of specific timesteps."
            " If it was intended to disable timestep bias, use `--timestep_bias_strategy none` instead."
            " A timestep bias multiplier less than or equal to 0 is not allowed."
        )

    # Apply the bias
    weights[bias_indices] *= args.timestep_bias_multiplier

    # Normalize
    weights /= weights.sum()

    return weights


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    # # model
    # parser.add_argument(
    #     "--model_args", default=None
    # )

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--pretrained_unet",
        type=str,
        default=None
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )

    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--test_negative_prompt", type=str, default='')

    # optimizer & lr scheduler & hyperparameters
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_scheduler_step_rules",
        type=str,
        default=None
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--timestep_bias_strategy",
        type=str,
        default="none",
        choices=["earlier", "later", "range", "none"],
        help=(
            "The timestep bias strategy, which may help direct the model toward learning low or high frequency details."
            " Choices: ['earlier', 'later', 'range', 'none']."
            " The default is 'none', which means no bias is applied, and training proceeds normally."
            " The value of 'later' will increase the frequency of the model's final training timesteps."
        ),
    )
    parser.add_argument(
        "--timestep_bias_multiplier",
        type=float,
        default=1.0,
        help=(
            "The multiplier for the bias. Defaults to 1.0, which means no bias is applied."
            " A value of 2.0 will double the weight of the bias, and a value of 0.5 will halve it."
        ),
    )
    parser.add_argument(
        "--timestep_bias_begin",
        type=int,
        default=0,
        help=(
            "When using `--timestep_bias_strategy=range`, the beginning (inclusive) timestep to bias."
            " Defaults to zero, which equates to having no specific bias."
        ),
    )
    parser.add_argument(
        "--timestep_bias_end",
        type=int,
        default=1000,
        help=(
            "When using `--timestep_bias_strategy=range`, the final timestep (inclusive) to bias."
            " Defaults to 1000, which is the number of timesteps that Stable Diffusion is trained on."
        ),
    )
    parser.add_argument(
        "--timestep_bias_portion",
        type=float,
        default=0.25,
        help=(
            "The portion of timesteps to bias. Defaults to 0.25, which 25%% of timesteps will be biased."
            " A value of 0.5 will bias one half of the timesteps. The value provided for `--timestep_bias_strategy` determines"
            " whether the biased portions are in the earlier or later timesteps."
        ),
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://huggingface.co/papers/2303.09556.",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", type=int, default=200)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    
    parser.add_argument(
        "--learnable_parameters", type=str, default='all'
    )

    # dataset
    parser.add_argument(
        "--dataset_args",
        default=None,
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--valset_args",
        default=None,
    )

    # logging, saving, reports & resume
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=10000,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=1
    )
    parser.add_argument(
        "--val_mask_thr",
        type=int,
        default=0.
    )
    parser.add_argument(
        "--visualization_steps",
        type=int,
        default=3000,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--max_visualize_samples",
        type=int,
        default=8,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="layerdiff3d",
    )
    parser.add_argument(
        "--tracker_init_kwargs",
        type=str,
        default=None,
    )

    # performance
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--visualization_size", default=512)
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )

    parser.add_argument(
        "--local_rank",
        type=str,
        default=0,
    )

    # added args
    parser.add_argument(
        "--config",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--init_from_ckpt",
        type=str,
        default=None
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--vae_use_offset_prob",
        type=float,
        default=0.5,
    )

    args = parser.parse_args()

    args = OmegaConf.create(vars(args))
    if args.config is not None:
        config = OmegaConf.load(args.config)
        args.merge_with(config)

    if args.output_dir is None:
        _base_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
        if args.config is not None:
            args.output_dir = osp.join(_base_dir, 'workspace/training_output',
                                       osp.splitext(osp.basename(args.config))[0])
        else: 
            args.output_dir = osp.join(_base_dir, 'workspace/training_output', args.tracker_project_name)
        

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_args is None:
        args.dataset_args = {}

    if hasattr(args, 'model_args') and args.model_args is None:
        args.model_args = {}

    return args



def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")



@torch.inference_mode()
def visualize_batch(args, batch, model, vae, trans_vae_encoder, trans_vae_decoder, text_encoders, tokenizers, dataset, accelerator: Accelerator, global_step, pipeline=None):

    def _cvt_img(img, is_argb=True):
        img_np = tensor2img(img, mean=dataset.pixel_mean, std=dataset.pixel_std, denormalize=True)
        if is_argb:
            img_np = np.concatenate([img_np[..., 1:], img_np[..., [0]]], axis=2)
        return Image.fromarray(visualize_rgba(img_np))
        # return img_np

    model = accelerator.unwrap_model(model)
    weight_dtype = model.dtype
    device = model.device
    rng = torch.Generator(device=model.device).manual_seed(12345)

    create_pipeline = False
    if pipeline is None:
        pipeline = KDiffusionStableDiffusionXLPipeline(
            vae=vae,
            text_encoder=text_encoders[0],
            tokenizer=tokenizers[0],
            text_encoder_2=text_encoders[1],
            tokenizer_2=tokenizers[1],
            unet=model,
            scheduler=None,  # We completely give up diffusers sampling system and use A1111's method
        )
        create_pipeline = True


    bsz, num_frames, _, im_h, im_w = batch['img_list'].shape

    vis_list = []
    
    img_list = batch['img_list'][0]
    caption_list = batch['caption_list'][0].split('\n')
    cond_fullpage = batch['cond_fullpage'][0]
    empty_mask = batch['empty_mask'][0]

    if not dataset.pad_argb:
        img_list = pad_rgb_torch(img_list.to(device='cuda'), return_format='argb')
    # vis_img_list = [_cvt_img(cond_fullpage)] + [_cvt_img(img) for jj, img in enumerate(img_list) if not empty_mask[jj]]

    vis_img_list = [_cvt_img(cond_fullpage)]

    # positive_cond, positive_pooler = pipeline.encode_cropped_prompt_77tokens(
    #     caption_list
    # )

    # negative_cond, negative_pooler = pipeline.encode_cropped_prompt_77tokens(args.test_negative_prompt)

    prompt_encodings = encode_prompt(
        caption_list, text_encoders, tokenizers)
    prompt_embeds = prompt_encodings["prompt_embeds"].to(accelerator.device, dtype=weight_dtype)
    pooled_prompt_embeds = prompt_encodings["pooled_prompt_embeds"].to(accelerator.device)

    prompt_encodings = encode_prompt(
        [''] * len(caption_list), text_encoders, tokenizers)
    negative_cond = prompt_encodings["prompt_embeds"].to(accelerator.device, dtype=weight_dtype)
    negative_pooler = prompt_encodings["pooled_prompt_embeds"].to(accelerator.device)

    c_concat = vae_encode(vae, trans_vae_encoder, batch['cond_fullpage'][[0]].to(dtype=vae.dtype, device=vae.device), use_offset=False).to(dtype=weight_dtype)
    c_concat = c_concat[:, None].expand(-1, num_frames, -1, -1, -1)
    lh, lw = c_concat.shape[-2:]

    initial_latent = torch.zeros(size=(1, num_frames, 4, lh, lw), dtype=model.dtype, device=model.device)
    latents = pipeline(
        initial_latent=initial_latent,
        strength=1.0,
        num_inference_steps=25,
        batch_size=1,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_cond,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooler,
        generator=rng,
        guidance_scale=args.guidance_scale, c_concat=c_concat,
        group_index=batch['group_index']
    )

    for jj, c in enumerate(caption_list):
        if empty_mask[jj]:
            continue
        # print(latents.shape, c)
        latent = latents[[jj]].to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
        result_list, vis_list_batch = trans_vae_decoder(vae, latent)
        vis_img_list.append(vis_list_batch[0])

    caption_list = ['condition page'] + [c for jj, c in enumerate(caption_list) if not empty_mask[jj]]
    vis = imglist2imgrid_with_tags(vis_img_list, caption_list, skip_empty=True, fix_size=512)

    sample_saved = False
    if args.report_to == 'wandb':
        try:
            import wandb
            wandb_tracker = accelerator.get_tracker("wandb")
            wandb_tracker.log({f'train/samples': wandb.Image(vis)}, step=global_step)
            sample_saved = True
        except:
            pass
    if not sample_saved:
        Image.fromarray(vis).save(osp.join(args.output_dir, 'step-' + str(global_step).zfill(4) + '.jpg'), q=97)

    if create_pipeline:
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

    return vis


def merge_sd_with_lora(model):
    from utils.torch_utils import _get_model_file, load_state_dict
    model_file = _get_model_file(pretrained_model_name_or_path="lllyasviel/LayerDiffuse_Diffusers", weights_name="ld_diffusers_sdxl_attn.safetensors")
    sd_offset = load_state_dict(model_file)

    # sd_offset = sf.load_file(path_ld_diffusers_sdxl_attn)
    sd_origin = model.state_dict()
    keys = sd_origin.keys()
    sd_merged = {}

    for k in sd_origin.keys():
        if k in sd_offset:
            sd_merged[k] = sd_origin[k] + sd_offset[k]
        else:
            sd_merged[k] = sd_origin[k]

    model.load_state_dict(sd_merged, strict=True)
    del sd_offset, sd_origin, sd_merged, keys, k



def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    use_deepspeed = accelerator.distributed_type == DistributedType.DEEPSPEED
    if use_deepspeed:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_batch_size

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision


    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        seed_everything(args.seed)

    # Handle the repository creation (所有 rank 创建 output_dir，供分片保存使用)
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)


    initial_global_step = 0

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    optimizer_name = args.optimizer.lower()
    if optimizer_name == 'adamw':

        # Initialize the optimizer
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                )

            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

    # some how init model here
    # model = None
    
    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    # Check for terminal SNR in combination with SNR Gamma
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )
    trans_vae_encoder: TransparentVAEEncoder = init_model_from_pretrained(
        'lllyasviel/LayerDiffuse_Diffusers',
        TransparentVAEEncoder, weights_name='ld_diffusers_sdxl_vae_transparent_encoder.safetensors',
        patch_state_dict_func=patch_transvae_sd
    )

    trans_vae_decoder: TransparentVAEDecoder = init_model_from_pretrained(
        'lllyasviel/LayerDiffuse_Diffusers',
        TransparentVAEDecoder, weights_name='ld_diffusers_sdxl_vae_transparent_decoder.safetensors',
        patch_state_dict_func=patch_transvae_sd
    )

    model = UNetFrameConditionModel.from_pretrained(
        args.pretrained_unet
    )
    model.init_weights(zero_group_embeds=args.get('zero_init_group_embeds', True))

    # Freeze vae and text encoders.
    vae.requires_grad_(False)
    trans_vae_encoder.requires_grad_(False)
    trans_vae_decoder.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    # Set unet as trainable.
    model.train()

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae.to(accelerator.device, dtype=torch.float32)
    trans_vae_decoder.to(accelerator.device, dtype=torch.float32)
    trans_vae_encoder.to(accelerator.device, dtype=torch.float32)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    text_encoders = [text_encoder_one, text_encoder_two]
    tokenizers = [tokenizer_one, tokenizer_two]


    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNetFrameConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNetFrameConditionModel.from_pretrained(input_dir)
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    spatial_params = []
    temporal_params = []
    for name, p in model.named_parameters():
        if 'temporal_transformer_blocks' in name:
            temporal_params.append(p)
        else:
            spatial_params.append(p)

    if args.learnable_parameters == 'all':
        # parameters_with_lr = [{"params": spatial_params, "lr": args.learning_rate}, {"params": temporal_params, "lr": args.learning_rate}]
        parameters_with_lr = [{"params": model.parameters(), "lr": args.learning_rate}]
    elif args.learnable_parameters == 'temporal':
        parameters_with_lr = [{"params": temporal_params, "lr": args.learning_rate}]
    elif args.learnable_parameters == 'spatial':
        parameters_with_lr = [{"params": spatial_params, "lr": args.learning_rate}]
    else:
        raise Exception(f'Invalid learnable_parameters: {args.learnable_parameters}')

    optimizer = optimizer_cls(
        parameters_with_lr,
        **OmegaConf.to_container(args.optimizer_args)
    )

    train_dataset = LayerDiffDataset3D(**OmegaConf.to_container(args.dataset_args))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        step_rules=args.lr_scheduler_step_rules
    )

    # Prepare everything with our `accelerator`.

    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler
    )
    train_dataloader = accelerator.prepare_data_loader(train_dataloader, device_placement=False)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # 归还已释放的 host 内存给 OS（glibc arena 不自动回收，RSS 高水位会顶爆 earlyoom）
    def trim_host_memory():
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except Exception:
            pass

    if accelerator.is_main_process:
        dcfg = OmegaConf.to_container(args)
        tracker_init_kwargs = dcfg['tracker_init_kwargs']
        if tracker_init_kwargs is None:
            tracker_init_kwargs = {}
        # otherwise accelerate refuse the config dict
        dcfg = flatten_dict(dcfg, separator='.')
        for k, v in dcfg.items():
            if isinstance(v, list):
                if len(v) > 0 and isinstance(v[0], (int, float)):
                    dcfg[k] = torch.tensor(v)
                elif isinstance(v, list):
                    dcfg[k] = None
                elif isinstance(v, dict):
                    dcfg[k] = None
                else:
                    dcfg[k] = v
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        if args.report_to == 'wandb' and 'wandb' not in tracker_init_kwargs:
            tracker_init_kwargs['wandb'] = {}
        if 'wandb' in tracker_init_kwargs:
            runname_prefix = osp.basename(args.output_dir)
            tracker_init_kwargs['wandb']['name'] = runname_prefix + '_' + now
            tracker_init_kwargs['wandb']['dir'] = args.output_dir
        accelerator.init_trackers(args.tracker_project_name, dcfg, tracker_init_kwargs)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    # 自动检测最新 checkpoint（未显式指定时）
    if not args.resume_from_checkpoint and os.path.isdir(args.output_dir):
        _dirs = [d for d in os.listdir(args.output_dir) if d.startswith('checkpoint')]
        if _dirs:
            args.resume_from_checkpoint = 'latest'
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            if os.path.isdir(args.output_dir):
                dirs = os.listdir(args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None
            else:
                path = None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            # 手工 FSDP SHARDED 分片加载：恢复模型权重 + 优化器状态（动量/二阶矩续训）
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.api import StateDictType, ShardedOptimStateDictConfig
            fsdp_model = accelerator.unwrap_model(model)
            # FSDP.optim_state_dict* 需要真正的 FSDP 根 module（unwrap 后的内部 module 不含 FSDP 状态）
            FSDP.set_state_dict_type(
                model,
                StateDictType.SHARDED_STATE_DICT,
                optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=True),
            )
            shard_path = os.path.join(args.output_dir, path, f'model_shard_rank{accelerator.process_index}.pt')
            if os.path.isfile(shard_path):
                shard = torch.load(shard_path, map_location='cpu', weights_only=False, mmap=True)
                fsdp_model.load_state_dict(shard)
                del shard
                trim_host_memory()
                accelerator.print(f'  Loaded FSDP shard: {shard_path}')
            else:
                raise FileNotFoundError(f'FSDP shard not found: {shard_path}')
            optim_shard_path = os.path.join(args.output_dir, path, f'optim_shard_rank{accelerator.process_index}.pt')
            raw_optimizer = getattr(optimizer, 'optimizer', optimizer)
            if os.path.isfile(optim_shard_path):
                optim_shard = torch.load(optim_shard_path, map_location='cpu', weights_only=False, mmap=True)
                optim_state = FSDP.optim_state_dict_to_load(model, raw_optimizer, optim_shard)
                del optim_shard
                raw_optimizer.load_state_dict(optim_state)
                del optim_state
                trim_host_memory()
                accelerator.print(f'  Loaded FSDP optimizer shard: {optim_shard_path}')
            else:
                accelerator.print(f'  WARNING: no optimizer shard in {path}, optimizer state starts from zero')
            FSDP.set_state_dict_type(model, StateDictType.FULL_STATE_DICT)
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            # 恢复 lr_scheduler 步数（warmup 连续性）
            for _ in range(initial_global_step):
                lr_scheduler.step()

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    torch.cuda.empty_cache()

    loss_dict = {}

    torch.cuda.empty_cache()

    def vae_encode_frames(vae, trans_vae_encoder, batched_img_list, use_offset=False, dtype=torch.float32, pad_argb=False):
        batched_latent_list = []
        for img_list in batched_img_list:
            img_list = img_list.to(device=vae.device, dtype=torch.float32)
            if pad_argb:
                img_list = pad_rgb_torch(img_list, return_format='argb')
            frame_latents = []
            for i in range(img_list.shape[0]):
                frame_latents.append(
                    vae_encode(vae, trans_vae_encoder, img_list[i:i+1], use_offset=use_offset)
                )
            batched_latent_list.append(torch.cat(frame_latents, dim=0).to(dtype=dtype))
        return torch.stack(batched_latent_list)

    noise_generator = None
    # if args.diff_noise_generator:
    rank = int(os.environ.get('LOCAL_RANK',-1))
    if args.seed:
        rank += args.seed
    noise_generator = torch.Generator(device=accelerator.device)
    noise_generator.manual_seed(rank)

    device = accelerator.device

    # if hasattr(args, 'temporal_lr'):
    #     for step, value in args.temporal_lr.items():
    #         if initial_global_step <= step:
    #             update_param_group_lr(optimizer.param_groups[1], value)

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # if accelerator.is_main_process:
            #     visualize_batch(args, batch, model, vae, trans_vae_encoder, trans_vae_decoder, text_encoders, tokenizers, train_dataset, accelerator, global_step)

            with accelerator.accumulate(model):
                vae_use_offset = random.random() < args.vae_use_offset_prob
                torch.cuda.reset_peak_memory_stats()

                img_latent = vae_encode_frames(vae, trans_vae_encoder, batch['img_list'], use_offset=vae_use_offset, pad_argb=not train_dataset.pad_argb, dtype=weight_dtype)

                c_concat = vae_encode(vae, trans_vae_encoder, batch['cond_fullpage'], use_offset=vae_use_offset).to(dtype=weight_dtype)
                m_encode = torch.cuda.max_memory_allocated() / 1024**3
                m_reserved = torch.cuda.max_memory_reserved() / 1024**3

                noise = torch.randn_like(img_latent)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (img_latent.shape[0], img_latent.shape[1], 1, 1), device=img_latent.device, generator=noise_generator
                    )

                bsz, num_frames = img_latent.shape[:2]
                if args.timestep_bias_strategy == "none":
                    # Sample a random timestep for each image without bias.
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=img_latent.device, generator=noise_generator
                    )
                else:
                    # Sample a random timestep for each image, potentially biased by the timestep weights.
                    # Biasing the timestep weights allows us to spend less time training irrelevant timesteps.
                    weights = generate_timestep_weights(args, noise_scheduler.config.num_train_timesteps).to(
                        img_latent.device
                    )
                    timesteps = torch.multinomial(weights, bsz, replacement=True, generator=noise_generator).long()

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(img_latent, noise, timesteps).to(dtype=weight_dtype)


                # time ids
                def compute_time_ids():
                    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                    target_size = (args.dataset_args.target_size, args.dataset_args.target_size)
                    add_time_ids = list(target_size + (0, 0) + target_size)
                    add_time_ids = torch.tensor([add_time_ids], device=accelerator.device, dtype=weight_dtype)
                    return add_time_ids

                add_time_ids = torch.cat(
                    [compute_time_ids() for _ in range(bsz * num_frames)]
                )

                # Predict the noise residual
                unet_added_conditions = {"time_ids": add_time_ids}
                caption_list = []
                for c in batch['caption_list']:
                    caption_list += c.split('\n')
                prompt_encodings = encode_prompt(
                    caption_list, text_encoders, tokenizers, 
                    proportion_empty_prompts=args.proportion_empty_prompts
                )
                prompt_embeds = prompt_encodings["prompt_embeds"].to(accelerator.device, dtype=weight_dtype)
                pooled_prompt_embeds = prompt_encodings["pooled_prompt_embeds"].to(accelerator.device)
                unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})
                # print(pooled_prompt_embeds.shape, prompt_embeds.shape, batch['group_index'])
                model_pred = model(
                    torch.cat([noisy_model_input, c_concat[:, None].expand(-1, num_frames, -1, -1, -1)], dim=2),
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=unet_added_conditions,
                    return_dict=False,
                    group_index=batch['group_index']
                )[0]

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(img_latent, noise, timesteps)
                elif noise_scheduler.config.prediction_type == "sample":
                    # We set the target to latents here, but the model_pred will return the noise sample prediction.
                    target = img_latent
                    # We will have to subtract the noise residual from the prediction to get the target sample.
                    model_pred = model_pred - noise
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.snr_gamma is None:
                    loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://huggingface.co/papers/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # # Gather the losses across all processes for logging (if we use distributed training).
                # avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                # train_loss += avg_loss.item() / args.gradient_accumulation_steps
                loss_dict_local = {}
                loss_dict_local['loss_latent'] = loss
                # Gather the losses across all processes for logging
                for loss_key, loss_val in loss_dict_local.items():
                    if loss_key not in loss_dict:
                        loss_dict[loss_key] = 0
                    avg_loss = accelerator.gather(loss_val.repeat(args.train_batch_size)).mean()
                    loss_dict[loss_key] += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                m_step = torch.cuda.max_memory_allocated() / 1024**3
                if global_step % 1 == 0 and accelerator.is_local_main_process:
                    print(f'[mem-rank{accelerator.local_process_index}] encode_peak={m_encode:.1f}G reserved={m_reserved:.1f}G step_peak={m_step:.1f}G', flush=True)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # if hasattr(args, 'temporal_lr'):
                #     for step, value in args.temporal_lr.items():
                #         if global_step < step:
                #             update_param_group_lr(optimizer.param_groups[1], value)
                #             break

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                last_lr = last_lr = [group["lr"] for group in lr_scheduler.scheduler.optimizer.param_groups]
                loss_dict['lr'] = last_lr[0]
                if len(last_lr) > 1:
                    loss_dict['lr_temporal'] = last_lr[1]
                accelerator.log(loss_dict, step=global_step)
                torch.cuda.empty_cache()  # periodic cache release after each optimizer step
                loss_dict = {}

                # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                # 分片保存：所有 rank 各存自己的分片；每个 rank 清理自己磁盘上超额的 checkpoint
                if True:
                    if global_step % args.checkpointing_steps == 0 or global_step == args.max_train_steps:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            if not os.path.isdir(args.output_dir):
                                os.makedirs(args.output_dir, exist_ok=True)
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    if os.path.isdir(removing_checkpoint):
                                        shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                        from torch.distributed.fsdp.api import StateDictType, ShardedOptimStateDictConfig
                        fsdp_model = accelerator.unwrap_model(model)
                        raw_optimizer = getattr(optimizer, 'optimizer', optimizer)
                        # unwrap 后的是内部 module；FSDP.*state_dict* API 必须作用于 FSDP 根 module（即 model 本身）
                        FSDP.set_state_dict_type(
                            model,
                            StateDictType.SHARDED_STATE_DICT,
                            optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=True),
                        )
                        # 模型分片与优化器分片分开写盘并及时释放，压低保存瞬间 host 内存峰值
                        shard = fsdp_model.state_dict()
                        torch.save(shard, os.path.join(save_path, f"model_shard_rank{accelerator.process_index}.pt"))
                        del shard
                        trim_host_memory()
                        optim_shard = FSDP.optim_state_dict(model, raw_optimizer)
                        torch.save(optim_shard, os.path.join(save_path, f"optim_shard_rank{accelerator.process_index}.pt"))
                        del optim_shard
                        trim_host_memory()
                        torch.cuda.empty_cache()
                        logger.info(f"Saved sharded model+optimizer to {save_path}")

                if accelerator.is_main_process:
                    if global_step % args.visualization_steps == 0:
                        visualize_batch(
                            args, batch, model, vae, trans_vae_encoder, trans_vae_decoder, text_encoders, tokenizers, train_dataset, accelerator, global_step)

            last_lr = [group["lr"] for group in lr_scheduler.scheduler.optimizer.param_groups]
            logs = {"step_loss": loss.detach().item(), "lr": last_lr[0]}
            if len(last_lr) > 1:
                logs['lr_temporal'] = last_lr[1]
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    main()