"""FSDP SHARDED_STATE_DICT 分片 → 完整 unet diffusers 目录 合并导出脚本（离线）

背景：train_layerdiff3d 手工分片保存（FSDP use_orig_params + SHARDED_STATE_DICT +
torch.save），每 rank 一份 model_shard_rankN.pt：FQN -> 1-D flat 本地切片；
本 rank 未覆盖的参数为空 tensor。rank i 的 flat chunk 在全局 flat param 中位于
rank i+1 之前，因此同一参数的分片按 rank 序 cat 即还原完整 flat param。

校验（脚本内置，任何一条不满足即中止，不会产出错误权重）：
- 各分片 key 集合一致，且值均为 1-D tensor
- 每个 key：各 rank numel 之和 == base unet 参数 numel
- 每个分片内：非空 key 的索引区间连续（FSDP 单 flat param 顺序切分）

合并后 load_state_dict(strict=True) 回灌 --base 模型（结构/config 全部继承），
再 save_pretrained 输出标准 diffusers unet 目录（config.json + safetensors 分片
+ index.json），可直接替换 repo_44/unet 推理。

用法（70bc venv）：
    python training/scripts/merge_fsdp_checkpoint.py \
        --base /home/ari/seethru_train/models/unet_group44 \
        --shards /path/model_shard_rank0.pt /path/model_shard_rank1.pt \
        --dst /home/ari/seethru_train/models/unet_44_ckpt4
"""
import argparse
import glob
import os
import os.path as osp
import sys

import torch
from safetensors import safe_open

sys.path.append(osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))), 'common'))

from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel


def _load_shard(path):
    """torch.load 一份 FSDP 分片。

    use_orig_params 下值为普通 1-D tensor，无需进程组；若 torch 版本把值
    反序列化成 ShardedTensor，则退回单进程 gloo group 再加载。
    """
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except Exception:
        import torch.distributed as dist
        if not dist.is_initialized():
            os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
            os.environ.setdefault('MASTER_PORT', '29517')
            dist.init_process_group('gloo', rank=0, world_size=1)
        return torch.load(path, map_location='cpu', weights_only=False)


def _shard_piece(sd, key):
    """取某个 key 在本 rank 分片中的 1-D flat 切片（兼容 ShardedTensor 值）。"""
    v = sd[key]
    if hasattr(v, 'local_shards'):
        ls = v.local_shards()
        if not ls:
            return None
        assert len(ls) == 1, f'{key}: 单文件含 {len(ls)} 个 local shard，不支持'
        t = ls[0].tensor
    else:
        t = v
    assert torch.is_tensor(t) and t.dim() <= 1, \
        f'{key}: 分片值不是 1-D tensor（shape={tuple(t.shape)}），分片格式不符'
    return t if t.numel() > 0 else None


def main():
    ap = argparse.ArgumentParser(description='Merge FSDP sharded_state_dict shards into a diffusers unet dir')
    ap.add_argument('--base', required=True,
                    help='参照 unet 目录（结构 + config 来源，如 unet_group44 或 repo_44/unet）')
    ap.add_argument('--shards', required=True, nargs='+',
                    help='各 rank 分片文件路径，必须按 rank 序给出')
    ap.add_argument('--dst', required=True, help='输出 unet diffusers 目录')
    ap.add_argument('--max_shard_size', default='10GB',
                    help='save_pretrained 单分片大小上限（默认 10GB，与 repo_44/unet 一致）')
    args = ap.parse_args()

    # ---- 加载全部分片 ----
    shards = []
    for p in args.shards:
        print(f'加载分片: {p}')
        sd = _load_shard(p)
        shards.append(sd)
        print(f'  keys={len(sd)} numel={sum(v.numel() if torch.is_tensor(v) else 0 for v in sd.values())}')

    key_sets = [set(sd.keys()) for sd in shards]
    for i, ks in enumerate(key_sets[1:], 1):
        assert ks == key_sets[0], \
            f'rank{i} 与 rank0 key 集合不一致: only-rank0={list(key_sets[0]-ks)[:5]} only-rank{i}={list(ks-key_sets[0])[:5]}'

    # ---- base 模型：结构 + config + 回灌目标 ----
    print(f'加载参照 unet: {args.base}')
    model = UNetFrameConditionModel.from_pretrained(args.base)
    ref_sd = model.state_dict()
    print(f'  参数量={sum(v.numel() for v in ref_sd.values())} keys={len(ref_sd)}')

    # ---- 逐 key 合并：rank 序 cat 1-D flat 切片 ----
    merged = {}
    n_fall_back = 0
    for k, ref in ref_sd.items():
        pieces = []
        for r, sd in enumerate(shards):
            if k not in sd:
                continue
            t = _shard_piece(sd, k)
            if t is not None:
                pieces.append((r, t))
        if not pieces:
            # 分片不覆盖该 key（如未被 FSDP 管理的 buffer）：沿用 base 值
            merged[k] = ref
            n_fall_back += 1
            continue
        flat = torch.cat([t for _, t in pieces]).to(ref.dtype)
        assert flat.numel() == ref.numel(), \
            f'{k}: 分片 numel 之和 {flat.numel()} != 参数 numel {ref.numel()}'
        merged[k] = flat.view(ref.shape)
    if n_fall_back:
        print(f'  {n_fall_back} 个 key 分片未覆盖，沿用 base 值')

    # ---- 校验分片布局：每个分片的非空 key 索引必须连续（单 flat param 顺序切分）----
    key_order = list(ref_sd.keys())
    pos = {k: i for i, k in enumerate(key_order)}
    for r, sd in enumerate(shards):
        idxs = sorted(pos[k] for k in sd if k in pos and _shard_piece(sd, k) is not None)
        if idxs:
            assert idxs == list(range(idxs[0], idxs[-1] + 1)), \
                f'rank{r}: 非空 key 索引不连续，分片布局不符合 FSDP 顺序切分假设'

    missing, unexpected = model.load_state_dict(merged, strict=False)
    assert not missing and not unexpected, \
        f'load_state_dict 对齐失败: missing={missing[:5]} unexpected={unexpected[:5]}'
    del merged, ref_sd, shards

    # ---- 导出 ----
    print(f'保存: {args.dst} (max_shard_size={args.max_shard_size})')
    model.save_pretrained(args.dst, max_shard_size=args.max_shard_size)
    del model

    # ---- 自检 ----
    print('自检：from_pretrained 读回 ...')
    chk = UNetFrameConditionModel.from_pretrained(args.dst)
    chk_sd = chk.state_dict()

    ref = UNetFrameConditionModel.from_pretrained(args.base)
    ref_sd = ref.state_dict()
    assert list(chk_sd.keys()) == list(ref_sd.keys()), '导出权重 key 集与 base 不一致'
    for k in ref_sd:
        assert chk_sd[k].shape == ref_sd[k].shape, f'{k} shape 不一致'
    n_diff = sum(0 if torch.equal(chk_sd[k], ref_sd[k]) else 1 for k in ref_sd)
    print(f'  与 base 不同权重 {n_diff}/{len(ref_sd)}（训练更新应 >0）')

    st_files = sorted(glob.glob(osp.join(args.dst, '*.safetensors')))
    assert st_files, '未产出 safetensors'
    total = 0
    for f in st_files:
        with safe_open(f, framework='pt') as sf:
            total += len(list(sf.keys()))
    assert total == len(chk_sd), f'safetensors 键数 {total} != 模型 {len(chk_sd)}'
    print(f'  safetensors 分片 {len(st_files)} 个、键数 {total}，config: '
          f'group_embedding_num={list(chk.config.group_embedding_num)} '
          f'tag_version={chk.config.get("tag_version")}')
    print('自检全部通过。')


if __name__ == '__main__':
    main()
