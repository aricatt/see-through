# See-through 44 语义标签架构改造方案

> 目标：用 347 角色数据集（44 细粒度部件标签）微调 LayerDiff 3D v0.0.2，让全部 44 个标签作为独立监督信号参与提示词切割（prompt→layer）。
> 本文档只出方案，不改代码。

---

## 0. 现状机制梳理（已读码确认）

### 0.1 GroupEmbedding 工作原理

`common/modules/layerdiffuse/layerdiff3d.py`：

- `GroupEmbedding(n_cls, ndim)`（L533-544）：`params` 形状 `(n_cls, ndim)` + `linear`（ndim→ndim）。forward 为 `linear(x + params)` —— **按 batch 维位置寻址**：`x` 的第 i 行加 `params[i]`。即每组的第 i 个 tag 对应 `params` 第 i 行，**纯位置语义**。
- `group_embeds[g]`（ndim=1280）：在 `get_aug_embed`（L1485-1503，`addition_embed_type="text_time"`）中作用于 pooled `text_embeds` `(F, 1280)`，结果进 `add_embedding`（TextTimeEmbedding）调制 time embedding。
- `group_embeds2[g]`（ndim=2048）：在 forward L1628 作用于 token 级 `encoder_hidden_states` `(F, 77, 2048)`，逐 token 加 bias 后过 linear，再残差加回。
- `group_index`（forward 参数，L1574）在 `ModuleList` 里选组。**F（num_frames）必须等于该组 tag 数**，且依赖 broadcast，因此 `train_batch_size` 必须为 1。
- `init_weights()`（L996-1004）：把每个 GroupEmbedding 的 `linear` zero-init（`params` 不动），同时 zero temporal `ff.net.2`。训练脚本 `train_layerdiff3d.py` L821 在 `from_pretrained` 后**无条件调用** —— 即当前微调流程其实已经丢弃了 v0.0.2 已训练的 group linear（只保留 params）。

### 0.2 44→24 信息丢失点

**不在代码里**，在远端 `/home/ari/seethru_train/prepare_dataset.py` 的 `PART_MAP`：把 44 个部件名合并映射到 24 tag，重写出 `{prefix}_{tag}.png` + `{tag}_depth.png`（gen_depth.py 用 `255-alpha` 造伪深度）+ 改写 `_ann.json`。训练侧 `LayerDiffDataset3D` 本身对 tag 名是不可知的——`load_blend_parts` 按 `mix_tag_groups[i]['tags']` 逐名读 `{prefix}_{tag}.png` 与 `{prefix}_{tag}_depth.png`，`ann['tag_info'][t]['xyxy']` 取框。

### 0.3 已确认的远端数据事实（ssh 只读验证）

- `/home/ari/seethru_train/datasets/images/`：347 个 `*_fullbody.png`，部件图命名 `{prefix}_{tag}.png`（如 `..._eye_left.png`），**每个部件图尺寸 == 其 `tag_info[t]['xyxy']` 尺寸**（裁好的局部图），完全符合 `load_blend_parts` 的输入契约。
- `_ann.json`：`{'final_size': [H, W], 'tag_info': {44 tags: {'valid', 'exists', 'xyxy'}}}`，`exists=False` 的 tag 无 `xyxy`、也无对应 png。
- **无任何 `_depth.png`** —— 直接喂给 `load_blend_parts` 会 `FileNotFoundError`（数据侧必须补伪深度，或改代码容忍缺失）。
- tag 覆盖率（347 样本）：accessory 98%、face 92%、topwear 85%、eye_l/r 84%……尾部 tail 6%、wing_l/r 6%、animal_ear_l/r 3%。`accessory` 是"兜底"层（bbox 常覆盖大半个身子）。
- v0.0.2 unet `config.json`：`group_embedding: true, group_embedding_num: [13, 11], in_channels: 8, tag_version: 'v3'`。权重键：`group_embeds{,2}.{0,1}.params`（(13|11,1280|2048)）、`.linear.weight/bias`，BF16。

### 0.4 训练数据流（LayerDiffDataset3D）

每个 `get_sample`：取 `mix_tag_groups[_current_group]` 的 tags → `load_blend_parts` 读全部部件 → crop = 组内存在部件的 union xyxy（`crop_tag_list` 默认=组内 tags 减 `size_skip_tags`）→ `img_list` 为按 tag 序的 F 帧（缺失 tag = 全零帧，`empty_mask` 标记但 **loss 不用它**，缺失 tag 也参与监督→学"该 tag 无内容"）→ `caption_list` = tag 名本身 → `group_index` 随样本轮转（`ngroup_iters`）。

### 0.5 推理侧现状

- `pipeline.cache_tag_embeds()`（`diffusers_kdiffusion_sdxl.py` L242-261）：`tag_version=='v3'` 时**硬编码** body(13)/head(11) 两个 tag 列表。
- `apply_layerdiff`（`common/utils/inference_utils.py` L104-180）：v3 分支两遍推理——pass1 用 body 列表（`group_index=0`），取输出 `images[2]`（即 'head'）alpha bbox → `_crop_head` 扩边得到头部裁剪 → pass2 用 head 列表（`group_index=1`）在裁剪上生成头部细节层。
- `inference_psd_quantized.py`（L148/166/205/208）、`inference_psd_blockswap.py`：同样硬编码列表与 `group_index`。
- `further_extr`/`tag_lr_split`：推理后对 eyes/handwear/ears 做 L/R 连通域拆分——44 标签原生拆分后该步骤可弱化。

---

## A. 架构改造方案

### A.1 `group_embedding_num` 新值

`group_embedding_num` 是**配置驱动**的：`UNetFrameConditionModel.__init__` 按它为每组各建一对 `GroupEmbedding(n,1280)` + `GroupEmbedding(n,2048)`（L725-731），维度从 `unet/config.json` 读，**模型代码本身零改动**即可支持任意组数/组大小。

| 方案 | `group_embedding_num` | 说明 |
|---|---|---|
| **推荐：2 组** | `(30, 15)` | group0=body（29 个数据 tag + 1 个合成 anchor tag `head`），group1=head（15）。详见 B。 |
| 备选：3 组 | `(22, 8, 15)` | upper/lower/head，把最大帧数从 30 降到 22，见 B.4。 |

### A.2 新 embedding 权重初始化策略

**推荐做法：离线"扩维预烘焙"checkpoint**，新增一次性脚本 `training/scripts/expand_group_embeds.py`：

1. 读 v0.0.2 unet（`diffusion_pytorch_model.safetensors` + `config.json`）。
2. 每组按新 tags 顺序分配 `params (n_new, d)`：
   - 新 tag 有旧 24 父类 → **复制父类所在组对应行的 params**（映射表见附录 B.5；44 个 tag 全部有父类，合成 `head` ← 旧 group0 'head' 行）。
   - `group_embeds` 与 `group_embeds2` 两套 params 用同一映射分别复制（1280/2048 两套独立行）。
3. `linear`：逐组复制——新 g0 ← 旧 g0 linear，新 g1 ← 旧 g1 linear（若用 3 组：新 g0、g1 ← 旧 g0，新 g2 ← 旧 g1）。`linear` 是组内所有 tag 共享的变换，直接继承即可。
4. 更新 `config.json`：`group_embedding_num` 新值、`tag_version: 'v4'`、新增 `tag_groups` 字段（完整组内 tag 顺序，供推理泛化读取）。
5. `save_pretrained` 输出新 unet 目录（如 `unet_group44/`）。之后 `from_pretrained` 无缝加载，**shape 完全对齐，不需要 strict=False**。

**初始化策略对比**：

| 策略 | 优点 | 缺点 |
|---|---|---|
| **复制映射（推荐）** | ① t0 时刻每个映射 tag 的 group 条件≈其父类行为，24 语义条件能力最大限度保留；② split 兄弟槽（eye_l/r←irides）从同一暖起点出发，靠文本提示词+监督快速分化；③ 收敛快，347 样本小数据下尤其重要 | split 槽初始完全相同（但 caption 已不同，微步即分化）；无本质风险 |
| 全随机 `params` | 实现最简 | ① 丢弃 v0.0.2 已学条件结构；② 若保留 pretrained linear，随机 params 经已训练 linear 会产生不可控偏移，t0 条件反而更差；③ 小数据下重新学习 24 语义属浪费 |
| 混合（有父类复制、无父类随机） | 理论上最"干净" | 本场景所有 44 tag 均有父类 → 等价于复制映射，无额外收益 |

**推荐**：复制映射 + **保留 pretrained linear**（见下）。

#### 关于 `init_weights()` 的一个现状问题

`train_layerdiff3d.py` L821 `model.init_weights()` 会把 group linear 全部 zero。官方当初 zero 是因为该模块是从零叠在 SDXL 上的新模块（zero-init 不扰动底座）；但 v0.0.2 的 group embed 已训练，微调时再 zero 属于丢弃已学函数。建议：

- `init_weights` 增加参数 `zero_group_embeds: bool = True`（默认保持原行为，cvt 脚本 `cvt_layerdiff2d_to_3d.py` L43 不受影响）；
- 训练脚本加 yaml 开关（如 `zero_init_group_embeds: false`），44 微调时传 `False` 以保留 linear。
- 若选择维持 zero（最保守、与当前 v1 微调行为一致），复制映射的 **params 依然生效**（params 不在 zero 范围），只是 group 通路从静默起步重学变换——作为兜底方案可接受，但会损失更多 24 能力。

### A.3 24 语义能力退化评估

| 风险点 | 程度 | 分析与缓解 |
|---|---|---|
| 原 24 tag **名称消失**（irides/eyebrow/ears/handwear/footwear/wings/front·back hair/objects 等合并名不再有专属槽位） | 中 | tag 空间整体替换，非能力退化而是接口变更。若需兼容旧提示词，保留 v0.0.2 权重作旧接口；新模型对外只承诺 44+1 语义。 |
| 映射 tag 的能力保持 | 低 | 复制 params+linear 后，1:1 tag（face/nose/mouth/neck/neckwear/topwear/bottomwear/legwear/tail/eyewear/headwear）t0 条件**逐位相同**；split tag（eye_l/r 等）=父类克隆起点。 |
| 文本通路 | 无风险 | tag 身份的主载体本来就是文本提示（每帧 caption=tag 名），CLIP 对 'eye left'/'eye right' 天然区分；group embed 是辅助偏置。建议 caption 做 `tag.replace('_',' ')` 归一化（见 A.5），与官方 'front hair' 风格一致。 |
| 稀疏 tag 欠训练（animal_ear 3%、tail/wing 6%） | 中 | 每样本对组内所有 tag 都有监督（缺失=空层负样本），负样本极多反而稳；缓解靠父类暖启动 + 训练后按 tag 抽查生成质量；若某 tag 持续退化可在 `src_list` 侧对含该 tag 的样本加 `sample_rate` 加权。 |
| `accessory` 98% 兜底层吞语义 | 中（数据侧） | 其 bbox 常覆盖大半身，可能学到"什么都往里塞"。建议数据准备时抽查；必要时对 accessory 做区域收缩或拆分标注（数据质量问题，非架构问题）。 |
| L/R 翻转歧义 | 低 | `random_flip=0.5` 会把 eye_left 镜像到视觉右侧——官方同样 flip；L/R 语义在生成侧本就应是"画面左/右"，可接受。若要严格语义，降为 0 或做 tag 名随 flip 互换（不推荐，复杂度高）。 |
| 旧 checkpoint 不可 resume | 预期内 | 44 模型与 24 ckpt shape 不同，无法续训/合并，需文档化告知。 |

### A.4 训练/推理兼容性

**训练侧**：

- `UNetFrameConditionModel.from_pretrained(扩维unet目录)`：config.json 携带新 `group_embedding_num` → 模块按新维度构建 → safetensors 逐键对齐。**无 missing/unexpected keys，无需 strict=False**。
- checkpoint resume（`load_model_hook` L883-886）：从保存目录 `from_pretrained`，保存的 config 已含新维度 → 一致。
- FSDP 分片保存/加载（L1029-1041、L1280-1285）：纯 state_dict 操作，与新参数无关，兼容。
- `train_batch_size` 必须保持 1（broadcast 约束，本就如此）。
- `pad_num_frames` 必须保持 -1/不设置：padding 会撑大 num_frames 破坏 `params` broadcast。
- `ngroup_iters`/`_current_group` 轮转机制对任意组数通用，无需改。

**推理侧（必须同步改）**：

| 位置 | 现状 | 改动 |
|---|---|---|
| `diffusers_kdiffusion_sdxl.py` `cache_tag_embeds` | v3 硬编码 13/11 | 新增分支：`tag_version=='v4'` 时从 `unet.config.tag_groups` 读组列表编码缓存（保留 v3/v2 兼容） |
| `inference_utils.py` `apply_layerdiff` | v3 硬编码两遍推理 + `images[2]` 'head' anchor | 新增 `'v4'` 分支：按 `tag_groups` 顺序跑各组 pass；head 裁剪 anchor 取 `head` tag 输出（位置=其在 group0 列表中的 index，不要硬编码 `images[2]`，用 `tag_list.index('head')`）；3 组方案则跑 3 个 pass |
| `inference_psd_quantized.py` / `inference_psd_blockswap.py` | 硬编码 tag 列表与 group_index | 改为从 `unet.config.tag_groups` 读取（或同步更新字面量） |
| `inference_utils.py` `further_extr`/`tag_lr_split`/`apply_marigold` | 对 eyes/handwear/ears 做 L/R 后拆分；marigold 按 V2 tag 表读 `{tag}.png` | 44 原生拆分后 `tag_lr_split` 可关；`apply_marigold` 的深度阶段若要沿用需换用新 tag 表（注意 marigold 是独立 v0.0.1 ckpt，其自身 tag 假设不变——读文件列表换新表即可，属次要项） |
| UI（`ui/ui/ui_config.py` 的 `tagcluster_bodypart_v2.json`） | 标注器词表 | 推理产物标签变更后可另行更新，非训练阻塞项 |

**tag_version**：扩维 ckpt 的 `config.json` 写 `tag_version: 'v4'`，推理按版本分支，避免新代码误用旧模型/旧代码误用新模型。

### A.5 改动文件清单

| 文件 | 改动 | 必需性 |
|---|---|---|
| `training/scripts/expand_group_embeds.py`（**新增**） | 离线扩维烘焙：44→父类映射表、`params`/`linear` 复制、写新 config（`group_embedding_num`、`tag_version:'v4'`、`tag_groups`）、`save_pretrained` | **必需** |
| `prepare_dataset_44.py`（远端新增，仿 `prepare_dataset.py`/`gen_depth.py`） | ① 拷贝 `*_fullbody.png` + 部件 png（命名已合规，免合并）；② 为每个部件图生成 `{tag}_depth.png = 255-alpha`（`load_blend_parts` 硬性需要，blended 图实为 dead path，伪深度足够）；③ 合成 `head` anchor 层（见 B.3）并写入新 `_ann.json`；④ 生成 `l2d_bodysamples_44.txt` 样本列表 | **必需** |
| `common/modules/layerdiffuse/layerdiff3d.py` | ① `init_weights(zero_group_embeds=True)` 加开关；② （可选）加 `get_tag_groups()` 返回 `config.tag_groups` 供推理用 | 必需（①）；推荐（②） |
| `training/train/train_layerdiff3d.py` | L821 调用处改为读 yaml 开关：`model.init_weights(zero_group_embeds=args.get('zero_init_group_embeds', True))` | 必需 |
| `training/train/dataset_layerdiff.py` | `LayerDiffDataset3D.get_sample` 中 `caption_list = [t.replace('_',' ') for t in target_tags]`（tag→caption 归一化；文件名仍用原 tag） | 推荐 |
| `training/configs/finetune_layerdiff3d_v002_44tag.yaml`（**新增**，复制 v1 改） | `pretrained_unet` 指扩维目录；`dataset_args.tag_list` 换 44+1；`mix_tag_groups` 换 B.2 组定义；`src_list` 指新样本列表；加 `zero_init_group_embeds: false` | **必需** |
| `common/modules/layerdiffuse/diffusers_kdiffusion_sdxl.py` | `cache_tag_embeds` 增加 v4 分支读 `config.tag_groups` | 推理必需 |
| `common/utils/inference_utils.py` | `apply_layerdiff` 加 `'v4'` 分支（多组 pass + `head` anchor 定位）；`apply_marigold`/`further_extr` 换 tag 表（次要） | 推理必需 |
| `inference/scripts/inference_psd_quantized.py`、`inference_psd_blockswap.py` | tag 列表/group_index 改从 config 读 | 推理必需（用这两个脚本时） |

**不需要改**：`layerdiff3d.py` 的 `__init__`/forward 主体（组数全由 config 驱动）、`load_blend_parts`（用伪深度规避代码改动）、`dataset_layerdiff.py` 的组轮转逻辑、FSDP 存取逻辑。

**已识别的坑（写进实施 checklist）**：

- `load_blend_parts` 对**每个存在的部件 png** 都要求 `{tag}_depth.png` 存在（L91-92，否则 `__getitem__` 死循环重试）；伪深度脚本必须覆盖全部部件 png。
- `size_skip_tags`/`crop_tag_list` 不要用：crop 是组内所有部件 union，skip 会导致被 skip tag 的部件图被裁断 → 监督目标残缺（与 2D 数据集类语义不同）。
- `caption_list` 同时是可视化标注与文本条件；下划线 tag 直接进 CLIP 会产生非自然 token 切分，故做空格归一化。
- `empty_mask` 不进 loss——缺省 tag 的空帧是刻意的负监督，勿"顺手"用它 mask loss（那会破坏"缺席→空层"的学习信号）。
- 远端 `prepare_dataset.py` 的 `PART_MAP` 里 `hair_ornament` 键重复定义（后者覆盖前者）——新数据管线不复用该表，不受影响，但说明旧转换本身的映射质量仅近似（如 `cloak→objects`、`dress→bottomwear`）。

---

## B. 44 语义分组方案

### B.1 分组设计：推荐 2 组（沿用官方 body/head 双通路结构）

官方 2 组的本质是**多分辨率监督**：body 组 crop≈全身（部件在整图语境学习），head 组 crop=头部 union（部件在放大头部语境学习，等效分辨率提升 ~3×）。44 标签沿用同一结构收益最大、改动最小：

- 推理仍是 2 pass，2 阶段 head-crop 流程可直接复用；
- 每组样本数 = 总样本/2（而非 /3），347 小数据下每组监督密度更高；
- `group_embedding_num=(30,15)`，组数与官方一致。

### B.2 各组完整标签清单与维度

**group0 = body（30 槽 = 29 数据 tag + 1 合成 anchor `head`）**，组内顺序（即 `params` 行序 / 帧序 / caption 序）：

```yaml
tags:
  - hair_back          # ← back hair
  - hair_front         # ← front hair
  - hair_bangs         # ← front hair
  - hair_side_left     # ← front hair
  - hair_side_right    # ← front hair
  - head               # ← head（合成 anchor 层，见 B.3）
  - neck               # ← neck
  - neckwear           # ← neckwear
  - ribbon             # ← neckwear
  - topwear            # ← topwear
  - dress              # ← topwear
  - cloak              # ← topwear
  - sleeve_left        # ← topwear
  - sleeve_right       # ← topwear
  - hand_left          # ← handwear
  - hand_right         # ← handwear
  - glove_left         # ← handwear
  - glove_right        # ← handwear
  - weapon             # ← objects
  - accessory          # ← objects
  - skirt              # ← bottomwear
  - bottomwear         # ← bottomwear
  - legwear            # ← legwear
  - leg_left           # ← legwear
  - leg_right          # ← legwear
  - footwear_left      # ← footwear
  - footwear_right     # ← footwear
  - tail               # ← tail
  - wing_left          # ← wings
  - wing_right         # ← wings
```

**group1 = head（15 槽）**：

```yaml
tags:
  - face               # ← face
  - eye_left           # ← irides
  - eye_right          # ← irides
  - eyebrow_left       # ← eyebrow
  - eyebrow_right      # ← eyebrow
  - nose               # ← nose
  - mouth              # ← mouth
  - ear_left           # ← ears
  - ear_right          # ← ears
  - animal_ear_left    # ← ears
  - animal_ear_right   # ← ears
  - eyewear            # ← eyewear
  - hat                # ← headwear
  - headwear           # ← headwear
  - hair_ornament      # ← headwear
```

`group_embedding_num = (30, 15)`。

**组内顺序约束**：数据 yaml 的 `tags` 顺序、扩维脚本 params 行序、推理 `tag_groups` 顺序**三者必须逐位一致**；顺序本身不影响正确性（只要一致），上表按"上→下"语义排列便于审阅。

### B.3 `head` 合成 anchor tag（推荐，+1 槽）

**动机**：v3 推理用 body pass 的 'head' 输出定位头部裁剪区。44 tag 里没有"整体头"概念，若不加，head pass 的 crop 输入无可靠来源。

**数据侧合成**：`prepare_dataset_44.py` 把头域部件 `{face, ear_*, animal_ear_*, eye_*, eyebrow_*, nose, mouth, eyewear, hat, headwear, hair_ornament, hair_*}` 的 alpha 在 final_size 画布上 union 合成 `{prefix}_head.png`（裁到 union xyxy），`_ann.json['tag_info']['head']` 写 `exists`（≥1 头域部件存在时）与 union `xyxy`，并生成 `_head_depth.png`。

- 含 hair 是故意的：anchor 需要框住整个头量（含发），否则 `_crop_head` 扩边后仍会切掉高发/头饰。
- params 初始化 ← 旧 group0 'head' 行，语义吻合。
- **备选**：不合成、推理改用 `annotators/anime_face_detector` 检测脸框后按经验比例放大为头框（省 1 槽、数据侧少一步，但多一个推理依赖且对夸张发型不稳）。推荐合成方案。

### B.4 备选：3 组方案 `(22, 8, 15)`

若 30 帧 @1024 显存不足，退到此方案：

- **g0 upper（22）**：`hair_back, hair_front, hair_bangs, hair_side_left, hair_side_right, head, neck, neckwear, ribbon, topwear, dress, cloak, sleeve_left, sleeve_right, hand_left, hand_right, glove_left, glove_right, weapon, accessory, wing_left, wing_right`
- **g1 lower（8）**：`skirt, bottomwear, legwear, leg_left, leg_right, footwear_left, footwear_right, tail`
- **g2 head（15）**：同 B.2 group1
- linear 复制：g0、g1 ← 旧 g0；g2 ← 旧 g1。`tag_version` 同用 'v4' + `tag_groups`。
- 代价：推理 3 pass（+50% 时延）；每组样本密度降至 1/3；g1 的 crop（裙摆+腿脚+尾）收益有限。
- 切换成本低：只改扩维脚本输出与新 yaml 的 `mix_tag_groups`，数据侧零改动。

### B.5 分组原则说明

1. **crop 语境一致性**（首要）：同组 tag 的部件应位于相近区域——crop=union bbox 才有意义。所以 hair 五兄弟留 body（发长及腰，归 head 会把 head crop 撑回全身）；dress/cloak 主体在躯干归 body；weapon/accessory 伴随手部归 body。
2. **官方语义对齐**：官方 body 组含 hair/head/四肢装备/尾翼/物，head 组只含头面部件（headwear 归 head 而非 body，注意）——本方案完全沿用这一划分直觉，`hat` 并入 head 组（与 headwear 同语义）；`animal_ear` 归 head（头顶部件）。
3. **embedding 维度 = 组大小**（硬约束）：`group_embedding_num[i] == len(mix_tag_groups[i]['tags'])`，且 `params` 行序与 tags 顺序一一对应。
4. **组数权衡**：组数↑ → 单 pass 帧数↓（显存）、crop 更聚焦（细节）、但 pass 数↑（推理成本）+每组监督密度↓（数据效率）。2 组是官方验证过的甜点；仅在显存成为问题时升 3 组。
5. **稀疏 tag 不集中**：animal_ear（3%）等与高频 tag 同组可共享 crop 语境并借组内共同训练稳态——不为稀疏 tag 单设组。

---

## C. 数据准备步骤（远端执行）

1. `python prepare_dataset_44.py`：扫描 `datasets/images/*_fullbody.png` → 输出 `see-through/workspace/datasets/l2d_bodysamples_44/`：
   - 拷贝 fullbody + 全部存在的 `_{tag}.png`（原名原样）；
   - 每个部件 png 生成 `_{tag}_depth.png = 255 - alpha`；
   - 合成 `_head.png` + `_head_depth.png`，`_ann.json` 增补 `head` 条目；
   - 输出 `l2d_bodysamples_44.txt`（每行 `workspace/datasets/l2d_bodysamples_44/{prefix}.png`）。
2. `python training/scripts/expand_group_embeds.py`：v0.0.2 unet → `unet_group44/`（按 B.2/B.5 映射复制）。
3. 新增 `finetune_layerdiff3d_v002_44tag.yaml`（字段见 A.5）。

## D. 验证与测试步骤（禁正式训练）

1. **扩维脚本自检**：加载烘焙后 unet，断言 `group_embeds{,2}.{g}.params.shape[0] == len(tag_groups[g])`；抽 `eye_left` 行 == 旧 `irides` 行（两套 embed 分别断言）；`linear` 逐组相等。
2. **数据集冒烟**（无 GPU）：实例化 `LayerDiffDataset3D(**new dataset_args)` 取数样本 → `img_list.shape == (30|15, 4, 1024, 1024)`、`caption_list` 数与序正确、`empty_mask` 合理、`group_index` 轮转。
3. **训练冒烟**：复用现有 `max_train_steps: 3` 套路（双机 FSDP 各 rank 跑通）→ loss 有限、checkpoint 分片保存、**记录 step_peak 显存**——若 OOM 即切 B.4 三组的唯一决策点。
4. **推理回归基线**：未训练前先对扩维 ckpt 跑 `apply_layerdiff`——复制映射 + 保留 linear 时应得到近似 v0.0.2 的分层输出（父类克隆预期：eye_l/r 输出相同属正常）。
5. **训练后验收**：同角色跑推理，检查 ① split tag 分化（eye_l/r 各自成层）；② 新语义生效（weapon/dress/animal_ear 独立层）；③ 缺席 tag → 空层；④ head anchor crop 合理。

## E. 风险清单（汇总）

| 风险 | 等级 | 缓解 |
|---|---|---|
| group0 30 帧 @1024 显存超限 | 高 | 冒烟先测 peak；超限切 `(22,8,15)` 三组 |
| `init_weights` 默认 zero 掉复制来的 linear | 中 | 加 `zero_group_embeds` 开关，微调传 False |
| 稀疏 tag（animal_ear 3%）欠训 | 中 | 父类暖启动 + 验收抽测 + 必要时 sample_rate 加权 |
| `accessory` 兜底层吞掉细粒度语义 | 中 | 数据侧抽检；必要时标注修正 |
| 推理硬编码列表未同步 → 输出错位 | 中 | `tag_version='v4'` + `tag_groups` 配置化，统一从 config 读 |
| 缺 `_depth.png` → dataset 死循环 | 高 | 伪深度脚本覆盖全量部件，跑前校验文件对 |
| 旧 24 ckpt 无法 resume 进新模型 | 低（预期） | 文档化；保留 v0.0.2 作旧接口模型 |
| 组间轮转在多 worker 下不均匀 | 低 | `ngroup_iters` 机制本身随机友好；如需严格均衡可固定 worker 数 |

## 附录：44 tag → 24 父类映射表（初始化用）

| 44 tag | 父类（旧组/行） |
|---|---|
| hair_back | group0 'back hair' |
| hair_front, hair_bangs, hair_side_left, hair_side_right | group0 'front hair' |
| head（合成） | group0 'head' |
| neck | group0 'neck' |
| neckwear, ribbon | group0 'neckwear' |
| topwear, dress, cloak, sleeve_left, sleeve_right | group0 'topwear' |
| skirt, bottomwear | group0 'bottomwear' |
| legwear, leg_left, leg_right | group0 'legwear' |
| footwear_left, footwear_right | group0 'footwear' |
| hand_left, hand_right, glove_left, glove_right | group0 'handwear' |
| weapon, accessory | group0 'objects' |
| tail | group0 'tail' |
| wing_left, wing_right | group0 'wings' |
| face | group1 'face' |
| eye_left, eye_right | group1 'irides' |
| eyebrow_left, eyebrow_right | group1 'eyebrow' |
| nose | group1 'nose' |
| mouth | group1 'mouth' |
| ear_left, ear_right, animal_ear_left, animal_ear_right | group1 'ears' |
| eyewear | group1 'eyewear' |
| hat, headwear, hair_ornament | group1 'headwear' |

未被使用的旧槽：`eyewhite`、`eyelash`、`earwear`（其功能被 face/eye_* 层吸收，预期内）。
