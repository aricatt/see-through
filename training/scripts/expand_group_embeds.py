"""v0.0.2 (13,11) group_embeds → v4 44-tag 扩维烘焙脚本（离线，一次性）

原理：
- group_embeds{,2}.{g}.params 按 batch 维位置寻址（第 i 行 ↔ 组内第 i 个 tag）
- 新 tag 按其 24 父类复制旧 params 行（PARENT_MAP，44 tag 全部有父类；split 兄弟
  如 eye_left/right 从同一父行克隆起步，靠 caption + 监督分化）
- group_embeds{,2}.{g}.linear 是组内共享变换，按 LINEAR_SRC_GROUP 整组继承
- 其余权重（unet 主体）逐键原样拷贝
- config 更新：group_embedding_num、tag_version='v4'、tag_groups（组内 tag 顺序，
  供推理 v4 分支读取；ConfigMixin 透传非 __init__ 字段，from_pretrained 可读回）

输出：save_pretrained 目录（diffusion_pytorch_model.safetensors + config.json），
之后 from_pretrained 无缝加载，shape 完全对齐，无需 strict=False。

用法（远端 venv）：
    python training/scripts/expand_group_embeds.py \
        --src /home/ari/seethru_train/models/hf/hub/models--layerdifforg--seethroughv0.0.2_layerdiff3d/snapshots/4477e6ce529bc6a141732e1a55a4932176db3b89/unet \
        --dst /home/ari/seethru_train/models/unet_group44
"""
import argparse
import os.path as osp
import sys

import torch

sys.path.append(osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))), 'common'))

from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel

# ---------------------------------------------------------------------------
# v0.0.2 旧组 tag 顺序（即 params 行序，与 cache_tag_embeds / 官方 mix_tag_groups 一致）
# ---------------------------------------------------------------------------
OLD_GROUPS = [
    ['front hair', 'back hair', 'head', 'neck', 'neckwear', 'topwear', 'handwear',
     'bottomwear', 'legwear', 'footwear', 'tail', 'wings', 'objects'],                    # 13
    ['headwear', 'face', 'irides', 'eyebrow', 'eyewhite', 'eyelash', 'eyewear',
     'ears', 'earwear', 'nose', 'mouth'],                                               # 11
]

# ---------------------------------------------------------------------------
# 新组定义（组内顺序 = params 行序 = 数据 yaml tags 序 = 推理 tag_groups 序，必须一致）
# ---------------------------------------------------------------------------
HEAD_TAGS = [
    'face', 'eye_left', 'eye_right', 'eyebrow_left', 'eyebrow_right',
    'nose', 'mouth', 'ear_left', 'ear_right', 'animal_ear_left', 'animal_ear_right',
    'eyewear', 'hat', 'headwear', 'hair_ornament',
]

UPPER_TAGS = [
    'hair_back', 'hair_front', 'hair_bangs', 'hair_side_left', 'hair_side_right',
    'head', 'neck', 'neckwear', 'ribbon',
    'topwear', 'dress', 'cloak', 'sleeve_left', 'sleeve_right',
    'hand_left', 'hand_right', 'glove_left', 'glove_right',
    'weapon', 'accessory',
    'wing_left', 'wing_right',
]

LOWER_TAGS = [
    'skirt', 'bottomwear', 'legwear', 'leg_left', 'leg_right',
    'footwear_left', 'footwear_right', 'tail',
]

# 方案 A（推荐，2 组）：group0=body 30 槽（29 数据 tag + 合成 anchor 'head'），
# 组内顺序 = plan B.2（与数据 yaml mix_tag_groups 逐位一致）
TAG_GROUPS_2 = [
    [
        'hair_back', 'hair_front', 'hair_bangs', 'hair_side_left', 'hair_side_right',
        'head', 'neck', 'neckwear', 'ribbon',
        'topwear', 'dress', 'cloak', 'sleeve_left', 'sleeve_right',
        'hand_left', 'hand_right', 'glove_left', 'glove_right',
        'weapon', 'accessory',
        'skirt', 'bottomwear', 'legwear', 'leg_left', 'leg_right',
        'footwear_left', 'footwear_right', 'tail', 'wing_left', 'wing_right',
    ],                            # 30
    HEAD_TAGS,                    # 15
]
LINEAR_SRC_2 = [0, 1]

# 方案 B（兜底，3 组）：仅当 30 帧 @1024 冒烟 step_peak 超限时切换
TAG_GROUPS_3 = [
    UPPER_TAGS,   # 22
    LOWER_TAGS,   # 8
    HEAD_TAGS,    # 15
]
LINEAR_SRC_3 = [0, 0, 1]

SCHEMES = {
    '2group': (TAG_GROUPS_2, LINEAR_SRC_2),
    '3group': (TAG_GROUPS_3, LINEAR_SRC_3),
}

# ---------------------------------------------------------------------------
# 44 tag → 旧 24 父类映射（新 tag: (旧组 index, 旧 tag 名)）
# ---------------------------------------------------------------------------
PARENT_MAP = {
    'hair_back': (0, 'back hair'),
    'hair_front': (0, 'front hair'), 'hair_bangs': (0, 'front hair'),
    'hair_side_left': (0, 'front hair'), 'hair_side_right': (0, 'front hair'),
    'head': (0, 'head'),
    'neck': (0, 'neck'),
    'neckwear': (0, 'neckwear'), 'ribbon': (0, 'neckwear'),
    'topwear': (0, 'topwear'), 'dress': (0, 'topwear'), 'cloak': (0, 'topwear'),
    'sleeve_left': (0, 'topwear'), 'sleeve_right': (0, 'topwear'),
    'hand_left': (0, 'handwear'), 'hand_right': (0, 'handwear'),
    'glove_left': (0, 'handwear'), 'glove_right': (0, 'handwear'),
    'weapon': (0, 'objects'), 'accessory': (0, 'objects'),
    'skirt': (0, 'bottomwear'), 'bottomwear': (0, 'bottomwear'),
    'legwear': (0, 'legwear'), 'leg_left': (0, 'legwear'), 'leg_right': (0, 'legwear'),
    'footwear_left': (0, 'footwear'), 'footwear_right': (0, 'footwear'),
    'tail': (0, 'tail'),
    'wing_left': (0, 'wings'), 'wing_right': (0, 'wings'),
    'face': (1, 'face'),
    'eye_left': (1, 'irides'), 'eye_right': (1, 'irides'),
    'eyebrow_left': (1, 'eyebrow'), 'eyebrow_right': (1, 'eyebrow'),
    'nose': (1, 'nose'), 'mouth': (1, 'mouth'),
    'ear_left': (1, 'ears'), 'ear_right': (1, 'ears'),
    'animal_ear_left': (1, 'ears'), 'animal_ear_right': (1, 'ears'),
    'eyewear': (1, 'eyewear'),
    'hat': (1, 'headwear'), 'headwear': (1, 'headwear'), 'hair_ornament': (1, 'headwear'),
}


def main():
    ap = argparse.ArgumentParser(description='Expand v0.0.2 group_embeds to 44-tag v4 groups')
    ap.add_argument('--src', required=True, help='v0.0.2 unet 目录（from_pretrained 可读）')
    ap.add_argument('--dst', required=True, help='输出 unet 目录')
    ap.add_argument('--scheme', choices=list(SCHEMES), default='2group')
    args = ap.parse_args()

    tag_groups, linear_src = SCHEMES[args.scheme]

    # 完备性检查：新组所有 tag 必须都有父类
    for g, tags in enumerate(tag_groups):
        for t in tags:
            assert t in PARENT_MAP, f'tag {t} (group {g}) 缺父类映射'
        assert len(tags) == len(set(tags)), f'group {g} 有重复 tag'

    print(f'加载源 unet: {args.src}')
    old = UNetFrameConditionModel.from_pretrained(args.src)
    old_sd = {k: v.clone() for k, v in old.state_dict().items()}
    old_num = list(old.config.group_embedding_num)
    assert old_num == [len(g) for g in OLD_GROUPS], \
        f'源 group_embedding_num={old_num} 与预期 {[len(g) for g in OLD_GROUPS]} 不符'

    # 新 config：非 __init__ 字段（tag_groups）由 ConfigMixin 透传，可安全写入
    cfg = dict(old.config)
    del old
    cfg['group_embedding_num'] = [len(g) for g in tag_groups]
    cfg['tag_version'] = 'v4'
    cfg['tag_groups'] = [list(g) for g in tag_groups]

    print(f'构建新模型: group_embedding_num={cfg["group_embedding_num"]}, tag_version=v4')
    new = UNetFrameConditionModel.from_config(cfg)

    # 合并 state dict：非 group_embeds 键原样拷贝；group_embeds{,2} 重新组装
    merged = {k: v for k, v in old_sd.items() if not k.startswith('group_embeds')}
    for g, tags in enumerate(tag_groups):
        for emb in ('group_embeds.', 'group_embeds2.'):
            rows = []
            for t in tags:
                og, ot = PARENT_MAP[t]
                rows.append(old_sd[f'{emb}{og}.params'][OLD_GROUPS[og].index(ot)])
            merged[f'{emb}{g}.params'] = torch.stack(rows).to(old_sd[f'{emb}0.params'].dtype)
            sg = linear_src[g]
            merged[f'{emb}{g}.linear.weight'] = old_sd[f'{emb}{sg}.linear.weight']
            merged[f'{emb}{g}.linear.bias'] = old_sd[f'{emb}{sg}.linear.bias']
    del old_sd

    missing, unexpected = new.load_state_dict(merged, strict=False)
    assert not missing and not unexpected, f'missing={missing[:5]} unexpected={unexpected[:5]}'

    print(f'保存: {args.dst}')
    new.save_pretrained(args.dst)
    del new

    # ------------------------- 自检 -------------------------
    print('自检：from_pretrained 读回 ...')
    chk = UNetFrameConditionModel.from_pretrained(args.dst)
    chk_sd = chk.state_dict()
    ref = UNetFrameConditionModel.from_pretrained(args.src)
    ref_sd = ref.state_dict()

    assert list(chk.config.group_embedding_num) == [len(g) for g in tag_groups]
    assert chk.config.get('tag_version') == 'v4', 'tag_version 未透传'
    assert [list(g) for g in chk.config.get('tag_groups')] == [list(g) for g in tag_groups], \
        'tag_groups 未透传（config 自定义字段丢失）'

    for g, tags in enumerate(tag_groups):
        for emb in ('group_embeds.', 'group_embeds2.'):
            p = chk_sd[f'{emb}{g}.params']
            assert p.shape[0] == len(tags), f'{emb}{g}.params shape {tuple(p.shape)} != {len(tags)}'
            # 逐行验证：新行 == 父类旧行
            for i, t in enumerate(tags):
                og, ot = PARENT_MAP[t]
                ref_row = ref_sd[f'{emb}{og}.params'][OLD_GROUPS[og].index(ot)]
                assert torch.equal(p[i].float(), ref_row.float()), \
                    f'{emb}{g}.params[{i}] ({t}) != 父类 {ot} 行'
            # linear 逐组相等
            sg = linear_src[g]
            for wn in ('linear.weight', 'linear.bias'):
                assert torch.equal(chk_sd[f'{emb}{g}.{wn}'].float(), ref_sd[f'{emb}{sg}.{wn}'].float()), \
                    f'{emb}{g}.{wn} != 源组 {sg}'

    # 关键 spot-check：eye_left 行 == 旧 irides 行（两套 embed 已在上环验证，再显式打印）
    hg = len(tag_groups) - 1
    gi = tag_groups[hg].index('eye_left')
    ri = OLD_GROUPS[1].index('irides')
    for emb in ('group_embeds.', 'group_embeds2.'):
        assert torch.equal(chk_sd[f'{emb}{hg}.params'][gi].float(),
                           ref_sd[f'{emb}1.params'][ri].float())
    print('  eye_left params == 旧 irides params: OK (group_embeds + group_embeds2)')

    # 非 group_embeds 权重全量相等
    n_diff = 0
    for k in ref_sd:
        if k.startswith('group_embeds'):
            continue
        if not torch.equal(chk_sd[k].float(), ref_sd[k].float()):
            n_diff += 1
            print(f'  DIFF: {k}')
    assert n_diff == 0, f'{n_diff} 个非 group 键不一致'

    print(f'自检全部通过。group_embedding_num={list(chk.config.group_embedding_num)}, '
          f'tag_groups={[len(g) for g in chk.config.tag_groups]}')


if __name__ == '__main__':
    main()
