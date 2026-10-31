#!/usr/bin/env python3
"""See-through 44 语义数据集准备脚本（347 角色 → LayerDiff 3D v4 训练格式）

与 prepare_dataset.py（44→24 合并版）不同：不做部件合并，44 tag 原样透传。

- 输入: {src_dir}/*_fullbody.png + *_ann.json + *_fullbody_{part}.png
        （ann['tag_info'] 含 44 部件 tag 的 exists/xyxy；部件图已按 xyxy 裁好）
- 输出: {out_dir}/
        {prefix}.png                       完整立绘（拷贝）
        {prefix}_{tag}.png                 各存在部件图（原样拷贝）
        {prefix}_{tag}_depth.png           伪深度 = 255 - alpha（load_blend_parts 硬性需要，
                                           覆盖全部存在的部件 png，否则 __getitem__ 死循环/报错）
        {prefix}_head.png + _head_depth.png 合成 anchor 层（头域部件 union，含 hair）
        {prefix}_ann.json                  原 tag_info + 增补 'head' 条目
        {list_path}                        样本列表（repo 相对路径）

用法（70bc 远端）：
    python training/scripts/prepare_dataset_44.py
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

SRC_DIR = Path("/home/ari/seethru_train/datasets/images")
OUT_DIR = Path("/home/ari/seethru_train/see-through/workspace/datasets/l2d_bodysamples_44")
LIST_PATH = Path("/home/ari/seethru_train/see-through/workspace/datasets/l2d_bodysamples_44.txt")
LIST_REL_DIR = "workspace/datasets/l2d_bodysamples_44"

# 合成 anchor 'head' 的头域部件集合（含 hair：anchor 需要框住整个头量，
# 否则推理 _crop_head 扩边后仍会切掉高发/头饰）
HEAD_PARTS = {
    'face', 'nose', 'mouth',
    'eye_left', 'eye_right', 'eyebrow_left', 'eyebrow_right',
    'ear_left', 'ear_right', 'animal_ear_left', 'animal_ear_right',
    'eyewear', 'hat', 'headwear', 'hair_ornament',
    'hair_back', 'hair_front', 'hair_bangs', 'hair_side_left', 'hair_side_right',
}


def merge_xyxy(boxes):
    if not boxes:
        return None
    return [int(min(b[0] for b in boxes)), int(min(b[1] for b in boxes)),
            int(max(b[2] for b in boxes)), int(max(b[3] for b in boxes))]


def write_pseudo_depth(part_png: Path):
    """depth = 255 - alpha（约定与 gen_depth.py 一致；load_blend_parts 读取时再做 255 - x 还原）"""
    img = np.array(Image.open(part_png).convert("RGBA"))
    depth = (255 - img[..., 3]).astype(np.uint8)
    Image.fromarray(depth).save(part_png.with_name(part_png.stem + "_depth.png"))


def overlay(canvas, layer, xyxy):
    """把局部部件图按 xyxy 叠进 final_size 画布（alpha 取 max，RGB 取更不透明侧）"""
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    lh, lw = layer.shape[:2]
    rh, rw = (y2 - y1), (x2 - x1)
    if (lh, lw) != (rh, rw):
        layer = np.array(Image.fromarray(layer).resize((rw, rh), Image.LANCZOS))
    region = canvas[y1:y2, x1:x2]
    a_src = layer[..., 3:4]
    a_dst = region[..., 3:4]
    new_a = np.maximum(a_dst, a_src)
    mask = a_src > a_dst
    rgb = region[..., :3].copy()
    rgb[mask[..., 0]] = layer[..., :3][mask[..., 0]]
    canvas[y1:y2, x1:x2, :3] = rgb
    canvas[y1:y2, x1:x2, 3] = new_a[..., 0]


def convert_one(fullbody_png: Path, out_dir: Path, stats: dict):
    prefix = fullbody_png.stem
    annp = fullbody_png.with_name(prefix + '_ann.json')
    if not annp.exists():
        stats['no_ann'] += 1
        return None
    ann = json.loads(annp.read_text(encoding='utf-8'))
    final_size = ann.get('final_size')
    if not final_size:
        stats['no_finalsize'] += 1
        return None
    tag_info = ann.get('tag_info', {})

    # 1) 拷贝全部存在部件 png + 伪深度
    H, W = final_size
    copied = []
    for tag, info in tag_info.items():
        if not info.get('exists') or not info.get('xyxy'):
            continue
        part_png = fullbody_png.with_name(f"{prefix}_{tag}.png")
        if not part_png.exists():
            stats['missing_part_png'] += 1
            print(f"  [warn] {prefix}_{tag}.png 缺失但 ann 标 exists，跳过该部件")
            info['exists'] = False
            info.pop('xyxy', None)
            continue
        shutil.copy2(part_png, out_dir / part_png.name)
        write_pseudo_depth(out_dir / part_png.name)
        copied.append(tag)
    if not copied:
        stats['no_valid_tag'] += 1
        return None

    # 2) 合成 head anchor 层：头域部件 union → 裁到 union xyxy
    head_items = [(t, i) for t, i in tag_info.items()
                  if t in HEAD_PARTS and i.get('exists') and i.get('xyxy')]
    if head_items:
        canvas = np.zeros((H, W, 4), dtype=np.uint8)
        for t, info in head_items:
            layer = np.array(Image.open(fullbody_png.with_name(f"{prefix}_{t}.png")).convert('RGBA'))
            overlay(canvas, layer, info['xyxy'])
        hx1, hy1, hx2, hy2 = merge_xyxy([i['xyxy'] for _, i in head_items])
        head_png = out_dir / f"{prefix}_head.png"
        Image.fromarray(canvas[hy1:hy2, hx1:hx2]).save(head_png)
        write_pseudo_depth(head_png)
        tag_info['head'] = {'valid': True, 'exists': True, 'xyxy': [hx1, hy1, hx2, hy2]}
        stats['head_synth'] += 1
    else:
        stats['no_head'] += 1

    # 3) 改写 ann.json（原 tag_info + head 条目）
    new_ann = {'final_size': final_size, 'tag_info': tag_info}
    (out_dir / f"{prefix}_ann.json").write_text(
        json.dumps(new_ann, ensure_ascii=False, indent=2), encoding='utf-8')

    # 4) 拷贝完整立绘
    shutil.copy2(fullbody_png, out_dir / fullbody_png.name)

    stats['converted'] += 1
    stats['parts_total'] += len(copied)
    return prefix


def main():
    ap = argparse.ArgumentParser(description='44-tag 数据集转换（部件原样透传 + head 合成 + 伪深度）')
    ap.add_argument('--src_dir', default=str(SRC_DIR))
    ap.add_argument('--out_dir', default=str(OUT_DIR))
    ap.add_argument('--list_path', default=str(LIST_PATH))
    ap.add_argument('--list_rel_dir', default=LIST_REL_DIR,
                    help='写入 txt 的 repo 相对目录前缀')
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {'converted': 0, 'no_ann': 0, 'no_finalsize': 0, 'no_valid_tag': 0,
             'missing_part_png': 0, 'head_synth': 0, 'no_head': 0, 'parts_total': 0}
    fullbodies = sorted(src_dir.glob('*_fullbody.png'))
    print(f"发现 {len(fullbodies)} 个完整立绘")
    lines = []
    skipped = []
    for fp in fullbodies:
        prefix = convert_one(fp, out_dir, stats)
        if prefix:
            lines.append(f"{args.list_rel_dir}/{prefix}.png")
        else:
            skipped.append(fp.name)
    Path(args.list_path).write_text('\n'.join(lines) + '\n', encoding='utf-8')

    # 收尾校验：输出目录里每个部件 png 必须有配对 _depth.png
    bad = [p.name for p in out_dir.glob('*_fullbody_*.png')
           if not p.stem.endswith('_depth')
           and not p.with_name(p.stem + '_depth.png').exists()]
    print(f"转换完成: {stats}")
    print(f"样本列表: {args.list_path} ({len(lines)} 行)")
    print(f"跳过: {len(skipped)} -> {skipped[:10]}")
    if bad:
        print(f"[FATAL] {len(bad)} 个部件图缺 _depth.png: {bad[:10]}")
        raise SystemExit(1)
    print("depth 覆盖校验: OK（全部部件图均有配对 _depth.png）")


if __name__ == '__main__':
    main()
