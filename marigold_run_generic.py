#!/usr/bin/env python3
"""See-through fallback：LayerDiff 图层已生成后，补跑 Marigold 深度 + PSD 组装。
用法: python marigold_run_generic.py <图片绝对路径> <stem>"""
import sys
import os

sys.path.insert(0, '/workspace/see-through')
os.chdir('/workspace/see-through')

from utils.inference_utils import apply_marigold, further_extr  # noqa: E402

img = sys.argv[1]
stem = sys.argv[2]

apply_marigold(
    img,
    '24yearsold/seethroughv0.0.1_marigold',
    save_dir='workspace/layerdiff_output_blockswap',
    seed=42,
    resolution=768,
    num_inference_steps=-1,
    group_offload=False,
)
further_extr(
    f'workspace/layerdiff_output_blockswap/{stem}',
    rotate=False,
    save_to_psd=True,
    tblr_split=False,
)
print('FALLBACK_DONE')
