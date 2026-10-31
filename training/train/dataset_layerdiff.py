import os.path as osp
from typing import Dict, List, Union, Tuple
import random
import traceback
import time

import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
from einops import rearrange

from utils.cv import rle2mask, pad_rgb, center_square_pad_resize, img_alpha_blending, make_random_irregular_mask, make_random_rectangle_mask, random_pad_img, random_crop
from utils.torch_utils import img2tensor, tensor2img
from utils.io_utils import json2dict, load_exec_list, find_all_imgs, imglist_from_dir_or_flist


MINIMUM_VISIBLE_ALPHA = 25

exclude_cls = \
{
    '1girl',
    'smile',
    'simple_background',
    'white_background',
    'solo',
    'closed_mouth',
    'looking_at_viewer',
    'standing',
    'full_body',
    'virtual_youtuber',
    'tachi-e',
    'elf',
    'transparent_background',
    'blush',
    'straight-on',
    'looking_to_the_side',
    'expressionless',
    'holding',
}


def filter_tags(tags):
    tag_filtered = []
    for t in tags.split(','):
        t = t.strip().lower()
        if 'commentary' in t or 'background' in t or t in exclude_cls:
            continue
        tag_filtered.append(t)
    return ','.join(tag_filtered)


def xyxy_from_taglist(ann, tag_list):
    h, w = ann['final_size']
    minx = miny = max(h, w)
    maxx = maxy = 0
    is_empty = True
    for t in tag_list:
        if t not in ann['tag_info']:
            continue
        if not ann['tag_info'][t]['exists']:
            continue
        xyxy = ann['tag_info'][t]['xyxy']
        minx = min(minx, xyxy[0])
        miny = min(miny, xyxy[1])
        maxx = max(maxx, xyxy[2])
        maxy = max(maxy, xyxy[3])
        is_empty = False
    if is_empty:
        return [0, 0, w, h]
    return [minx, miny, maxx, maxy]


def load_blend_parts(
        srcp: str, 
        tag_list, ann=None, 
        keep_loaded_parts=False,
        keep_depth=False
    ):
    part_list = []
    if ann is None:
        ann = osp.splitext(srcp)[0] + '_ann.json'
        ann = json2dict(ann)

    h, w = ann['final_size']
    for t in tag_list:
        p = osp.splitext(srcp)[0] + '_' + t + '.png'
        if not osp.exists(p):
            continue
        depthp = osp.splitext(srcp)[0] + '_' + t + '_depth.png'
        depth = 255 - np.array(Image.open(depthp))
        img = np.array(Image.open(p))
        xyxy = ann['tag_info'][t]['xyxy']
        part_list.append({'depth': depth, 'img': img, 'tag': t, 'xyxy': xyxy})

    if len(part_list) == 0:
        img = np.zeros((h, w, 4), dtype=np.uint8)
        minx = miny = maxx = maxy = 0
        rst = {'is_empty': True, 'img': img}
    else:
        rst = img_alpha_blending(part_list, final_size=ann['final_size'], output_type='dict')
        rst['is_empty'] = False

    if keep_loaded_parts:
        for part in part_list:
            img = np.zeros((h, w, 4), dtype=np.uint8)
            xyxy = part['xyxy']
            img[xyxy[1]: xyxy[3], xyxy[0]: xyxy[2]] = part['img']
            part['img'] = img
            depth = part.pop('depth')
            if keep_depth:
                part['depth'] = np.zeros((h, w), dtype=np.uint8)
                part['depth'][xyxy[1]: xyxy[3], xyxy[0]: xyxy[2]] = 255 - depth
        rst['part_list'] = part_list
            
    return rst


def load_sample_list(src_list):
    sample_list = []
    if not isinstance(src_list, List):
        src_list = [src_list]
    for src in src_list:
        label_dir = None
        label_list = None
        if isinstance(src, str):
            lst = imglist_from_dir_or_flist(src)
        else:
            assert isinstance(src, Dict)
            lst = imglist_from_dir_or_flist(src['sample_list'])
            if 'sample_rate' in src:
                lst = lst * src['sample_rate']
            if 'label_dir' in src:
                label_dir = src['label_dir']
            if 'label_list' in src:
                label_list = load_exec_list(src['label_list'])
        if label_dir is not None:
            for imgp in lst:
                sample_list.append({'img': imgp, 'label_dir': label_dir})
        elif label_list is not None:
            for imgp, labelp in zip(lst, label_list):
                sample_list.append({'img': imgp, 'label': labelp})
        else:
            for imgp in lst:
                sample_list.append({'img': imgp})
    return sample_list



def tag_exists(tag: str, info):
    if tag in info and 'xyxy' in info[tag]:
        return True
    return False



class LayerDiffDataset(Dataset):
    def __init__(
        self, 
        src_list: Union[str, Dict, List],
        target_size: Union[int, Tuple] = 1024,
        random_flip=0.,
        pixel_mean: List[float] = [0., 0., 0., 0.],
        pixel_std: List[float] = [255., 255., 255., 255.],       # SAM input std
        tag_list: List[str] = None,
        contrl_signals: str = None,
        random_pad_prob: bool = 0.0,
        load_ojects: bool = True,
        mask_cond: bool = True,
        random_aug_mask: float = 0.,
        mix_tag_prob: float = 0.,
        mix_tag_groups: list = None,
        target_size_groups: list = None,
        **kwargs) -> None:

        super().__init__()

        self.target_size = target_size
        self.tag_list = tag_list
        self.sample_list = load_sample_list(src_list)
        
        self.random_flip = random_flip
        self.random_aug_mask = random_aug_mask

        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

        self.random_pad_prob = random_pad_prob
        self.load_objects = load_ojects
        self.contrl_signals = contrl_signals
        self.mask_cond = mask_cond
        self.mix_tag_prob = mix_tag_prob
        self.mix_tag_groups = mix_tag_groups
        if target_size_groups is None:
            target_size_groups = []
        self.target_size_groups = target_size_groups

    def __len__(self):
        return len(self.sample_list)


    def random_aug_fullbody(self, img_rgba):
        img, alpha = img_rgba[..., :3], img_rgba[..., -1]
        h, w = img.shape[:2]
        if random.random() < 0.5:
            morph_ksize = random.randint(0, 7)
            if morph_ksize != 0:
                morph_ksize = morph_ksize * 2 + 1
                morph_func = cv2.dilate if random.random() > 0.5 else cv2.erode
                morph_kernel = cv2.getStructuringElement(
                    random.choice([cv2.MORPH_ELLIPSE, cv2.MORPH_RECT]), (morph_ksize, morph_ksize))
                alpha = morph_func(alpha, morph_kernel)
        else:
            if random.random() < 0.5:
                ir_mask = make_random_irregular_mask((h, w))
            else:
                ir_mask = make_random_rectangle_mask((h, w))
            alpha = ((1 - ir_mask) * alpha).astype(np.uint8)
        img_rgba = np.concatenate([img, alpha[..., None]], axis=2)
        return img_rgba

    def get_sample(self, index: int):

        def _aug_size_transform(img):
            if flip_tb:
                img = img[::-1]
            if flip_lr:
                img = img[:, ::-1]
            img = center_square_pad_resize(img, target_size=self.target_size)
            return img

        flip_lr = flip_tb = False
        if self.random_flip > 0 and random.random() < self.random_flip:
            flip_lr = True

        sample = self.sample_list[index]
        imgp = sample['img']
        imgname = osp.join(osp.dirname(imgp), osp.splitext(osp.basename(imgp))[0])
        annp = imgname + '_ann.json'
        mask_annp = imgname + '.json'
        valid_tag_list = []

        fullpage = np.array(Image.open(imgp))

        footwear_exists = False
        if osp.exists(annp):
            ann = json2dict(annp)
            tag_infos = ann['tag_info']
            
            for k, v in tag_infos.items():
                if not self.load_objects and k == 'objects':
                    continue
                if v['exists']:
                    v['tag'] = k
                    if k == 'footwear':
                        footwear_exists = True
                    valid_tag_list.append(v)

        load_mix_tags = False

        is_fullbody = False

        if len(valid_tag_list) == 0:
            valid_tag_list.append({'tag': ''})
        target_tag = None
        if len(valid_tag_list) > 0:
            target_tag = random.choice(valid_tag_list)
        else:
            raise Exception(f'No valid tag found!')
        
        tag = target_tag['tag']
        is_fullbody = tag == ''

        if not is_fullbody:
            imgp = imgname + '_' + target_tag['tag'] + '.png'
            tag_img = np.array(Image.open(imgp))
            captionp = imgname + '_' + target_tag['tag'] + '.txt'
            tag_idx = self.tag_list.index(tag)
            target_size_group = random.choice([tag] + [tp for tp in self.target_size_groups if tag in tp])
            if not isinstance(target_size_group, str):
                assert isinstance(target_size_group, list)
                tx1, ty1, tx2, ty2 = target_tag['xyxy']
                _tag_img = np.zeros(ann['final_size'] + [4], dtype=np.uint8)
                _tag_img[ty1: ty2, tx1: tx2] = tag_img
                target_tag['xyxy'] = tx1, ty1, tx2, ty2 = xyxy_from_taglist(ann, target_size_group)
                tag_img = _tag_img[ty1: ty2, tx1: tx2].copy()
                # tx1, ty1, tx2, ty2 = target_tag['xyxy']
                # tag_img = 
            tx1, ty1, tx2, ty2 = target_tag['xyxy']
            if self.mask_cond:
                visible_mask = rle2mask(json2dict(mask_annp)[tag_idx])[ty1: ty2, tx1: tx2, None].astype(np.uint8) * 255
        else:
            target_xyxy = cv2.boundingRect(cv2.findNonZero(fullpage[..., -1]))
            x1, y1, x2, y2 = target_xyxy
            x2 += x1
            y2 += y1
            target_tag['xyxy'] = [x1, y1, x2 , y2]
            tag_img = fullpage[y1: y2, x1: x2].copy()
            if self.mask_cond:
                visible_mask = (tag_img[..., [-1]] > MINIMUM_VISIBLE_ALPHA).astype(np.uint8) * 255
            captionp = imgname + '.txt'

        if osp.exists(captionp):
            with open(captionp, 'r', encoding='utf8') as f:
                caption = f.read()
            caption = filter_tags(caption)
        else:
            if not footwear_exists and tag == 'legwear':
                caption = 'legwear,footwear'
            else:
                caption = tag

        tx1, ty1, tx2, ty2 = target_tag['xyxy']
        random_pad = random.random() < self.random_pad_prob

        if random_pad:
            fh, fw = fullpage.shape[:2]
            tmax, lmax = ty1, tx1
            rmax = fw - tx2 - 1
            bmax = fh - ty2 - 1
            tag_img, (t, b, l, r) = random_pad_img(tag_img, tmax, bmax, lmax, rmax)
            if t > 0 or b > 0 or l > 0 or r > 0:
                if self.mask_cond:
                    visible_mask = cv2.copyMakeBorder(visible_mask[..., 0], t, b, l, r, borderType=cv2.BORDER_CONSTANT, value=0)[..., None]
            tx1 -= l
            ty1 -= t
            th, tw = tag_img.shape[:2]
            fullpage = fullpage[ty1: ty1 + th, tx1: tx1 + tw].copy()
            if load_mix_tags:
                tag_img = blended_rst['img'][ty1: ty1 + th, tx1: tx1 + tw].copy()
        else:
            fullpage = fullpage[ty1: ty2, tx1: tx2].copy()

        tag_img = _aug_size_transform(tag_img)
        fullpage = _aug_size_transform(fullpage)

        if is_fullbody and random.random() < self.random_aug_mask and self.mask_cond:
            fullpage = self.random_aug_fullbody(fullpage)
            
        if self.mask_cond:
            visible_mask = _aug_size_transform(visible_mask[..., 0])[..., None]
            cond_fullpage_visble_mask = (fullpage[..., [-1]] > MINIMUM_VISIBLE_ALPHA).astype(np.float32)
        else:
            cond_fullpage_visble_mask = np.ones_like(fullpage[..., [-1]]).astype(np.float32)

        cond_fullpage = np.concatenate([fullpage[..., :3] * cond_fullpage_visble_mask, cond_fullpage_visble_mask * 255], axis=2)
        cond_fullpage = (np.concatenate([cond_fullpage_visble_mask, pad_rgb(cond_fullpage)], axis=2) * 255).astype(np.uint8)

        if self.mask_cond:
            if is_fullbody:
                cond_tag_img = cond_fullpage.copy()
            else:
                cond_tag_img = tag_img * (visible_mask.astype(np.float32) / 255.)
                cond_tag_img[..., [-1]] = visible_mask.astype(np.float32)
                cond_tag_img = np.concatenate([visible_mask, (pad_rgb(cond_tag_img) * 255).astype(np.uint8)], axis=2)
            cond_tag_img = img2tensor(img=cond_tag_img, normalize=True, mean=self.pixel_mean, std=self.pixel_std, dim_order='chw')

        tag_img = np.concatenate([tag_img[..., [-1]], (pad_rgb(tag_img) * 255).astype(np.uint8)], axis=2)
        tag_img = img2tensor(img=tag_img, normalize=True, mean=self.pixel_mean, std=self.pixel_std, dim_order='chw')
        cond_fullpage = img2tensor(img=cond_fullpage, normalize=True, mean=self.pixel_mean, std=self.pixel_std, dim_order='chw')

        batch = {
            'img': tag_img,
            'cond_fullpage': cond_fullpage,
            'srcp': imgp,
            'caption': caption,
            'original_sizes': (self.target_size, self.target_size),
            'crop_top_lefts': (0, 0)
        }
        if self.mask_cond:
            batch['cond_tag_img'] = cond_tag_img
        
        return batch

        

    def __getitem__(self, index):

        while True:
            try:
                samplep = self.sample_list[index]['img']
                return self.get_sample(index)
            except:
                print(f'Failed to load {samplep}: ')
                print(traceback.format_exc())
                index = random.randint(0, len(self) - 1)
                continue


class LayerDiffDataset3D(Dataset):
    def __init__(
        self, 
        src_list: Union[str, Dict, List],
        target_size: Union[int, Tuple] = 1024,
        random_flip=0.,
        pixel_mean: List[float] = [0., 0., 0., 0.],
        pixel_std: List[float] = [255., 255., 255., 255.],       # SAM input std
        tag_list: List[str] = None,
        contrl_signals: str = None,
        random_pad_prob: bool = 0.0,
        load_ojects: bool = True,
        random_aug_mask: float = 0.,
        mix_tag_groups: list = None,
        ngroup_iters = 1,
        include_blend_cls: bool = False,
        pad_num_frames=-1,
        pad_argb: bool = True,
        **kwargs) -> None:

        super().__init__()

        self.target_size = target_size

        self.tag_list = tag_list

        # parse source files
        self.sample_list = load_sample_list(src_list)
        
        self.random_flip = random_flip
        self.random_aug_mask = random_aug_mask

        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

        self.random_pad_prob = random_pad_prob
        self.load_objects = load_ojects
        self.contrl_signals = contrl_signals
        self.mix_tag_groups = mix_tag_groups

        self._current_group = 0
        self._current_group_iters = 0
        self.ngroup_iters = ngroup_iters
        self.ngroup = len(mix_tag_groups)
        self.include_blend_cls = include_blend_cls
        self.pad_num_frames = pad_num_frames
        self.pad_argb = pad_argb

    def __len__(self):
        return len(self.sample_list)

    def get_sample(self, index: int):

        def _aug_size_transform(img):
            if flip_tb:
                img = img[::-1]
            if flip_lr:
                img = img[:, ::-1]
            img = center_square_pad_resize(img, target_size=self.target_size)
            return img

        flip_lr = flip_tb = False
        if self.random_flip > 0 and random.random() < self.random_flip:
            flip_lr = True

        sample = self.sample_list[index]
        imgp = sample['img']
        imgname = osp.join(osp.dirname(imgp), osp.splitext(osp.basename(imgp))[0])
        annp = imgname + '_ann.json'
        ann = json2dict(annp)

        fullpage = np.array(Image.open(imgp).convert('RGBA'))

        tag_group_info = self.mix_tag_groups[self._current_group]
        target_tags = tag_group_info['tags'].copy()
        blended_rst = load_blend_parts(imgp, tag_group_info['tags'], ann=ann, keep_loaded_parts=True)
        
        crop_tag_list = tag_group_info.get('crop_tag_list', 
                                           [t for t in target_tags if t not in tag_group_info.get('size_skip_tags', [])])
        tx1, ty1, tx2, ty2 = xyxy_from_taglist(ann, crop_tag_list)
        blended_img = blended_rst['img'][ty1: ty2, tx1: tx2]

        random_pad = random.random() < self.random_pad_prob
        if random_pad:
            fh, fw = fullpage.shape[:2]
            tmax, lmax = ty1, tx1
            rmax = fw - tx2 - 1
            bmax = fh - ty2 - 1
            blended_img, (t, b, l, r) = random_pad_img(blended_img, tmax, bmax, lmax, rmax)
            tx1 -= l
            ty1 -= t
            th, tw = blended_img.shape[:2]
            tx2 = tx1 + tw
            ty2 = ty1 + th
            fullpage = fullpage[ty1: ty2, tx1: tx2].copy()
            blended_img = blended_rst['img'][ty1: ty2, tx1: tx2].copy()
        else:
            fullpage = fullpage[ty1: ty2, tx1: tx2].copy()
        
        fullpage = _aug_size_transform(fullpage)
        fh, fw = fullpage.shape[:2]

        empty_tensor = torch.zeros((4, fh, fw), dtype=torch.float32)
        
        cond_fullpage = np.concatenate([np.full_like(fullpage[..., :1], fill_value=255), fullpage[..., :3]], axis=2)
        cond_fullpage = img2tensor(img=cond_fullpage, normalize=True, mean=self.pixel_mean, std=self.pixel_std, dim_order='chw')

        def _np_transform(img):
            h, w = img.shape[:2]
            img = _aug_size_transform(img)
            if self.pad_argb:
                img = pad_rgb(img, return_format='argb')
            else:
                img = np.concatenate([img[..., 3:], img[..., :3]], axis=2).astype(np.float32) / 255.
            img = img2tensor(img=img, dim_order='chw')
            return img

        tgt_img_list = [empty_tensor] * len(target_tags)
        caption_list = [t.replace('_', ' ') for t in target_tags]
        empty_mask = [True] * len(target_tags)

        footwear_exist = False
        for part in blended_rst['part_list']:
            tag_img = _np_transform(part['img'][ty1: ty2, tx1: tx2])
            tag_index = target_tags.index(part['tag'])
            if part['tag'] == 'footwear':
                footwear_exist = True
            empty_mask[tag_index] = False
            tgt_img_list[tag_index] = tag_img

        if not footwear_exist and 'legwear' in caption_list and 'footwear' in target_tags:
            caption_list[caption_list.index('legwear')] = 'legwear,footwear'

        npad = self.pad_num_frames - len(caption_list)
        if npad > 0:
            pad_caption = ''
            pad_img = empty_tensor
            caption_list = caption_list + npad * [pad_caption]
            tgt_img_list += [pad_img] * npad
            empty_mask += [True] * npad
                
        batch = {
            'img_list': torch.stack(tgt_img_list),
            'cond_fullpage': cond_fullpage,
            'srcp': imgp,
            'caption_list': '\n'.join(caption_list),
            'original_sizes': (self.target_size, self.target_size),
            'crop_top_lefts': (0, 0),
            'empty_mask': torch.tensor(empty_mask),
            'group_index': self._current_group
        }

        self._current_group_iters += 1
        if self._current_group_iters % self.ngroup_iters == 0:
            self._current_group_iters = 0
            self._current_group = (self._current_group + 1) % self.ngroup
        
        return batch


    def __getitem__(self, index):

        while True:
            try:
                samplep = self.sample_list[index]['img']
                return self.get_sample(index)
            except:
                raise
                print(f'Failed to load {samplep}: ')
                print(traceback.format_exc())
                index = random.randint(0, len(self) - 1)
                continue




class LayerDiffVAEDataset(Dataset):
    def __init__(
        self, 
        src_list: Union[str, Dict, List],
        target_size: Union[int, Tuple] = 1024,
        random_flip=0.,
        pixel_mean: List[float] = [0., 0., 0., 0.],
        pixel_std: List[float] = [255., 255., 255., 255.],       # SAM input std
        rgb_prob=0.0,
        random_crop=0.,
        tag_list: List[str] = None,
        random_pad_prob: bool = 0.0,
        pad_argb: bool = True,
        random_rotate_prob: float = 0,
        fullpage_prob: float = 0.1,
        load_all=False,
        load_rgb_cond=False,
        mix_tag_groups: list = None,
        **kwargs) -> None:

        super().__init__()

        self.target_size = target_size

        self.tag_list = tag_list

        self.sample_list = load_sample_list(src_list)
        
        self.random_flip = random_flip

        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

        self.random_pad_prob = random_pad_prob
        self.pad_argb = pad_argb

        self.random_rotate_prob = random_rotate_prob
        self.fullpage_prob = fullpage_prob
        self.load_all = load_all
        self.load_rgb_cond = load_rgb_cond
        self.rgb_prob = rgb_prob
        self.random_crop = random_crop
        self.mix_tag_groups = mix_tag_groups

    def __len__(self):
        return len(self.sample_list)

    def get_sample(self, index: int):

        def _aug_size_transform(img):
            if flip_tb:
                img = img[::-1]
            if flip_lr:
                img = img[:, ::-1]
            if rot_k > 0:
                np.rot90(img, k=rot_k, axes=(0, 1))
            img = center_square_pad_resize(img, target_size=self.target_size)
            return img

        rot_k = 0
        flip_lr = flip_tb = False
        if self.random_flip > 0 and random.random() < self.random_flip:
            flip_lr = True
        if self.random_flip > 0 and random.random() < self.random_flip:
            flip_tb = True
        if self.random_rotate_prob > 0 and random.random() < self.random_rotate_prob:
            rot_k = random.choice([1, 2, 3])

        sample = self.sample_list[index]
        imgp = sample['img']
        imgname = osp.join(osp.dirname(imgp), osp.splitext(osp.basename(imgp))[0])
        annp = imgname + '_ann.json'
        ann = json2dict(annp)

        valid_taglist = [t for t in ann['tag_info'] if ann['tag_info'][t]['exists']]
        fullpage_only = len(valid_taglist) == 0 or random.random() < self.fullpage_prob

        if fullpage_only:
            tag = 'fullpage'
            fullpage = np.array(Image.open(imgp).convert('RGBA'))
            tx1, ty1, tx2, ty2 = xyxy_from_taglist(ann, valid_taglist)
        else:
            # if self.random_crop > 0 and random.random() < self.random_crop:
            #     img = random_crop
            # else:
            h, w = ann['final_size']
            tag = random.choice(valid_taglist)
            tx1, ty1, tx2, ty2 = ann['tag_info'][tag]['xyxy']
            p = osp.splitext(imgp)[0] + '_' + tag + '.png'
            fullpage = np.zeros((h, w, 4), dtype=np.uint8)
            fullpage[ty1: ty2, tx1: tx2] = np.array(Image.open(p))

            # crop_tag_list = tag_group_info.get('crop_tag_list', 
            #                                    [t for t in target_tags if t not in tag_group_info.get('size_skip_tags', [])])
            
            if self.mix_tag_groups is not None:
                crop_tag_list = None
                for cl in self.mix_tag_groups:
                    if tag in cl['tags']:
                        crop_tag_list = cl['tags']
                        break
                if crop_tag_list is not None:
                    tx1, ty1, tx2, ty2 = xyxy_from_taglist(ann, crop_tag_list)
                    # print(tx1, ty1, tx2, ty2)

        random_pad = not fullpage_only and random.random() < self.random_pad_prob
        fh, fw = fullpage.shape[:2]
        img = fullpage[ty1: ty2, tx1: tx2].copy()
        if random_pad:
            tmax, lmax = ty1, tx1
            rmax = fw - tx2 - 1
            bmax = fh - ty2 - 1
            img, (t, b, l, r) = random_pad_img(img, tmax, bmax, lmax, rmax)
            tx1 -= l
            ty1 -= t
            th, tw = img.shape[:2]
            tx2 = tx1 + tw
            ty2 = ty1 + th

        if self.rgb_prob > 0 and random.random() < self.rgb_prob:
            img[..., -1] = 255

        def _np_transform(img):
            img = _aug_size_transform(img)
            if self.pad_argb:
                img = pad_rgb(img, return_format='argb')
            else:
                img = np.concatenate([img[..., 3:], img[..., :3]], axis=2).astype(np.float32) / 255.
            img = img2tensor(img=img, dim_order='chw')
            return img

        img = _np_transform(img)
        batch = {
            'img': img,
            'srcp': imgp,
            'tag': tag
        }

        if self.load_rgb_cond:
            if not fullpage_only:
                fullpage = np.array(Image.open(imgp).convert('RGBA'))
            rgb_cond = fullpage[ty1: ty2, tx1: tx2, :3].copy()
            rgb_cond = _aug_size_transform(rgb_cond)
            rgb_cond = img2tensor(img=rgb_cond, dim_order='chw', normalize=True)
            batch['rgb_cond'] = rgb_cond

        return batch

    def __getitem__(self, index):

        while True:
            try:
                samplep = self.sample_list[index]['img']
                return self.get_sample(index)
            except:
                raise
                print(f'Failed to load {samplep}: ')
                print(traceback.format_exc())
                index = random.randint(0, len(self) - 1)
                continue

