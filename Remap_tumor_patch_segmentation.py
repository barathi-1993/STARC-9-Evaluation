# -*- coding: utf-8 -*-

import os
from PIL import Image, ImageDraw
import numpy as np


def remap_patch_tumor_segmentation(patch_image_path,
                                   classified_tile_root,
                                   output_mask_base,
                                   tile_size=256,
                                   tumor_class_name='TUM',
                                   mask_color=(255, 0, 0),
                                   scale_factors=(1,16)):

    patch = Image.open(patch_image_path)
    width, height = patch.size
    mask = Image.new('RGB', (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(mask)

    slide = os.path.splitext(os.path.basename(patch_image_path))[0]
    tumor_dir = os.path.join(classified_tile_root,tumor_class_name)
    if not os.path.isdir(tumor_dir):
        raise FileNotFoundError(f"Tumor folder not found: {tumor_dir}")

    for fname in os.listdir(tumor_dir):
        if not fname.lower().endswith(('.png','.tif','.tiff','.jpg','.jpeg')):
            continue
        parts = fname.split('_')
        x = y = None
        for p in parts:
            if p.startswith('x') and p[1:].isdigit(): x = int(p[1:])
            if p.startswith('y') and p[1:].isdigit(): y = int(p[1:])
        if x is None or y is None:
            continue
        draw.rectangle([x, y, x + tile_size, y + tile_size], fill=mask_color)

    base, _ = os.path.splitext(output_mask_base)
    os.makedirs(os.path.dirname(base), exist_ok=True)

    # Determine save format by extension
    ext = os.path.splitext(output_mask_base)[1].lower()
    fmt = 'TIFF' if ext in ['.tif', '.tiff'] else 'PNG'
    full_path = base + ext
    # Save full-resolution mask
    if fmt == 'TIFF':
        mask.save(full_path, format=fmt, compression='tiff_deflate')
    else:
        mask.save(full_path, format=fmt)

    # Downsample and side-by-side
    for sf in scale_factors:
        new_w, new_h = width // sf, height // sf
        mask_ds = mask.resize((new_w, new_h), Image.NEAREST)
        down_path = f"{base}_down{sf}x{ext}"
        if fmt == 'TIFF':
            mask_ds.save(down_path, format=fmt, compression='tiff_deflate')
        else:
            mask_ds.save(down_path, format=fmt)

        patch_ds = patch.resize((new_w, new_h), Image.BILINEAR)
        side = Image.new('RGB', (new_w * 2, new_h))
        side.paste(patch_ds, (0, 0))
        side.paste(mask_ds, (new_w, 0))
        side_path = f"{base}_sidebyside{sf}x{ext}"
        side.save(side_path, format=fmt)

    return full_path


def reconstruct_gt_by_tiles(patch_image_path,
                             classified_tile_root,
                             ground_truth_mask_path,
                             output_gt_recon_path,
                             tile_size=256,
                             tumor_class_name='TUM'):
    patch = Image.open(patch_image_path)
    W, H = patch.size
    gt_full = Image.open(ground_truth_mask_path).convert('1')
    recon = Image.new('1', (W, H), 0)

    slide = os.path.splitext(os.path.basename(patch_image_path))[0]
    tumor_dir = os.path.join(classified_tile_root,tumor_class_name)
    if not os.path.isdir(tumor_dir):
        raise FileNotFoundError(f"Tumor folder not found: {tumor_dir}")

    for fname in os.listdir(tumor_dir):
        parts = fname.split('_')
        x = y = None
        for p in parts:
            if p.startswith('x') and p[1:].isdigit(): x = int(p[1:])
            if p.startswith('y') and p[1:].isdigit(): y = int(p[1:])
        if x is None or y is None:
            continue
        tile_gt = gt_full.crop((x, y, x + tile_size, y + tile_size))
        recon.paste(tile_gt, (x, y))

    # Ensure output uses .tif extension
    base, _ = os.path.splitext(output_gt_recon_path)
    out_path = base + '.tif'
    os.makedirs(os.path.dirname(base), exist_ok=True)
    recon.save(out_path, format='TIFF', compression='tiff_deflate')
    return out_path


# Example usage
if __name__ == '__main__':
    patch_path = '/path/to/2048_patchsize_image.tiff'
    classified_root = '/path/to/classified_tiles_from_the_patch_of_size_256'  

    # 2) reconstruct GT restricted to predicted tiles (TIFF)
    gt_recon = reconstruct_gt_by_tiles(
        patch_image_path=patch_path,
        classified_tile_root=classified_root,
        ground_truth_mask_path='/path/to/2048_patsize_ground_truth_mask.tif',
        output_gt_recon_path='/path/to/save_predicted_mask',
        tile_size=256,
        tumor_class_name='TUM'
    )
