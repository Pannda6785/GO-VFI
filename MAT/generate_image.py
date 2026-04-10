# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""
import cv2
import pyspng
import glob
import os
import re
import random
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors_file

import legacy
from datasets.mask_generator_512 import RandomMask
from networks.mat import Generator


def resolve_input_paths(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, '*.png')) + glob.glob(os.path.join(path, '*.jpg')))
    return []


def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = {name: tensor for name, tensor in named_params_and_buffers(src_module)}
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)


def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())


def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())


def load_generator_checkpoint(network_path: str, device: torch.device, resolution: int) -> torch.nn.Module:
    if network_path.endswith('.safetensors'):
        net_res = 512 if resolution > 512 else resolution
        model = Generator(
            z_dim=512,
            c_dim=0,
            w_dim=512,
            img_resolution=net_res,
            img_channels=3,
        ).to(device).eval().requires_grad_(False)
        state_dict = load_safetensors_file(network_path, device=str(device))
        incompatible = model.load_state_dict(state_dict, strict=False)
        allowed_missing = {
            name
            for name in incompatible.missing_keys
            if name.endswith('noise_const') or name.endswith('noise_strength')
        }
        unexpected = set(incompatible.unexpected_keys)
        disallowed_missing = set(incompatible.missing_keys) - allowed_missing
        if disallowed_missing or unexpected:
            raise RuntimeError(
                'Failed to load safetensors checkpoint. '
                f'Missing keys: {sorted(disallowed_missing)}. '
                f'Unexpected keys: {sorted(unexpected)}.'
            )
        return model

    with dnnlib.util.open_url(network_path) as f:
        model = legacy.load_network_pkl(f)['G_ema'].to(device).eval().requires_grad_(False) # type: ignore
    return model


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network checkpoint filename (.pkl or .safetensors)', required=True)
@click.option('--dpath', help='the path of the input image', required=True)
@click.option('--mpath', help='the path of the mask')
@click.option('--mask-dilate', type=int, default=0, show_default=True, help='Dilate the inpaint mask by this many pixels before inverting it')
@click.option('--resolution', type=int, help='resolution of input image', default=512, show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    dpath: str,
    mpath: Optional[str],
    mask_dilate: int,
    resolution: int,
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
):
    """
    Generate images using pretrained network pickle.
    """
    seed = 240  # pick up a random number
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print(f'Loading data from: {dpath}')
    img_list = resolve_input_paths(dpath)
    if not img_list:
        raise click.ClickException(f'No input images found for: {dpath}')

    if mpath is not None:
        print(f'Loading mask from: {mpath}')
        mask_list = resolve_input_paths(mpath)
        if not mask_list:
            raise click.ClickException(f'No mask images found for: {mpath}')
        assert len(img_list) == len(mask_list), 'illegal mapping'

    print(f'Loading networks from: {network_pkl}')
    device = torch.device('cuda')
    G_saved = load_generator_checkpoint(network_pkl, device, resolution)
    net_res = 512 if resolution > 512 else resolution
    G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=net_res, img_channels=3).to(device).eval().requires_grad_(False)
    copy_params_and_buffers(G_saved, G, require_all=True)

    os.makedirs(outdir, exist_ok=True)

    # no Labels.
    label = torch.zeros([1, G.c_dim], device=device)

    def read_image(image_path):
        with open(image_path, 'rb') as f:
            if pyspng is not None and image_path.endswith('.png'):
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
            image = np.repeat(image, 3, axis=2)
        image = image.transpose(2, 0, 1) # HWC => CHW
        image = image[:3]
        return image

    def pad_image_to_square(image, target_resolution):
        _, height, width = image.shape
        scale = min(target_resolution / height, target_resolution / width)
        new_height = max(1, int(round(height * scale)))
        new_width = max(1, int(round(width * scale)))
        image_hwc = np.transpose(image, (1, 2, 0))
        resized = cv2.resize(image_hwc, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        if resized.ndim == 2:
            resized = resized[:, :, np.newaxis]
        canvas = np.zeros((target_resolution, target_resolution, resized.shape[2]), dtype=resized.dtype)
        top = (target_resolution - new_height) // 2
        left = (target_resolution - new_width) // 2
        canvas[top:top + new_height, left:left + new_width] = resized
        return canvas.transpose(2, 0, 1), (top, left, new_height, new_width)

    def read_mask(mask_path, target_resolution, dilate_pixels, pad_info):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise click.ClickException(f'Failed to read mask image: {mask_path}')
        top, left, new_height, new_width = pad_info
        mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8)
        if dilate_pixels > 0:
            kernel_size = dilate_pixels * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask = cv2.dilate(mask, kernel, iterations=1)
        canvas = np.zeros((target_resolution, target_resolution), dtype=np.uint8)
        canvas[top:top + new_height, left:left + new_width] = mask
        # User mask marks the region to inpaint; the model expects 1 for known pixels.
        mask = 1.0 - canvas.astype(np.float32)
        return mask

    def to_image(image, lo, hi):
        image = np.asarray(image, dtype=np.float32)
        image = (image - lo) * (255 / (hi - lo))
        image = np.rint(image).clip(0, 255).astype(np.uint8)
        image = np.transpose(image, (1, 2, 0))
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        return image

    def unpad_image(image, pad_info, original_height, original_width):
        top, left, new_height, new_width = pad_info
        cropped = image[top:top + new_height, left:left + new_width]
        restored = cv2.resize(cropped, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        return restored

    if resolution != 512:
        noise_mode = 'random'
    with torch.no_grad():
        for i, ipath in enumerate(img_list):
            iname = os.path.basename(ipath).replace('.jpg', '.png')
            print(f'Prcessing: {iname}')
            image = read_image(ipath)
            _, original_height, original_width = image.shape
            image, pad_info = pad_image_to_square(image, resolution)
            image = (torch.from_numpy(image).float().to(device) / 127.5 - 1).unsqueeze(0)

            if mpath is not None:
                mask = read_mask(mask_list[i], resolution, mask_dilate, pad_info)
                mask = torch.from_numpy(mask).float().to(device).unsqueeze(0).unsqueeze(0)
            else:
                mask = RandomMask(resolution) # adjust the masking ratio by using 'hole_range'
                mask = torch.from_numpy(mask).float().to(device).unsqueeze(0)

            z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
            output = G(image, mask, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            output = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
            output = output[0].cpu().numpy()
            output = unpad_image(output, pad_info, original_height, original_width)
            PIL.Image.fromarray(output, 'RGB').save(f'{outdir}/{iname}')


if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
