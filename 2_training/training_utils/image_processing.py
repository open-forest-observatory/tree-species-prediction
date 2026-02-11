import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from functools import partial

from configs.model_config import model_config


def letterbox_to_square(img, target: int, fill_color=(128, 128, 128), padding_mode='constant'):
    """
    Resize and pad image to square.

    Args:
        img: PIL Image
        target: target square size
        fill_color: RGB tuple for constant padding (default gray)
        padding_mode: 'constant', 'reflect', 'replicate', 'circular'
    """
    w, h = img.size
    scale = target / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = F.resize(img, [new_h, new_w], interpolation=InterpolationMode.BICUBIC, antialias=True)
    pad_w = target - new_w
    pad_h = target - new_h

    # pad equally: (left, top, right, bottom)
    padding = [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2]

    if padding_mode == 'constant':
        img = F.pad(img, padding, fill=fill_color, padding_mode='constant')
    else:
        # for reflect/replicate/circular, convert to tensor, pad, convert back
        img_tensor = F.to_tensor(img)
        # torch.nn.functional.pad uses (left, right, top, bottom) order
        img_tensor = torch.nn.functional.pad(
            img_tensor,
            (padding[0], padding[2], padding[1], padding[3]),
            mode=padding_mode
        )
        img = F.to_pil_image(img_tensor)

    return img

def downsample_long_side_if_needed(img, long_side_thresh: int, downsample_to: int):
    w, h = img.size
    long_side = max(w, h)
    if long_side <= long_side_thresh:
        return img
    scale = downsample_to / long_side
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return F.resize(img, [new_h, new_w], interpolation=InterpolationMode.BICUBIC, antialias=True)

def static_preprocess(
    img,
    target: int,
    long_side_thresh: int,
    downsample_to: int,
    fill_color=(128, 128, 128),
    padding_mode='constant',
):
    """
    Static preprocessing: downsample if needed, then letterbox to square.

    Args:
        img: PIL Image
        target: target square size
        long_side_thresh: threshold for downsampling
        downsample_to: target size for downsampling
        fill_color: RGB tuple for padding
        padding_mode: 'constant', 'reflect', 'replicate', 'circular'
    """
    # if extremely large, downsample first to control compute/encode
    w, h = img.size
    long_side = max(w, h)
    if long_side > long_side_thresh:
        img = downsample_long_side_if_needed(img, long_side_thresh, downsample_to)

    # Always letterbox to a square of `target` once (avoids later re-resizes)
    img = letterbox_to_square(img, target, fill_color=fill_color, padding_mode=padding_mode)
    return img

def unnormalize(img_t: torch.Tensor, mean, std):
    """
    img_t: (C,H,W) tensor that has been normalized with given mean/std
    returns: (C,H,W) tensor in [0,1] suitable for to_pil_image
    """
    mean = torch.as_tensor(mean, device=img_t.device).view(-1, 1, 1)
    std  = torch.as_tensor(std,  device=img_t.device).view(-1, 1, 1)
    x = img_t * std + mean
    return x.clamp(0.0, 1.0)

def build_transforms(
    target,             # final input size for the model (square)
    long_side_thresh,   # 'too big' cutoff to trigger downsampling
    downsample_to,      # target for downsample specifically
    mean,               # stats of backbone training data for normalizing
    std,
    fill_color=None,    # RGB tuple for padding (uses model_config.pad_color if None)
    padding_mode=None,  # padding mode (uses model_config.pad_mode if None)
    color_jitter=None,  # color jitter strength (uses model_config.color_jitter_strength if None)
):
    # use config defaults if not specified
    if fill_color is None:
        fill_color = model_config.pad_color
    if padding_mode is None:
        padding_mode = model_config.pad_mode
    if color_jitter is None:
        color_jitter = model_config.color_jitter_strength

    # static deterministic transforms for resizing to input dim
    static_tf = T.Lambda(lambda im: static_preprocess(
        im, target=target, long_side_thresh=long_side_thresh, downsample_to=downsample_to,
        fill_color=fill_color, padding_mode=padding_mode
    ))

    # random train -> stochastic augs that preserve size, then tensor+norm
    train_transforms = [
        T.RandomResizedCrop(
            (target, target), scale=(0.85, 1.0),
            interpolation=InterpolationMode.BICUBIC, antialias=True
        ),
        T.RandomHorizontalFlip(0.5),
    ]
    # only add color jitter if strength > 0
    if color_jitter > 0:
        train_transforms.append(T.ColorJitter(
            brightness=color_jitter,
            contrast=color_jitter,
            saturation=color_jitter,
            hue=0.0  # no hue shift for tree species (color-sensitive)
        ))
    train_transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    random_train_tf = T.Compose(train_transforms)

    # random eval -> deterministic finishing only
    random_eval_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    return static_tf, random_train_tf, random_eval_tf


def build_tta_transforms(target, mean, std, n_crops=5):
    """
    Build test-time augmentation transforms for multi-crop inference.

    Returns a list of transforms that produce different views of the same image.
    Final prediction is averaged across all views.
    """
    transforms = [
        # center crop (standard eval)
        T.Compose([
            T.CenterCrop(target),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]),
        # horizontal flip
        T.Compose([
            T.CenterCrop(target),
            T.RandomHorizontalFlip(p=1.0),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]),
    ]

    if n_crops >= 5:
        # 4 corners
        corners = [
            (0, 0),                     # top-left
            (0, target),                # top-right
            (target, 0),                # bottom-left
            (target, target),           # bottom-right
        ]
        for i, (top, left) in enumerate(corners[:n_crops - 1]):
            transforms.append(T.Compose([
                T.Lambda(lambda img, t=top, l=left: F.crop(img, t, l, target, target)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]))

    return transforms
