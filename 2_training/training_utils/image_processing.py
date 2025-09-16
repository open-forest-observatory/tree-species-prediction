import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from functools import partial

# TODO: combine these first two fns to only need to resize once. Note: conditional preprocess will also need to be updated
def letterbox_to_square(img, target: int):
    w, h = img.size
    scale = target / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = F.resize(img, [new_h, new_w], interpolation=InterpolationMode.BICUBIC, antialias=True)
    pad_w = target - new_w
    pad_h = target - new_h
    # pad equally: (left, top, right, bottom)
    # TODO: have option to pad with avg color instead of just black 
    img = F.pad(img, [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2], fill=0)
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
):
    # if extremely large, downsample first to control compute/encode
    w, h = img.size
    long_side = max(w, h)
    if long_side > long_side_thresh:
        img = downsample_long_side_if_needed(img, long_side_thresh, downsample_to)

    # Always letterbox to a square of `target` once (avoids later re-resizes)
    img = letterbox_to_square(img, target)
    return img

def build_transforms(
    target,             # final input size for the model (square)
    long_side_thresh,   # 'too big' cutoff to trigger downsampling
    downsample_to,      # target for downsample specifically
    mean,               # stats of backbone training data for normalizing
    std,
):
    # static deterministic transforms for resizing to input dim
    static_tf = T.Lambda(lambda im: static_preprocess(
        im, target=target, long_side_thresh=long_side_thresh, downsample_to=downsample_to
    ))

    # random train -> stochastic augs that preserve size, then tensor+norm
    random_train_tf = T.Compose([
        T.RandomResizedCrop(
            (target, target), scale=(0.85, 1.0),
            interpolation=InterpolationMode.BICUBIC, antialias=True
        ),
        T.RandomHorizontalFlip(0.15),
        T.ColorJitter(0.1, 0.1, 0.1, 0.0),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    # random eval -> deterministic finishing only
    random_eval_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    # heavier augementations for rare classes
    rare_random_train_tf = T.Compose([
        T.RandomResizedCrop((target, target), scale=(0.7, 1.0), interpolation=InterpolationMode.BICUBIC, antialias=True),
        T.RandomHorizontalFlip(0.5),
        T.ColorJitter(0.2, 0.2, 0.2, 0.1), # brightness, contrast, saturation, hue
        T.RandomAffine(degrees=15, translate=(0.2,0.2), scale=(0.8,1.2)),
        T.Resize((target, target)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    return static_tf, random_train_tf, random_eval_tf, rare_random_train_tf
