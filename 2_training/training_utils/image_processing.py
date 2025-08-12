import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from functools import partial

def letterbox_to_square(img, target: int):
    w, h = img.size
    scale = target / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = F.resize(img, [new_h, new_w], interpolation=InterpolationMode.BICUBIC, antialias=True)
    pad_w = target - new_w
    pad_h = target - new_h
    # pad equally: (left, top, right, bottom)
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

def conditional_preprocess(
    img,
    is_training: bool,
    target: int,
    long_side_thresh: int,
    downsample_to: int,
):
    w, h = img.size
    long_side = max(w, h)

    if long_side > long_side_thresh:
        # keep field-of-view, control token count
        img = downsample_long_side_if_needed(img, long_side_thresh, downsample_to)
        img = letterbox_to_square(img, target)
    else:
        # smaller images: crop for focus/aug (train) or deterministic center crop (val)
        if is_training:
            img = T.RandomResizedCrop(
                (target, target), scale=(0.85, 1.0),
                interpolation=InterpolationMode.BICUBIC, antialias=True
            )(img)
            img = T.RandomHorizontalFlip()(img)
        else:
            if min(w, h) >= target:
                img = F.center_crop(img, (target, target))
            else:
                img = letterbox_to_square(img, target)
    return img

def build_transforms(
    target,             # final input size for the model (square)
    long_side_thresh,   # 'too big' cutoff to trigger downsampling
    downsample_to,      # target for downsample specifically
    mean,               # stats of backbone training data for normalizing
    std,
):
    preprocess_train = partial(
        conditional_preprocess,
        is_training=True,
        target=target,
        long_side_thresh=long_side_thresh,
        downsample_to=downsample_to,
    )
    preprocess_eval = partial(
        conditional_preprocess,
        is_training=False,
        target=target,
        long_side_thresh=long_side_thresh,
        downsample_to=downsample_to,
    )
    train_transform = T.Compose([
        T.Lambda(preprocess_train),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    eval_transform = T.Compose([
        T.Lambda(preprocess_eval),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    return train_transform, eval_transform
