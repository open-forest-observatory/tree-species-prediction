import torch
from torch.amp import autocast
import torch.nn.functional as F

from tqdm import tqdm

from configs.model_config import model_config
from training_utils.metrics import update_tp_fp_fn, compute_epoch_metrics

def _step_epoch(
        tree_model, dataloader, device, criterion,
        optim=None, scaler=None, training=False, epoch_num=None,
        sample_weight_map=None
    ):
    # same fn used for training and validation, so setup accordingly
    if training:
        tree_model.train()
        pbar_msg = f"Training epoch {epoch_num if epoch_num is not None else ''}"
        use_amp = True if scaler else False
        assert optim is not None
    else:
        tree_model.eval()
        pbar_msg = f"Validation epoch {epoch_num if epoch_num is not None else ''}"
        use_amp = False
    
    # loss / acc tracking init
    running_loss = 0.0
    total = 0
    correct = 0
    num_classes = None # found in loop using model out dim
    eps = 1e-12 # avoid div by 0

    # iterate through batches
    pbar = tqdm(dataloader, desc=pbar_msg)
    for batch_idx, (imgs, labels, metas) in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = labels.size(0)

        # forward with optional automatic mixed precision
        with autocast(enabled=use_amp, device_type=str(device), dtype=model_config.amp_dtype):
            logits = tree_model(imgs)

            # scale loss by weights given from OMP per batch importance
            if training and sample_weight_map is not None:
                # per-sample loss
                loss_vec = F.cross_entropy(logits, labels, reduction='none')  # (B,)

                # DEBUG: make sure samples indexed correctly from data
                ids = [m['ds_idx'] for m in metas]
                missing = [i for i in ids if i not in sample_weight_map]
                if missing:
                    raise KeyError(f"Missing {len(missing)}/{len(ids)} weights. Example missing ds_idx: {missing[:10]}")


                # weights lookup by global_idx
                w = torch.tensor(
                    [sample_weight_map[int(m['ds_idx'])] for m in metas],
                    device=labels.device,
                    dtype=loss_vec.dtype,
                )  # (B,)
                # sanity checks
                assert torch.isfinite(w).all()
                assert (w >= 0).all()

                # normalize weights so loss scale stays comparable epoch-to-epoch
                #w = w / (w.mean() + 1e-12)

                loss = (loss_vec * w).mean()
            else:
                loss = criterion(logits, labels)

        if training:
            # zero previous gradients before backprop
            optim.zero_grad()
            if scaler is not None:
                # scale back to fp32 and then backprop
                scaler.scale(loss).backward()
                scaler.step(optim) # step down the gradient
                scaler.update()
            else:
                loss.backward() # back propagation
                optim.step() # step down the gradient

        # bunch of running metrics
        with torch.no_grad(): # don't track gradients just for metrics
            running_loss += float(loss.detach().item()) * batch_size
            preds = logits.argmax(dim=1) # predictions are largest nums of output logits
            correct += (preds == labels).sum().item() # count correct predictions
            total += batch_size

            # init per class metrics counters on first pass (if num_classes is still none)
            if num_classes is None:
                num_classes = logits.size(1) # length of last layer output equates to num classes
                tp = torch.zeros(num_classes, dtype=torch.long) # true positives
                fp = torch.zeros(num_classes, dtype=torch.long) # true negatives
                fn = torch.zeros(num_classes, dtype=torch.long) # false negatives

            # calculate per class tp/fp/fn per batch
            update_tp_fp_fn(tp, fp,fn, preds, labels, num_classes)
            running_macros = compute_epoch_metrics(running_loss, total, correct, tp, fp, fn, average='macro', eps=eps)
            running_micros = compute_epoch_metrics(running_loss, total, correct, tp, fp, fn, average='micro', eps=eps)

            pbar.set_postfix({
                'loss':f"{running_macros['loss']:.4f}",
                'acc':f"{running_macros['accuracy']*100:.2f}%",
                #'+prec':f"{running_macros['precision']*100:.2f}%",
                #'+rec':f"{running_macros['recall']*100:.2f}%",
                'macF1':f"{running_macros['f1']*100:.2f}%",
                'micF1':f"{running_micros['f1']*100:.2f}%",
            })

    # final epoch metrics (metrics in loop are running estimates)
    # final epoch metrics
    macro = compute_epoch_metrics(running_loss, total, correct, tp, fp, fn, average="macro", eps=eps)
    micro = compute_epoch_metrics(running_loss, total, correct, tp, fp, fn, average="micro", eps=eps)

    metrics = {
        "loss": macro["loss"],  # same loss either way
        "accuracy": macro["accuracy"],

        "precision_macro": macro["precision"],
        "recall_macro": macro["recall"],
        "f1_macro": macro["f1"],

        "precision_micro": micro["precision"],
        "recall_micro": micro["recall"],
        "f1_micro": micro["f1"],
    }
    return metrics