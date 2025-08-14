import torch
from torch.amp import autocast

from tqdm import tqdm

from configs.model_config import model_config

def _step_epoch(tree_model, dataloader, device, criterion, optim=None, scaler=None, early_stopper=None, training=False):
    # same fn used for training and validation, so setup accordingly
    if training:
        tree_model.train()
        pbar_msg = 'Training'
        use_amp = True if scaler else False
        assert optim is not None
    else:
        tree_model.eval()
        pbar_msg = 'Validation'
        use_amp = False

    # loss / acc tracking init
    running_loss = 0.0
    total = 0
    correct = 0
    num_classes = None # found in loop using model out dim
    eps = 1e-12 # avoid div by 0

    # iterate through batches
    pbar = tqdm(dataloader, desc=pbar_msg)
    for batch_idx, (imgs, labels, metadata) in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=False)
        labels = labels.to(device, non_blocking=False)
        batch_size = labels.size(0)

        # forward with optional automatic mixed precision
        with autocast(enabled=use_amp, device_type='cuda', dtype=model_config.amp_dtype):
            logits = tree_model(imgs)
            loss = criterion(logits, labels)

        if training:
            # zero previous gradients before backprop
            optim.zero_grad()
            if scaler is not None:
                # scale back to fp32 and then backprop
                scaler.scale(loss).backward()
                scaler.step(optim) # step down the gradient
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
                num_classes = logits.size(0) # length of last layer output equates to num classes
                tp = torch.zeros(num_classes, dtype=torch.long) # true positives
                fp = torch.zeros(num_classes, dtype=torch.long) # true negatives
                fn = torch.zeros(num_classes, dtype=torch.long) # false negatives

            # calculate per class tp/fp/fn
            for label in range(num_classes):
                label_preds = (preds == label)
                label_truth = (labels == label)
                tp[label] += (label_preds & label_truth).sum().cpu()
                fp[label] += (label_preds & ~label_truth).sum().cpu()
                fn[label] += (~label_preds & label_truth).sum().cpu()

            # compute the accumulated metrics
            per_label_precision = tp.float() / (tp + fp + eps)
            per_label_recall = tp.float() / (tp + fn + eps)
            per_label_f1 = 2 * per_label_precision * per_label_recall / (per_label_precision + per_label_recall + eps)
            support = (tp + fn) > 0  # ignore classes not seen so far

            precision = per_label_precision[support].mean().item()
            recall = per_label_recall[support].mean().item()
            f1 = per_label_f1[support].mean().item()
            accuracy = correct / total
            avg_loss = running_loss / total

            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                acc=f"{accuracy*100:.2f}%",
                prec=f"{precision*100:.2f}%",
                rec=f"{recall*100:.2f}%",
                F1=f"{f1*100:.2f}%"
            )

    # final epoch metrics (metrics in loop are running estimates)
    accuracy = correct / total
    per_label_precision = tp.float() / (tp + fp + eps) #
    per_label_recall  = tp.float() / (tp + fn + eps)
    per_label_f1 = 2 * per_label_precision * per_label_recall / (per_label_precision + per_label_recall + eps)
    support = (tp + fn) > 0
    precision = per_label_precision[support].mean().item()
    recall = per_label_recall[support].mean().item()
    f1 = per_label_f1[support].mean().item()

    metrics = {
        'loss': running_loss / total,   # avg negative log likelihood
        'accuracy': accuracy,           # how many predictions were correct
        'precision': precision,         # purity; high prec -> minimize false positives
        'recall': recall,               # sensitivity; high recall -> minimize false negatives
        'f1': f1,                       # balance of precision and recall
    }

    # early stopping check -> parent train fn should handle the early_stopper.stop_now
    if early_stopper is not None and early_stopper.enabled:
        stop_now, _ = early_stopper.step(metrics)

    return metrics