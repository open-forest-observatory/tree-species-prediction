import torch
from torch.amp import autocast
from collections import defaultdict, Counter

from tqdm import tqdm

from configs.model_config import model_config

def _step_epoch(tree_model, dataloader, device, criterion, optim=None, scaler=None, early_stopper=None, training=False, epoch_num=None):
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

    # Tree-level aggregation tracking
    tree_predictions = defaultdict(list)  # Pred for each image of each tree {tree_id: [pred1, pred2, ...]}
    tree_confidences = defaultdict(list)  # Confidence score {tree_id: [conf1, conf2, ...]}
    tree_labels = {}  # {tree_id: true_label}

    # iterate through batches
    pbar = tqdm(dataloader, desc=pbar_msg)
    for batch_idx, (imgs, labels, metas) in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
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

            # Collect tree-level predictions (only for validation)
            if not training:
                probabilities = torch.softmax(logits, dim=1)
                confidences, _ = torch.max(probabilities, dim=1) # we ignore indices here because we have preds already
                
                for i in range(batch_size):
                    tree_id = metas[i]['unique_treeID']
                    tree_predictions[tree_id].append(preds[i].item())
                    tree_confidences[tree_id].append(confidences[i].item())
                    tree_labels[tree_id] = labels[i].item()

            # init per class metrics counters on first pass (if num_classes is still none)
            if num_classes is None:
                num_classes = logits.size(1) # length of last layer output equates to num classes
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

    # Add tree-level metrics for validation
    if not training and tree_predictions:
        tree_metrics = compute_tree_level_metrics(tree_predictions, tree_labels, tree_confidences)
        metrics.update(tree_metrics)

    # early stopping check -> parent train fn should handle the early_stopper.stop_now
    if early_stopper is not None and early_stopper.enabled:
        stop_now, _ = early_stopper.step(metrics)

    return metrics

def compute_tree_level_metrics(tree_predictions, tree_labels, tree_confidences):
    """For each tree, what's our best single prediction when we aggregate it across images?"""

    """
    Example imput:

    tree_predictions = {
    "tree_A": [0, 2, 0, 0],     # 4 images: mostly class 0
    "tree_B": [1, 1, 2]        # 3 images: mostly class 1
    }

    tree_labels = {
        "tree_A": 0,       # GT is class 0
        "tree_B": 1        # GT is class 1  
    }

    tree_confidences = {
        "tree_A": [0.9, 0.6, 0.8, 0.7],
        "tree_B": [0.5, 0.9, 0.4]
    }
    """
    # Method 1: Majority vote aggregation
    # majority_preds = {"tree_A": 0, "tree_B": 1}
    
    # Method 2: Confidence-weighted aggregation
    # For each class this will sum up the confidence scores by class, and pick class with the highest
    """
    A scenario where this might be better:

    tree_predictions = [1, 1, 1, 0, 0]
    tree_confidences = [0.51, 0.52, 0.53, 0.95, 0.94]
    
    Majority vote -> class 1
    Confidence weighted -> class 0 (0.95 + 0.94 > 0.51 + 0.52 + 0.53)

    """
    
    # return metrics