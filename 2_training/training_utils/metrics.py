import torch

def update_tp_fp_fn(tp, fp, fn, preds, labels, num_classes):
    """
    Vectorized update of per-class TP/FP/FN for single-label multiclass classification.
    tp, fp, fn: (C,) long tensors on CPU
    preds, labels: (B,) tensors (any device)
    """
    preds_cpu = preds.detach().to("cpu")
    labels_cpu = labels.detach().to("cpu")

    # TP: predicted == truth == c
    tp += torch.bincount(labels_cpu[preds_cpu == labels_cpu], minlength=num_classes).to(tp.dtype)

    # Predicted counts per class
    pred_cnt = torch.bincount(preds_cpu, minlength=num_classes).to(tp.dtype)
    # True counts per class (support)
    true_cnt = torch.bincount(labels_cpu, minlength=num_classes).to(tp.dtype)

    fp += (pred_cnt - torch.bincount(labels_cpu[preds_cpu == labels_cpu], minlength=num_classes).to(tp.dtype))
    fn += (true_cnt - torch.bincount(labels_cpu[preds_cpu == labels_cpu], minlength=num_classes).to(tp.dtype))


def compute_epoch_metrics(running_loss, total, correct, tp, fp, fn, average="macro", eps=1e-12):
    """
    average:
      - "macro": mean of per-class metrics over classes with support > 0
      - "micro": global P/R/F1 from summed TP/FP/FN
      - "weighted": per-class metrics weighted by support (tp+fn)
    """
    assert average in {"macro", "micro", "weighted"}

    avg_loss = running_loss / max(total, 1)

    if average == "micro":
        tp_sum = tp.sum().float()
        fp_sum = fp.sum().float()
        fn_sum = fn.sum().float()

        precision = (tp_sum / (tp_sum + fp_sum + eps)).item()
        recall = (tp_sum / (tp_sum + fn_sum + eps)).item()
        f1 = (2 * tp_sum / (2 * tp_sum + fp_sum + fn_sum + eps)).item()
    else:
        tp_f = tp.float()
        fp_f = fp.float()
        fn_f = fn.float()

        per_p = tp_f / (tp_f + fp_f + eps)
        per_r = tp_f / (tp_f + fn_f + eps)
        per_f1 = 2 * per_p * per_r / (per_p + per_r + eps)

        support = (tp + fn).float()
        seen = support > 0

        if average == "macro":
            precision = per_p[seen].mean().item() if seen.any() else 0.0
            recall = per_r[seen].mean().item() if seen.any() else 0.0
            f1 = per_f1[seen].mean().item() if seen.any() else 0.0
        else:  # weighted
            w = support[seen]
            wsum = (w.sum() + eps)
            precision = (per_p[seen] * w).sum().item() / wsum.item()
            recall = (per_r[seen] * w).sum().item() / wsum.item()
            f1 = (per_f1[seen] * w).sum().item() / wsum.item()

    accuracy = correct / max(total, 1)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average": average,
    }