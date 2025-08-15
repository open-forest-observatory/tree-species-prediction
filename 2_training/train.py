import torch
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json, yaml
from dataclasses import asdict
import matplotlib.pyplot as plt

import _bootstrap
from configs.model_config import model_config
from configs.path_config import path_config
from training_utils.initializers import init_training
from training_utils.step_epoch import _step_epoch
from training_utils.data_utils import get_classes_from_gpd_file_paths, stratified_split
from training_utils.visualization import confusion_matrix
from torch.utils.tensorboard import SummaryWriter

''' TODO:
- output model cfg to yaml in ckpt dir
- 
- context manager for safe exiting on training
- plotting/visualizing of metrics and loss
'''

def train():
    # grab species labels and map them to idxs (labels sorted alphabetically)
    # might be useful later for referencing
    gpkg_dsets = list(Path(path_config.drone_crowns_with_field_attributes).glob("*"))
    unique_species_labels = get_classes_from_gpd_file_paths(gpkg_dsets)

    # for naming ckpt dir
    ts = datetime.now().strftime('%m%d-%H%M%S')
    cur_training_out_dir = path_config.training_ckpt_dir / f"{ts}-{model_config.ckpt_dir_tag}"
    cur_training_out_dir.mkdir(parents=True)
    tb_writer = SummaryWriter(log_dir=cur_training_out_dir / "tensorboard_logs")

    # save current model config to yaml file
    with open(cur_training_out_dir / "cfg_dump.yaml", 'w') as cfg_file:
        cfg_dict = asdict(model_config)
        yaml.dump(cfg_dict, cfg_file, default_flow_style=False)

    # init everything required for model training
    tree_model, tree_dset, train_loader, val_loader, optim, criterion, scheduler, scaler, device, early_stopper = init_training()

    # count n_samples per class
    labels = torch.tensor([m['label_idx'] for m in tree_dset.meta])
    unique, counts = torch.unique(labels, return_counts=True)
    for idx, count in zip(unique.tolist(), counts.tolist()):
        print(tree_dset.idx2label_map[idx], count)

    # sanity check test (comment out for actual training)
    #imgs, labels, metas = next(iter(train_loader))
    #print(imgs.shape, labels.shape, type(metas), len(metas))
    #print(len(train_loader), len(val_loader))

    n_layers_unfrozen = 0
    for epoch in range(model_config.epochs):
        # toggle backbone trainability and add its params to optimizer once `freeze_backbone_epochs` reached
        if epoch >= model_config.freeze_backbone_epochs and n_layers_unfrozen <= model_config.n_last_layers_to_unfreeze:
            n_layers_unfrozen += model_config.layer_unfreeze_step
            tree_model.unfreeze_last_n_backbone_layers(n=n_layers_unfrozen)
        else:
            early_stopper.enabled


        # keep early_stopper disabled until after opening all layers, since it dips for a bit when unfreezing

            
        # train one step
        train_metrics = _step_epoch(
            tree_model, train_loader, device, criterion,
            optim, scaler, early_stopper=None, training=True, epoch_num=epoch+1
        )
        scheduler.step()

        # validation -> use same train epoch fn but with training=False
        with torch.no_grad():
            val_metrics = _step_epoch(
                tree_model, val_loader, device, criterion,
                optim=None, scaler=None, early_stopper=early_stopper, training=False, epoch_num=epoch+1
            )

            # Compute confusion matrix for validation set
            matrix_fig = confusion_matrix(unique_species_labels, val_loader, tree_model, device)
            tb_writer.add_figure("Confusion Matrix", matrix_fig, epoch + 1)
            plt.close(matrix_fig)

        # Logging train metrics
        for key, value in train_metrics.items():
            tb_writer.add_scalar(f"Train/{key}", value, epoch + 1)

        # Logging validation metrics
        for key, value in val_metrics.items():
            tb_writer.add_scalar(f"Val/{key}", value, epoch + 1)
        # Log learning rate
        tb_writer.add_scalar("Learning_Rate", scheduler.get_last_lr()[0], epoch + 1)


        # save ckpt for future analysis
        ckpt_path = cur_training_out_dir / f"ckpt_epoch-{epoch+1}_valF1-{val_metrics['f1']:.4f}-.pt"
        log_path = ckpt_path.parent / f"{ckpt_path.stem}.json"
        torch.save({
            'epoch': epoch + 1,
            'model_state': tree_model.state_dict(),
            'optim_state': optim.state_dict(),
            'sched_state': scheduler.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }, ckpt_path)
        print(f"*** Saved checkpoint to: {ckpt_path}")

        with open(log_path, 'w') as log_file:
            epoch_log = {'epoch': epoch+1, 'train_metrics': train_metrics, 'val_metrics': val_metrics}
            json.dump(epoch_log, log_file, indent=4)

        if early_stopper is not None and early_stopper.stopped:
            print(f"No improvements of validation metric {early_stopper.monitor_metric} in the last {early_stopper.patience} epochs. Stopping training...")
            break

    tb_writer.close()

if __name__ == '__main__':
    train()