import torch

from pathlib import Path
from tqdm import tqdm
from datetime import datetime

import _bootstrap
from configs.model_config import model_config
from configs.path_config import path_config
from training_utils.initializers import init_training
from training_utils.step_epoch import _step_epoch
from training_utils.data_utils import get_classes_from_gpd_file_paths, stratified_split

''' TODO:
- output model cfg to yaml in ckpt dir
- metrics:
    - accuracy
    - precision
    - f1 score
    - confusion matrix
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
    cur_training_out_dir = path_config.training_ckpt_dir / ts
    cur_training_out_dir.mkdir(parents=True)

    # init everything required for model training
    tree_model, tree_dset, train_loader, val_loader, optim, criterion, scheduler, scaler, device = init_training()
    
    # sanity check test (comment out for actual training)
    imgs, labels, metas = next(iter(train_loader))
    print(imgs.shape, labels.shape, type(metas), len(metas))

    pbar = tqdm(range(model_config.epochs))
    for epoch in pbar:
        # toggle backbone trainability and add its params to optimizer once `freeze_backbone_epochs` reached
        if epoch == model_config.freeze_backbone_epochs:
            tree_model.toggle_backbone_weights_trainability(True)
            backbone_trainable_params = [p for p in tree_model.backbone.parameters() if p.required_grad]
            optim.add_param_group({"backbone_params": backbone_trainable_params})

        # train one step
        train_metrics = _step_epoch(
            tree_model, train_loader, device,
            optim, criterion, scaler,
            training=True
        )
        scheduler.step()

        # validation -> use same train epoch fn but with training=False
        with torch.no_grad():
            val_metrics = _step_epoch(
                tree_model, val_loader, device,
                optim=None, criterion=None, scaler=None,
                training=False
            )

        # save ckpt for future analysis
        torch.save({
            'epoch': epoch + 1,
            'model_state': tree_model.state_dict(),
            'optim_state': optim.state_dict(),
            'sched_state': scheduler.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }, cur_training_out_dir / f'{epoch+1}.pt')

if __name__ == '__main__':
    train()