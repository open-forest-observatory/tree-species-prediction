import torch
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import json
import yaml
from dataclasses import asdict
import matplotlib.pyplot as plt
import copy

import _bootstrap
from configs.model_config import model_config
from configs.path_config import path_config
from configs.data_reduction_config import dr_config
from training_utils.initializers import init_training
from training_utils.step_epoch import _step_epoch
from training_utils.visualization import confusion_matrix
from training_utils.data_utils import get_classes_from_gpd_file_paths, summarize_split_by_tree, check_no_tree_overlap
from training_utils.ctx import vram_ctx

from training_utils.data_reduction.selection import GradMatchPBSelector 
from training_utils.data_reduction.utils import make_selection_loader, rebuild_train_loader

''' TODO:
Implement:
- configurable fc layers for classifier head
'''

def train():
    tb_writer = SummaryWriter(log_dir=model_config.cur_training_out_dir / "tensorboard_logs")

    # save current model config to yaml file
    with open(model_config.cur_training_out_dir / "cfg_dump.yaml", 'w') as cfg_file:
        cfg_dict = asdict(model_config)
        yaml.dump(cfg_dict, cfg_file, default_flow_style=False)

    # grab species labels and map them to idxs (labels sorted alphabetically)
    # might be useful later for referencing
    gpkg_dsets = list(Path(path_config.drone_crowns_with_field_attributes).glob("*"))
    unique_species_labels = get_classes_from_gpd_file_paths(gpkg_dsets)

    with vram_ctx('INIT'):
        # init everything required for model training
        tree_model, tree_dset, train_loader, val_loader, train_subset, val_subset, static_transform, \
        train_transform, val_transform, optim, criterion, scheduler, scaler, early_stopper = init_training()

    # for data reduction
    subset_selector, selection_loader, omp_diag = None, None, None
    if model_config.use_data_reduction:
        # for choosing optimal data points we only want to look at transformed imgs w/o rng dependence (just scaling and normalizing)
        dset_copy = copy.copy(tree_dset)
        full_train_loader = copy.copy(train_loader)
        train_dset_val_transform = Subset(dset_copy, train_subset.indices)
        train_dset_val_transform.dataset.static_transform = static_transform
        train_dset_val_transform.dataset.random_transform = val_transform

        visuals_info = { # miscellaneous info needed for metrics/plotting/visualizations purposes; not necessary for core method
            'backbone_data_mean': tree_model.backbone_data_cfg['mean'],
            'backbone_data_std': tree_model.backbone_data_cfg['std'],
            'train_transform': train_transform
        }

        selection_loader = make_selection_loader(tree_dset, train_subset, static_transform, val_transform)
        subset_selector = GradMatchPBSelector(
            model=tree_model,
            criterion=criterion,
            device=model_config.device,
            lam=dr_config.omp_regularizer_strength,
            eps=dr_config.omp_tol,
            strategy=dr_config.strategy,
            log_dir=model_config.cur_training_out_dir / 'data_reduction',
            visuals_info=visuals_info
        )

    # VRAM simulation mode: quickly test memory usage without full training
    if dr_config.simulate_vram:
        if subset_selector is None:
            print("VRAM simulation requires use_data_reduction=True to test gradient matrix allocation")
            return
        success = subset_selector.simulate_vram(selection_loader, train_loader)
        print(f"VRAM simulation {'PASSED' if success else 'FAILED'}")
        return

    prev_sample_idxs, cur_batch_weights, omp_diag = None, None, None # for tracking each iteration's similarity and weights
    sample_weight_map = None  # persists between epochs; None during warm-start, then set by selection

    n_layers_unfrozen = 0
    for epoch in range(model_config.epochs):
        # toggle backbone trainability and add its params to optimizer once `freeze_backbone_epochs` reached
        if epoch >= model_config.freeze_backbone_epochs and n_layers_unfrozen <= model_config.n_last_layers_to_unfreeze:
            n_layers_unfrozen += model_config.layer_unfreeze_step
            tree_model.unfreeze_last_n_backbone_layers(n=n_layers_unfrozen)

        # Data reduction -> GradMatchPB
        # During warm-start: train on full data (sample_weight_map remains None)
        # At selection epochs: recompute subset and weights
        # Between selection epochs: reuse previous subset and weights
        if subset_selector is not None and subset_selector._is_subset_epoch(epoch, model_config.epochs):
            # optionally reshuffle training indices so batch compositions differ each selection round
            if dr_config.shuffle_before_selection:
                selection_loader = make_selection_loader(
                    tree_dset, train_subset, static_transform, val_transform,
                    shuffle_indices=True,
                )
            chosen_subset_indices, sample_weight_map, omp_diag = subset_selector.select_perbatch(
                selection_loader=selection_loader,
                subset_ratio=dr_config.subset_ratio, positive=True,
                save_plots=dr_config.save_plots, save_images=dr_config.save_n_omp_rated_imgs
            )
            # rebuild training loader on the selected indices training transforms
            train_loader = rebuild_train_loader(train_subset, chosen_subset_indices)
        # Note: sample_weight_map is preserved from last selection epoch (or None during warm-start)
            
        # train one epoch
        # if data reduction enabled and is subset epoch -> train with full data and pool gradients
        # if data reduction enabled and is NOT subset epoch -> train with previously computed subset
        train_metrics = _step_epoch(
            tree_model, train_loader, model_config.device, criterion,
            optim, scaler, training=True, epoch_num=epoch+1,
            sample_weight_map=sample_weight_map
        )
        scheduler.step()

        # validation -> use same train epoch fn but with training=False
        with torch.no_grad():
            val_metrics = _step_epoch(
                tree_model, val_loader, model_config.device, criterion,
                optim=None, scaler=None, training=False, epoch_num=epoch+1,
                sample_weight_map=None
            )

            # Compute confusion matrix for validation set
            matrix_fig = confusion_matrix(unique_species_labels, val_loader, tree_model, model_config.device, exclude_empty=True)
            tb_writer.add_figure("Val Confusion Matrix", matrix_fig, epoch + 1)
            plt.close(matrix_fig)
        
        # early stopping check
        if early_stopper is not None and early_stopper.enabled:
            early_stopper.step(epoch, val_metrics)

        # Logging train metrics
        for key, value in train_metrics.items():
            tb_writer.add_scalar(f"Train/{key}", value, epoch + 1)

        # Logging validation metrics
        for key, value in val_metrics.items():
            tb_writer.add_scalar(f"Val/{key}", value, epoch + 1)
        # Log learning rate
        tb_writer.add_scalar("Learning_Rate", scheduler.get_last_lr()[0], epoch + 1)

        #prev_sample_idxs = chosen_idxs_pool

        # save ckpt for future analysis
        ckpt_path = model_config.cur_training_out_dir / f"ckpt_epoch-{epoch+1}_valF1-{val_metrics['f1_macro']:.4f}.pt"
        log_path = ckpt_path.parent / f"{ckpt_path.stem}.json"
        torch.save({
            'epoch': epoch + 1,
            'model_state': tree_model.state_dict(),
            'optim_state': optim.state_dict(),
            'sched_state': scheduler.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'omp_diag': omp_diag,
        }, ckpt_path)
        print(f"*** Saved checkpoint to: {ckpt_path}")

        with open(log_path, 'w') as log_file:
            epoch_log = {'epoch': epoch+1, 'train_metrics': train_metrics, 'val_metrics': val_metrics, 'omp_diag': omp_diag}
            # DEBUG: temporary will remove soon; verifies dtypes of all logged materials
            '''for ko, vo in epoch_log.items():
                print(f"\n\n\n*** {ko}\n")
                if isinstance(vo, dict):
                    for ki, vi in vo.items():
                        print(f"ki: {ki}\ntype_vi:{type(vi)}\nvi:{vi}")'''
            json.dump(epoch_log, log_file, indent=4)

        if early_stopper is not None and early_stopper.stopped:
            print(f"No improvements of validation metric {early_stopper.monitor_metric} in the last {early_stopper.patience} epochs. Stopping training...")
            break

    tb_writer.close()

if __name__ == '__main__':
    train()
