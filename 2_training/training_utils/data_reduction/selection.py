import math
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import _bootstrap
from configs.data_reduction_config import dr_config
from configs.path_config import path_config
from training_utils.data_reduction.omp import OrthogonalMP_REG_Parallel_V1
from training_utils.data_reduction.metrics import omp_diagnostics
from training_utils.data_reduction.plotting import plot_projection_sorted, plot_singleton_quality


class GradMatchPBSelector:
    def __init__(self, model, criterion, device, lam=0.0, eps=1e-4, strategy='gradmatch'):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.lam = lam
        self.eps = eps
        self.strategy = strategy
        self.epoch = 0

    def _batch_grad_head(self, imgs, labels):
        """
        Batch gradient atom = grad(loss_mean, head_params) flattened to (d,)
        """
        self.model.eval()
        head_params = self.model.head_parameters()
        if not head_params:
            raise RuntimeError("No trainable classifier_head params found for selection.")

        logits = self.model(imgs)
        loss = self.criterion(logits, labels)  # mean CE by default

        grads = torch.autograd.grad(
            loss,
            head_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        g = torch.cat([gi.reshape(-1) for gi in grads], dim=0).detach()

        # IMPORTANT: keep OMP numerically stable
        return g.float()

    def _is_subset_epoch(self, epoch, total_epochs):
        self.epoch = epoch
        return (
            (epoch <= total_epochs) and  # if we haven't trained for the full spec'd num epochs and,
            (epoch >= dr_config.num_warm_start_epochs) and  # if we have trained enough on the full data and,
            (epoch % dr_config.epoch_selection_interval == 0)  # we are at a selection interval epoch,
        )

    def select_perbatch(self, selection_loader, subset_ratio, **kwargs):
        strategy_fn = {
            "gradmatch": self.gradmatch_select,
            "random": self.random_select,
        }.get(self.strategy)
    
        if strategy_fn is None:
            raise ValueError(f"Error: Data reduction strategy should be either 'gradmatch' or 'random'; Found: {self.strategy}")

        return strategy_fn(selection_loader, subset_ratio, **kwargs)

    def random_select(self, selection_loader, subset_ratio, positive=True, save_plots=False):
        """
        Random per-batch selection baseline.

        Returns exactly the same objects as select_perbatch():
            chosen_subset_indices : list[int]   (subset indices into selection_loader.dataset)
            sample_weight_map     : dict[int -> float]  (base_ds_idx -> weight)
            diag                  : dict
        """
        assert 0 < subset_ratio <= 1.0

        # deterministic batch structure (same as GradMatchPB)
        batch_wise_indices = list(selection_loader.batch_sampler)
        n_batches = len(batch_wise_indices)
        k_batches = max(1, int(math.ceil(subset_ratio * n_batches)))

        # --- randomly select batches ---
        chosen_batch_ids = random.sample(range(n_batches), k_batches)

        sel_subset = selection_loader.dataset
        base_indices = sel_subset.indices  # subset_idx -> base_ds_idx

        chosen_subset_indices = []
        sample_weight_map = {}

        for bid in tqdm(
            chosen_batch_ids,
            desc=f"RandomPB: expanding {k_batches} batches -> samples",
            dynamic_ncols=True,
        ):
            subset_batch_idxs = batch_wise_indices[bid]
            chosen_subset_indices.extend(subset_batch_idxs)

            # uniform weight (baseline)
            for sub_i in subset_batch_idxs:
                base_i = int(base_indices[int(sub_i)])
                sample_weight_map[base_i] = 1.0

        diag = {
            "method": "random_perbatch",
            "n_batches_total": n_batches,
            "k_batches": k_batches,
            "reg_nnz": k_batches,                     # analogous to nnz
            "n_points_selected": len(chosen_subset_indices),
            "subset_ratio": subset_ratio,
        }

        # match GradMatch behavior: attach as attribute
        self.last_omp_diag = diag

        return chosen_subset_indices, sample_weight_map, diag

    def gradmatch_select(self, selection_loader, subset_ratio, positive=True, save_plots=False):
        """This returns chosen_subset_indices relative to selection_loader.dataset (which is the train split)

        Returns:
            chosen_subset_indices: list[int] (indices w.r.t. selection_loader.dataset)
            sample_weight_map: dict[int -> float]  (global_idx -> gamma)
            diag: dict
        """
        assert 0 < subset_ratio <= 1.0

        # batch_sampler yields the dataset indices for each batch (deterministic because shuffle=False)
        batch_wise_indices = list(selection_loader.batch_sampler)
        n_batches = len(batch_wise_indices)
        k_batches = max(1, int(math.ceil(subset_ratio * n_batches)))

        # build grad atoms (one per batch)
        grad_list = []
        grad_pbar = tqdm(
            selection_loader,
            desc=f"GradMatchPB: computing {n_batches} batch-grads",
            leave=True,
            dynamic_ncols=True,
        )
        for imgs, labels, _metas in grad_pbar:
            imgs, labels = imgs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            g_b = self._batch_grad_head(imgs, labels)
            grad_list.append(g_b)

        grads_per_batch = torch.stack(grad_list, dim=0)          # (n_batches, d)
        A = grads_per_batch.transpose(0, 1).contiguous()         # (d, n_batches)
        b = grads_per_batch.sum(dim=0).contiguous()              # (d,)

        # OMP on GPU
        print(f"Gradient Matrix shape: {A.shape} (n_class_head_params x n_batches)")
        reg = OrthogonalMP_REG_Parallel_V1(
            A, b,
            nnz=k_batches,
            positive=positive,
            lam=self.lam,
            tol=self.eps,
            device=str(self.device),
        )  # (n_batches,)

        chosen_batch_ids = torch.nonzero(reg).view(-1).tolist()

        # selection_loader.dataset is a Subset(sel_cp, train_subset.indices)
        sel_subset = selection_loader.dataset
        base_indices = sel_subset.indices   # maps subset_idx -> base_ds_idx
        chosen_subset_indices = []          # subset indices into sel_subset (0..len(sel_subset)-1)
        sample_weight_map = {}              # base_ds_idx -> gamma

        gradmatch_pbar = tqdm(chosen_batch_ids, desc=f"GradMatchPB: expanding {len(chosen_batch_ids)} batches -> samples", dynamic_ncols=True)
        for bid in gradmatch_pbar:
            gamma = float(reg[bid].item())
            subset_batch_idxs = batch_wise_indices[bid]  # subset indices into sel_subset

            chosen_subset_indices.extend(subset_batch_idxs)

            # key weights by BASE dataset index (stable across any future Subset nesting)
            for sub_i in subset_batch_idxs:
                base_i = int(base_indices[sub_i])
                sample_weight_map[base_i] = gamma

        diag = {
            "n_batches_total": n_batches,
            "k_batches": k_batches,
            "reg_nnz": len(chosen_batch_ids),
            "n_points_selected": len(chosen_subset_indices),
            "subset_ratio": subset_ratio,
            **omp_diagnostics(A, b, reg, k=k_batches, positive=positive)
        }

        # visuals for omp
        if save_plots:
            plot_fp = path_config.training_ckpt_dir / 'metrics'
            plot_projection_sorted(diag['proj_vals'], plot_fp, epoch=self.epoch, title_prefix="OMP")
            plot_singleton_quality(
                diag['topk_cos'], diag['topk_res_ratio'], 
                diag['bottomk_cos'], diag['bottomk_res_ratio'],
                plot_fp, epoch=self.epoch
            )

        return chosen_subset_indices, sample_weight_map, diag

