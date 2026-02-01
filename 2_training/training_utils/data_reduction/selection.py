import math
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from torchvision.transforms.functional import to_pil_image

import _bootstrap
from configs.data_reduction_config import dr_config
from configs.path_config import path_config
from data.dataset import collate_batch
from training_utils.image_processing import unnormalize
from training_utils.ctx import vram_ctx
from training_utils.data_reduction.omp import OrthogonalMP_REG_Parallel_V1
from training_utils.data_reduction.metrics import omp_diagnostics, compute_pairwise_jaccard_similarity
from training_utils.data_reduction.plotting import plot_projection_sorted, plot_singleton_quality


class GradMatchPBSelector:
    def __init__(self, model, criterion, device, lam=0.0, eps=1e-4, strategy='gradmatch', log_dir=None, visuals_info=None):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.lam = lam
        self.eps = eps
        self.strategy = strategy
        self.epoch = 0
        self.log_dir = log_dir
        
        # for visualization/metrics/logging purposes
        self.visuals_info = visuals_info
        self.backbone_data_mean = visuals_info['backbone_data_mean']
        self.backbone_data_std = visuals_info['backbone_data_std']
        self.train_transform = visuals_info['train_transform']
        self.plot_dir = self.log_dir / 'plots'
        self.imgs_dir = self.log_dir / 'imgs'
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.imgs_dir.mkdir(parents=True, exist_ok=True)


    def _batch_grad_head(self, imgs, labels):
        """
        Batch gradient atom = grad(loss, head_params) flattened to (d,).
        Uses .sum() reduction on loss to match CORDS semantics.

        If dr_config.use_closed_form_grads is True, uses closed-form gradient
        for the final linear layer only (faster but requires linear final layer).
        """
        self.model.eval()

        if dr_config.use_closed_form_grads:
            return self._batch_grad_closed_form(imgs, labels)
        else:
            return self._batch_grad_autograd(imgs, labels)

    def _batch_grad_autograd(self, imgs, labels):
        """
        Compute gradients via autograd. Works with any head architecture.
        Uses loss.sum() to match CORDS gradient scaling.
        """
        head_params = self.model.head_parameters()
        if not head_params:
            raise RuntimeError("No trainable classifier_head params found for selection.")

        logits = self.model(imgs)
        # Use sum reduction to match CORDS (not mean)
        loss = F.cross_entropy(logits, labels, reduction='sum')

        grads = torch.autograd.grad(
            loss,
            head_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        g = torch.cat([gi.reshape(-1) for gi in grads], dim=0).detach()

        return g.float()

    def _batch_grad_closed_form(self, imgs, labels):
        """
        Closed-form gradient computation for the final linear layer.
        This is faster than autograd but only computes gradients for the last layer.

        For a linear layer: logits = W @ prelogit + bias
          - dL/d_bias = dL/d_logits (summed over batch)
          - dL/d_W = sum_i( outer(dL/d_logits[i], prelogit[i]) )

        Returns gradient vector of shape (out_features + out_features * in_features,)
        i.e., [bias_grad, weight_grad.flatten()]
        """
        # get logits and pre-logit features (input to final linear)
        logits, prelogit = self.model.forward_with_prelogit(imgs)

        # compute per-sample loss and gradient of loss w.r.t. logits
        # dL/d_logits = softmax(logits) - one_hot(labels)  for cross-entropy
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)  # (B, num_classes)
            one_hot = F.one_hot(labels, num_classes=logits.size(1)).float()  # (B, num_classes)
            # gradient of cross-entropy loss w.r.t. logits (summed, not averaged)
            dl_dlogits = probs - one_hot  # (B, num_classes)

        # closed-form gradients for final linear layer
        # bias gradient: sum over batch of dL/d_logits
        bias_grad = dl_dlogits.sum(dim=0)  # (num_classes,)

        # weight gradient: sum_i( outer(dL/d_logits[i], prelogit[i]) )
        # = dl_dlogits.T @ prelogit  -> (num_classes, prelogit_dim)
        weight_grad = dl_dlogits.T @ prelogit.detach()  # (num_classes, prelogit_dim)

        # flatten and concatenate: [bias, weight.flatten()]
        g = torch.cat([bias_grad.flatten(), weight_grad.flatten()], dim=0)

        return g.float()

    def _is_subset_epoch(self, epoch, total_epochs):
        self.epoch = epoch
        return (
            (epoch <= total_epochs) and  # if we haven't trained for the full spec'd num epochs and,
            (epoch >= dr_config.num_warm_start_epochs) and  # if we have trained enough on the full data and,
            (epoch % dr_config.epoch_selection_interval == 0)  # we are at a selection interval epoch,
        )

    def select_perbatch(self, selection_loader, subset_ratio, **kwargs):
        # route based on class balancing mode
        if dr_config.force_class_balancing:
            strategy_fn = {
                "gradmatch": self.gradmatch_select_perclass,
                "random": self.random_select_perclass,
            }.get(self.strategy)
        else:
            strategy_fn = {
                "gradmatch": self.gradmatch_select,
                "random": self.random_select,
            }.get(self.strategy)

        if strategy_fn is None:
            raise ValueError(f"Error: Data reduction strategy should be either 'gradmatch' or 'random'; Found: {self.strategy}")

        return strategy_fn(selection_loader, subset_ratio, **kwargs)

    def _extract_labels(self, selection_loader):
        """
        Extract all labels from the selection_loader's dataset.
        Returns:
            labels: torch.LongTensor of shape (N,) where N is len(selection_loader.dataset)
        """
        sel_subset = selection_loader.dataset
        labels = []
        for i in range(len(sel_subset)):
            # sel_subset[i] returns (img, label, meta)
            _, label, _ = sel_subset[i]
            labels.append(label)
        return torch.tensor(labels, dtype=torch.long)

    def _compute_gradients_for_loader(self, loader, desc_prefix=""):
        """
        Compute per-sample gradients for all samples in a loader.
        Returns:
            grads: torch.Tensor of shape (N, d) where N is num samples, d is grad dim
        """
        grad_list = []
        grad_pbar = tqdm(
            loader,
            desc=f"{desc_prefix}computing sample grads",
            leave=False,
            dynamic_ncols=True,
        )
        for batch_idx, (imgs, labels, _metas) in enumerate(grad_pbar):
            with vram_ctx('DR-FORWARD-perclass', step=batch_idx, epoch=self.epoch):
                imgs = imgs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                # compute per-sample gradients (batch_size=1 for per-sample)
                g = self._batch_grad_head(imgs, labels)
                grad_list.append(g)
        return torch.stack(grad_list, dim=0)  # (N, d)

    def _run_omp_and_collect(self, A, b, budget, positive, base_indices, sample_indices):
        """
        Run OMP and collect selected indices and weights.

        Args:
            A: (d, n) gradient matrix
            b: (d,) target gradient
            budget: number of samples to select
            positive: enforce positive coefficients
            base_indices: mapping from local idx to base dataset idx
            sample_indices: mapping from local idx to subset idx

        Returns:
            chosen_subset_indices: list of subset indices
            sample_weight_map: dict mapping base_ds_idx -> gamma
            reg: OMP coefficients tensor
        """
        reg = OrthogonalMP_REG_Parallel_V1(
            A, b,
            nnz=budget,
            positive=positive,
            lam=self.lam,
            tol=self.eps,
            device='cpu',
        )

        chosen_local_ids = torch.nonzero(reg).view(-1).tolist()
        chosen_subset_indices = []
        sample_weight_map = {}

        for lid in chosen_local_ids:
            gamma = float(reg[lid].item())
            sub_i = sample_indices[lid]
            base_i = int(base_indices[sub_i])
            chosen_subset_indices.append(sub_i)
            sample_weight_map[base_i] = gamma

        return chosen_subset_indices, sample_weight_map, reg

    def random_select(self, selection_loader, subset_ratio, positive=True, save_plots=False, save_images=0):
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

        # randomly select batches 
        chosen_batch_ids = np.random.default_rng().choice(range(n_batches), k_batches)

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

        self.last_omp_diag = diag

        return chosen_subset_indices, sample_weight_map, diag

    def gradmatch_select(self, selection_loader, subset_ratio, positive=True, save_plots=False, save_images=0):
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
        for batch_idx, (imgs, labels, _metas) in enumerate(grad_pbar):
            with vram_ctx('DR-FORWARD', step=batch_idx, epoch=self.epoch):
                imgs, labels = imgs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                g_b = self._batch_grad_head(imgs, labels)
                grad_list.append(g_b)

        with vram_ctx('DR-assemble-gradmat', epoch=self.epoch):
            grads_per_batch = torch.stack(grad_list, dim=0)          # (n_batches, d)
            A = grads_per_batch.transpose(0, 1).contiguous()         # (d, n_batches)
            b = grads_per_batch.sum(dim=0).contiguous()              # (d,)

        # OMP on GPU
        print(f"Gradient Matrix shape: {A.shape} (n_class_head_params x n_batches)")
        with vram_ctx('DR-omp', epoch=self.epoch):
            reg = OrthogonalMP_REG_Parallel_V1(
                A, b,
                nnz=k_batches,
                positive=positive,
                lam=self.lam,
                tol=self.eps,
                #device=str(self.device),
                device='cpu',
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

        # Note: more vram logging in training_utils.data_reduction.metrics.omp_diagnostics()
        extended_omp_diag = omp_diagnostics(A, b, reg, k=k_batches, positive=positive, device='cpu', epoch=self.epoch)
        diag = {
            "n_batches_total": n_batches,
            "k_batches": k_batches,
            "reg_nnz": len(chosen_batch_ids),
            "n_points_selected": len(chosen_subset_indices),
            "subset_ratio": subset_ratio,
            **extended_omp_diag,
        }

        # visuals for omp
        if save_plots and self.plot_dir is not None:
            plot_projection_sorted(diag['proj_vals'], self.plot_dir, epoch=self.epoch, title_prefix="OMP")
            plot_singleton_quality(
                diag['topk_cos'], diag['topk_res_ratio'], 
                diag['bottomk_cos'], diag['bottomk_res_ratio'],
                self.plot_dir, epoch=self.epoch
            )

        # save omp rated best, worst, and midmost images
        if save_images > 0 and self.imgs_dir is not None:
            self.save_informative_samples(
                selection_loader=selection_loader,
                batch_wise_indices=batch_wise_indices,
                proj_vals=diag['proj_vals'],
                reg_coeffs=reg,
                x=save_images,
                out_dir=self.imgs_dir / f"epoch{self.epoch}",  # pass epoch in somehow
                use_gamma_for_selected=True,
            )

        return chosen_subset_indices, sample_weight_map, diag

    def gradmatch_select_perclass(self, selection_loader, subset_ratio, positive=True, save_plots=False, save_images=0):
        """
        PerClass GradMatch selection: run OMP separately for each class with proportional budget.
        This ensures class-balanced subset selection.

        Returns:
            chosen_subset_indices: list[int] (indices w.r.t. selection_loader.dataset)
            sample_weight_map: dict[int -> float] (base_ds_idx -> gamma)
            diag: dict
        """
        assert 0 < subset_ratio <= 1.0

        sel_subset = selection_loader.dataset
        base_indices = sel_subset.indices  # maps subset_idx -> base_ds_idx
        N_total = len(sel_subset)

        # extract all labels (this iterates through dataset once)
        print("GradMatchPC: extracting labels...")
        all_labels = self._extract_labels(selection_loader)
        num_classes = int(all_labels.max().item()) + 1

        # total budget
        total_budget = max(1, int(math.ceil(subset_ratio * N_total)))

        # collect results across all classes
        all_chosen_subset_indices = []
        all_sample_weight_map = {}
        per_class_diag = {}

        for class_idx in tqdm(range(num_classes), desc="GradMatchPC: processing classes"):
            # find samples belonging to this class
            class_mask = (all_labels == class_idx)
            class_subset_idxs = torch.where(class_mask)[0].tolist()  # subset indices for this class
            n_class = len(class_subset_idxs)

            if n_class == 0:
                continue

            # proportional budget for this class
            class_budget = max(1, int(math.ceil(total_budget * n_class / N_total)))

            # create a loader for this class's samples (batch_size=1 for per-sample gradients)
            class_data = Subset(sel_subset, class_subset_idxs)
            class_loader = DataLoader(
                class_data,
                batch_size=1,  # per-sample gradients
                shuffle=False,
                num_workers=0,  # avoid multiprocessing overhead for small subsets
                pin_memory=True,
                collate_fn=collate_batch,
            )

            # compute per-sample gradients for this class
            grads = self._compute_gradients_for_loader(class_loader, desc_prefix=f"Class {class_idx}: ")
            A = grads.transpose(0, 1).contiguous()  # (d, n_class)
            b = grads.sum(dim=0).contiguous()  # (d,)

            # run OMP for this class
            with vram_ctx(f'DR-omp-class{class_idx}', epoch=self.epoch):
                class_chosen, class_weights, reg = self._run_omp_and_collect(
                    A, b, class_budget, positive,
                    base_indices, class_subset_idxs  # map local -> subset -> base
                )

            all_chosen_subset_indices.extend(class_chosen)
            all_sample_weight_map.update(class_weights)

            per_class_diag[f"class_{class_idx}"] = {
                "n_samples": n_class,
                "budget": class_budget,
                "n_selected": len(class_chosen),
            }

        # fill remaining budget with random samples if needed (due to rounding)
        diff = total_budget - len(all_chosen_subset_indices)
        if diff > 0:
            remaining = set(range(N_total)) - set(all_chosen_subset_indices)
            extra = np.random.choice(list(remaining), size=min(diff, len(remaining)), replace=False)
            for sub_i in extra:
                all_chosen_subset_indices.append(int(sub_i))
                base_i = int(base_indices[int(sub_i)])
                all_sample_weight_map[base_i] = 1.0

        # shuffle to mix classes during training
        rand_perm = np.random.permutation(len(all_chosen_subset_indices))
        all_chosen_subset_indices = [all_chosen_subset_indices[i] for i in rand_perm]

        diag = {
            "method": "gradmatch_perclass",
            "n_total": N_total,
            "total_budget": total_budget,
            "n_points_selected": len(all_chosen_subset_indices),
            "subset_ratio": subset_ratio,
            "num_classes": num_classes,
            "per_class": per_class_diag,
            "random_fill": max(0, diff),
        }

        return all_chosen_subset_indices, all_sample_weight_map, diag

    def random_select_perclass(self, selection_loader, subset_ratio, positive=True, save_plots=False, save_images=0):
        """
        PerClass random selection baseline: randomly select proportionally from each class.

        Returns:
            chosen_subset_indices: list[int] (indices w.r.t. selection_loader.dataset)
            sample_weight_map: dict[int -> float] (base_ds_idx -> weight=1.0)
            diag: dict
        """
        assert 0 < subset_ratio <= 1.0

        sel_subset = selection_loader.dataset
        base_indices = sel_subset.indices
        N_total = len(sel_subset)

        # extract all labels
        print("RandomPC: extracting labels...")
        all_labels = self._extract_labels(selection_loader)
        num_classes = int(all_labels.max().item()) + 1

        total_budget = max(1, int(math.ceil(subset_ratio * N_total)))

        all_chosen_subset_indices = []
        all_sample_weight_map = {}
        per_class_diag = {}
        rng = np.random.default_rng()

        for class_idx in tqdm(range(num_classes), desc="RandomPC: processing classes"):
            class_mask = (all_labels == class_idx)
            class_subset_idxs = torch.where(class_mask)[0].tolist()
            n_class = len(class_subset_idxs)

            if n_class == 0:
                continue

            class_budget = max(1, int(math.ceil(total_budget * n_class / N_total)))
            # randomly select from this class
            selected_local = rng.choice(n_class, size=min(class_budget, n_class), replace=False)

            for lid in selected_local:
                sub_i = class_subset_idxs[lid]
                base_i = int(base_indices[sub_i])
                all_chosen_subset_indices.append(sub_i)
                all_sample_weight_map[base_i] = 1.0

            per_class_diag[f"class_{class_idx}"] = {
                "n_samples": n_class,
                "budget": class_budget,
                "n_selected": len(selected_local),
            }

        # shuffle
        rand_perm = np.random.permutation(len(all_chosen_subset_indices))
        all_chosen_subset_indices = [all_chosen_subset_indices[i] for i in rand_perm]

        diag = {
            "method": "random_perclass",
            "n_total": N_total,
            "total_budget": total_budget,
            "n_points_selected": len(all_chosen_subset_indices),
            "subset_ratio": subset_ratio,
            "num_classes": num_classes,
            "per_class": per_class_diag,
        }

        return all_chosen_subset_indices, all_sample_weight_map, diag

    def _get_subset_meta(self, selection_loader, subset_idx: int):
        """
        selection_loader.dataset is Subset(sel_cp, train_subset.indices)
        subset_idx indexes into that Subset.
        Returns meta dict (includes 'path') for the *base dataset index*.
        """
        sel_subset = selection_loader.dataset
        base_ds = sel_subset.dataset           # TreeDataset copy
        base_i = int(sel_subset.indices[int(subset_idx)])  # map subset_idx -> base_ds_idx
        meta = base_ds.meta[base_i]
        return meta, base_i

    def _pick_sample_subset_idxs_from_batch_ranking(self, batch_wise_indices, batch_rank_ids, x):
        """
        Given a ranking of batch ids, expand into sample subset indices until you have x samples.
        Returns list[int] of subset indices (indices into selection_loader.dataset).
        """
        picked = []
        for bid in batch_rank_ids:
            for sub_i in batch_wise_indices[int(bid)]:
                picked.append(int(sub_i))
                if len(picked) >= x:
                    return picked
        return picked

    def _save_samples(self, selection_loader, subset_idxs, out_dir, prefix, save_grid=True):
        """
        Saves original images and (optionally) a side-by-side grid:
        each row: [noT | trainT | valT]
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        triplets = []

        for rank, sub_i in enumerate(
            tqdm(subset_idxs, desc=f"Saving {prefix} samples", dynamic_ncols=True),
            start=1
        ):
            meta, base_i = self._get_subset_meta(selection_loader, sub_i)
            src = Path(meta["path"])
            ext = src.suffix.lower() if src.suffix else ".png"

            # --- save original (as before) ---
            dst = out_dir / f"{prefix}_rank{rank:04d}_subset{sub_i}_base{base_i}{ext}"
            img = Image.open(src).convert("RGB")
            img.save(dst)

            # --- collect triplet for grid ---
            if save_grid:
                noT_pil, trainT_pil, valT_pil = self._get_triplet_pils(selection_loader, sub_i)
                triplets.append((noT_pil, trainT_pil, valT_pil))

        # --- write grid once ---
        if save_grid and len(triplets) > 0:
            grid_fp = out_dir / f"{prefix}_grid_{len(triplets)}rows.png"
            self._save_triplet_grid(triplets, grid_fp)


    def save_informative_samples(
        self,
        selection_loader,
        batch_wise_indices,
        proj_vals, # torch.Tensor (n_batches,) or np array
        reg_coeffs, # torch.Tensor (n_batches,) from OMP
        x,
        out_dir,
        use_gamma_for_selected=True,
    ):
        """
        Save x most / middle / least informative samples according to OMP signals.

        - "informativeness" across *all* batches uses |proj| = |A^T b| (OMP first-iter signal).
        - Optionally, for "most selected" you can rank selected batches by |gamma| (OMP coefficients).

        Writes:
            out_dir/
              most_proj/
              mid_proj/
              least_proj/
              most_gamma_selected/  (optional)
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # ---- batch scoring ----
        proj = proj_vals.detach().float().abs().cpu().numpy() if torch.is_tensor(proj_vals) else np.abs(np.asarray(proj_vals, dtype=float))
        n_batches = len(proj)
        assert n_batches == len(batch_wise_indices), "proj_vals length must match number of batches."

        # rank batches by projection
        order = np.argsort(proj)[::-1]  # descending
        top_batch_ids = order
        bot_batch_ids = order[::-1]

        # "middlemost": take batches around the median projection rank
        mid_center = n_batches // 2
        # pick batches closest to median rank (expand outward)
        mid_batch_ids = []
        l = mid_center - 1
        r = mid_center
        while len(mid_batch_ids) < n_batches:
            if r < n_batches:
                mid_batch_ids.append(order[r])
                r += 1
            if l >= 0:
                mid_batch_ids.append(order[l])
                l -= 1

        # ---- expand batches -> sample subset indices ----
        most_subset_idxs  = self._pick_sample_subset_idxs_from_batch_ranking(batch_wise_indices, top_batch_ids, x)
        least_subset_idxs = self._pick_sample_subset_idxs_from_batch_ranking(batch_wise_indices, bot_batch_ids, x)
        mid_subset_idxs   = self._pick_sample_subset_idxs_from_batch_ranking(batch_wise_indices, mid_batch_ids, x)

        # ---- save ----
        self._save_samples(selection_loader, most_subset_idxs,  out_dir / "most_proj",  "most_proj")
        self._save_samples(selection_loader, mid_subset_idxs,   out_dir / "mid_proj",   "mid_proj")
        self._save_samples(selection_loader, least_subset_idxs, out_dir / "least_proj", "least_proj")

        # ---- optional: rank SELECTED batches by |gamma| ----
        if use_gamma_for_selected:
            reg = reg_coeffs.detach().float().cpu() if torch.is_tensor(reg_coeffs) else torch.tensor(reg_coeffs, dtype=torch.float32)
            selected_batch_ids = torch.nonzero(reg).view(-1).cpu().numpy()
            if len(selected_batch_ids) > 0:
                gam = reg.abs().cpu().numpy()
                sel_order = selected_batch_ids[np.argsort(gam[selected_batch_ids])[::-1]]  # descending |gamma|
                most_gamma_subset_idxs = self._pick_sample_subset_idxs_from_batch_ranking(batch_wise_indices, sel_order, x)
                self._save_samples(selection_loader, most_gamma_subset_idxs, out_dir / "most_gamma_selected", "most_gamma")

    def _get_triplet_pils(self, selection_loader, subset_idx: int):
        """
        Returns (noT_pil, trainT_pil, valT_pil) for a given subset_idx.
        - noT: static only (PIL)
        - trainT: static + train transforms (tensor -> PIL)
        - valT: static + val transforms (tensor -> PIL)
        """
        sel_subset = selection_loader.dataset
        base_ds = sel_subset.dataset  # sel_cp (TreeDataset copy)
        base_i = int(sel_subset.indices[int(subset_idx)])
        meta = base_ds.meta[base_i]
        src = Path(meta["path"])

        # load raw
        img = Image.open(src).convert("RGB")

        # static only
        static_T = getattr(base_ds, "static_transform", None)
        img_static = static_T(img) if static_T is not None else img
        noT_pil = img_static if isinstance(img_static, Image.Image) else to_pil_image(img_static)

        # IMPORTANT:
        # - val_T should be the val transform used in selection_loader (base_ds.random_transform)
        # - train_T should be the *training* transform used in your real train loader
        val_T = getattr(base_ds, "random_transform", None)

        if not hasattr(self, "train_transform") or self.train_transform is None:
            raise RuntimeError(
                "GradMatch selector is missing self.train_transform. "
                "Set it at init so we can render trainT images."
            )

        train_T = self.train_transform

        img_train = train_T(img_static) if train_T is not None else img_static
        img_val   = val_T(img_static)   if val_T is not None else img_static

        if isinstance(img_train, torch.Tensor):
            img_train = to_pil_image(unnormalize(img_train, self.backbone_data_mean, self.backbone_data_std))
        if isinstance(img_val, torch.Tensor):
            img_val = to_pil_image(unnormalize(img_val, self.backbone_data_mean, self.backbone_data_std))

        return noT_pil, img_train, img_val


    def _save_triplet_grid(self, triplets, out_path, col_titles=("noT", "trainT", "valT"), pad=6):
        """
        triplets: list of (noT_pil, trainT_pil, valT_pil)
        Saves one PNG grid: rows=len(triplets), cols=3
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if len(triplets) == 0:
            return

        # standardize cell size from first image
        w, h = triplets[0][0].size
        cols = 3
        rows = len(triplets)

        title_h = 0  # keep simple; no text overlay unless you want it
        grid_w = cols * w + (cols + 1) * pad
        grid_h = rows * h + (rows + 1) * pad + title_h

        grid = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))

        y = pad + title_h
        for r, (a, b, c) in enumerate(triplets):
            x = pad
            for im in (a, b, c):
                if im.size != (w, h):
                    im = im.resize((w, h))
                grid.paste(im, (x, y))
                x += w + pad
            y += h + pad

        grid.save(out_path)
