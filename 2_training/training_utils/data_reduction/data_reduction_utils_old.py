import torch
from torch.utils.data import DataLoader, Subset
import gc
import time
from itertools import product
import math
import numpy as np
from tqdm import tqdm
import copy

from models.TreeSpeciesClassifier import TreeSpeciesClassifierFromPretrained
from data.dataset import collate_batch, ListBatchSampler
from training_utils.omp import omp_select
from training_utils.data_utils import assemble_dataloaders
from training_utils.metrics import compute_pairwise_jaccard_similarity
from training_utils.visualization import plot_projection_sorted, plot_projection_scatter, plot_singleton_quality
from configs.data_reduction_config import dr_config
from configs.model_config import model_config

class GradBuffer:
    '''
    pool gradients of training epoch to use for data reduction.
    This avoids needing an additional forward pass on all data as we gather during training
    '''
    def __init__(self):
        self.grads = [] # list of (param_dim,) tensors
        self.batch_sample_idxs = [] # list of lists containing sample ids per batch

    def add(self, grad_vec, batch_ids):
        self.grads.append(grad_vec.detach().cpu())
        self.batch_sample_idxs.append(list(map(int, batch_ids)))

    def assemble_grad_mat_and_sum(self, row_norm=False):
        grads = torch.stack(self.grads, dim=0) # (n_batches, param_dim)
        if row_norm:
            grads = torch.nn.functional.normalize(grads, dim=1, eps=1e-12)
        grads_sum = grads.sum(dim=0) # (param_dim,)
        return grads, grads_sum, self.batch_sample_idxs


def generate_params(**params):
    """
    Yields dictionaries of parameter combinations dynamically
    Example usage:
    for combo in generate_params(reduction_ratio=[0.25, 0.5], warm_start_epoch=[100, 200]):
        ... do stuf with combo ...
    """
    # TODO: Switch to sklearn.paramgrid
    keys, values = zip(*params.items()) if params else ([], [])
    for combo in product(*values):
        yield dict(zip(keys, combo))

def _flatten_param_grads(model, param_keep_type=None):
    """
    Flattens gradients from model.parameters(), optionally keeping only
    parameters whose names match `name_allow_regex` (e.g., r'^(head|classifier|fc)\.').
    """
    keep = []
    for name, p in model.named_parameters():
        if p.grad is None: 
            continue # only care about params that contribute to the gradient
            
        # append param grads if it is one of the specified keep types, or if no keep type spec'd
        if (param_keep_type is not None and param_keep_type in name) or param_keep_type is None: 
            keep.append(p.grad.reshape(-1))

    return torch.cat(keep, dim=0)

def save_selection(out_fp, idxs):
    out_dir = out_fp.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_fp, np.array(idxs, dtype=np.int32))

def load_selection(out_dir, epoch):
    fp = Path(out_dir) / 'subsets' / f"selection_epoch{epoch}.npy"
    return np.load(fp)

# Function to chunk a list into pieces of size chunk_size
def chunk_list(lst, chunk_size):
    for start in range(0, len(lst), chunk_size):
        yield lst[start : start + chunk_size]

def gather_gradients(
    model, 
    dataset, 
    criterion,
    device,
    subbatch_size,
    batch_size,
    preshuffled_idxs=None   # local idxs to subset; if idxs shuffled once before hand, 
                            # more likely to converge to consistent subset choices
):
    """
    DEPRECATED FOR NOW, LEAVING HERE UNTIL CONFIRMED NO LONGER NECESSARY

    3 deep for loops looks bad I know, but I swear it's still just O(n),
    there's just a lot of chunks since we have a batch size for model training,
    which is different than the subbatch size we use for a better gradient approximation.
    
    The alternatives are find gradients PER SAMPLE (essentially subbatch_size of 1) which would take longer than a full model training,
    Or choose best full batches to approximate the gradient, which is hugely lacking precision.
    And given our batch size it would mean there's only 10 choices, which means we couldn't subset < 10% of the original dataset

    Gathers per-sample gradient of the model.
    
    Returns:
        grads_matrix: (N, param_dim)
        grad_sum: (param_dim,)
    """    
    # copy model so gradient gathering doesn't effect model's state dict
    model_copy = copy.deepcopy(model).to(device)
    model_copy.eval() # eval mode for consistency

    # figure out if we’re looking at a Subset view
    is_subset = isinstance(dataset, Subset)
    base_ds = dataset.dataset if is_subset else dataset
    idx_list = dataset.indices if is_subset else list(range(len(dataset)))  # GLOBAL ids if Subset

    subbatch_grads = [] # store gradients calculated for each subbatch
    subbatch_global_idxs = [] # store sample idxs of samples in subbatches relative to whole dset

    # idxs of all samples in dataset
    # idx tracking of samples is needed so we can subset samples in dataset correctly,
    # when using subbatch for gradient approximations
    # note: global -> idxs of the whole dataset; local -> idxs of the subset being selected from (training dset since val dset unaffected)
    if preshuffled_idxs is None:
        all_local_idxs = list(range(len(dataset)))
    else:
        all_local_idxs = list(map(int, preshuffled_idxs))
    N = len(all_local_idxs)

    # iterate over idxs of batches
    # batch_idxs is a list of sample idxs of len batch_size
    for batch_local_idxs in tqdm(chunk_list(all_local_idxs, batch_size), total=math.ceil(N / batch_size), desc="Gathering gradients for batches"):
        
        # Now split batch_idxs into smaller slices of size minibatch_size
        for cur_subbatch_local_idxs in chunk_list(batch_local_idxs, subbatch_size):
            # get all samples of this current subbatch
            imgs, labels, cur_sub_global = [], [], []
            
            for loc_i in cur_subbatch_local_idxs:
                # map local to global idx
                glob_i = idx_list[loc_i] if is_subset else loc_i
                cur_sub_global.append(glob_i)

                # fetch from dset so transforms applied (val view tfs)
                cur_img, cur_lbl = dataset[loc_i][0], dataset[loc_i][1]
                imgs.append(cur_img)
                labels.append(cur_lbl)

            X = torch.stack(imgs, dim=0).to(device, non_blocking=True)
            y = torch.tensor(labels, device=device, dtype=torch.long)

            # Zero old grads
            model_copy.zero_grad(set_to_none=True)
            
            # shape: (subbatch_size, num_classes); or  (subbatch_size % num_subbatches, num_classes) if last subbatch
            logits = model_copy(X) # Forward pass for this sub-batch

            loss = criterion(logits, y)
            loss.backward() # backprop to compute gradients

            # Flatten param grads
            param_keep_type = 'classifier' if not dr_config.use_backbone_gradients else None
            grads_flat = _flatten_param_grads(model_copy, param_keep_type=param_keep_type)

            subbatch_grads.append(grads_flat.detach().cpu())
            subbatch_global_idxs.append(cur_sub_global)

    # list -> tensor
    grads_matrix = torch.stack(subbatch_grads, dim=0) # shape: (N, param_dim)
    # sum grads across all samples
    grad_sum = grads_matrix.sum(dim=0) # shape: (param_dim,)

    # delete model copy
    del model_copy
    gc.collect()
    torch.cuda.empty_cache()

    return grads_matrix, grad_sum, subbatch_global_idxs

def _is_subset_epoch(epoch):
    return (epoch <= model_config.epochs and # if we haven't trained for the full spec'd num epochs and,
            epoch >= dr_config.num_warm_start_epochs and # if we have trained enough on the full data and,
            epoch % dr_config.epoch_selection_interval == 0) # we are at a selection interval epoch,  

def gradsel_reduce(
        model,
        grad_buffer,
        train_dset,             # same samples as in `dset_vew` but with training transforms (current training split)
        static_transform,
        train_transform,
        reduction_ratio,
        device,
        lam=0.5,
        subsets_save_fp=None,
        plots_save_fp=None,
        prev_sample_idxs=None,   # for jaccard sim comparison
        track_diagnostics=False
    ): 
    grad_mat, grad_sum, batch_ids_list = grad_buffer.assemble_grad_mat_and_sum(row_norm=True) # grad_mat -> (n_batches, param_dim)
    budget = int(reduction_ratio * grad_mat.shape[0])
    print(f"Will select {budget} batches; Gradient Matrix dim = {grad_mat.shape}")
        
    # update subset selections
    print("*** Performing data reduction ***")   
    t0_subset = time.time()  
    
    # call OMP to choose the optimal batches of images
    diagnostics = None
    if track_diagnostics:
        sel_rows, sel_coeffs, diagnostics = omp_select(grad_mat, grad_sum, budget, device=device, lam=lam, positive=True, track_diagnostics=track_diagnostics)
        
        # visuals for omp
        if plots_save_fp is not None:
            plot_projection_sorted(omp_diagnostics['proj_vals'], plots_save_fp, epoch=epoch, title_prefix="OMP")
            plot_projection_scatter(omp_diagnostics['proj_vals'], plots_save_fp, epoch=epoch, title_prefix="OMP")
            plot_singleton_quality(
                omp_diagnostics['topk_cos'], omp_diagnostics['topk_rr'], 
                omp_diagnostics['botk_cos'], omp_diagnostics['botk_rr'],
                plots_save_fp, epoch=epoch
            )
    else:
        sel_rows, sel_coeffs = omp_select(grad_mat, grad_sum, budget, device=device, lam=lam, positive=True, track_diagnostics=track_diagnostics)
    subset_sel_time = time.time() - t0_subset # track time for selecting subset of data
    
    # flatten selected batches to get just sample ids; maintain batch contents from selected batches
    selected_batches = [batch_ids_list[r] for r in sel_rows] # List[List[int]]
    batch_weights = sel_coeffs.tolist() # List[float]
    sel_sample_ids = [sid for batch in selected_batches for sid in batch] # flatten sample ids for logging purposes only
    
    # init new subset and dataloader with chosen idxs
    train_base = train_dset.dataset
    train_base.static_transform = static_transform
    train_base.random_transform = train_transform

    batch_sampler = ListBatchSampler(selected_batches)
    new_train_loader = DataLoader(
        train_base,
        batch_sampler=batch_sampler,
        #shuffle=True,
        num_workers=model_config.num_workers,
        pin_memory=True,
        collate_fn=collate_batch
    )
    print(f"New subset has {len(sel_sample_ids)} items.\nSubset selection took {subset_sel_time:.4f} seconds to compute")

    # save computed subset idxs
    if subsets_save_fp is not None:
        save_selection(subsets_save_fp, sel_sample_ids)

    del grad_mat, grad_sum
    gc.collect()
    torch.cuda.empty_cache()

    # jaccard similarity compares how different set A (previously chosen subset) is from set B (newly chosen subset)
    if prev_sample_idxs is not None:
        jaccard_sim = compute_pairwise_jaccard_similarity(prev_sample_idxs, sel_sample_ids)
        print(f"New subset is {jaccard_sim*100:.4f}% similar to previous subset")


    return new_train_loader, sel_sample_ids, batch_weights, diagnostics