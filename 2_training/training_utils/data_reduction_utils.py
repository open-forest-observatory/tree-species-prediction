import torch
from torch.utils.data import DataLoader, Subset
import gc
import time
from itertools import product
import math
from tqdm import tqdm
import copy

from models.TreeSpeciesClassifier import TreeSpeciesClassifierFromPretrained
from training_utils.omp import omp_select
from training_utils.data_utils import assemble_dataloaders
from training_utils.metrics import compute_pairwise_jaccard_similarity
from configs.data_reduction_config import dr_config
from configs.model_config import model_config

def generate_params(**params):
    """
    Yields dictionaries of parameter combinations dynamically
    Example usage:
    for combo in generate_params(reduction_ratio=[0.25, 0.5], warm_start_epoch=[100, 200]):
        ... do stuf with combo ...
    """
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
    batch_size
):
    """
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

    subbatch_grads = [] # store gradients calculated for each subbatch
    subbatch_idxs = [] # store sample idxs of samples in subbatches relative to whole dset

    # idxs of all samples in dataset
    # idx tracking of samples is needed so we can subset samples in dataset correctly,
    # when using subbatch for gradient approximations
    all_idxs = list(range(len(dataset)))
    N = len(all_idxs)

    # iterate over idxs of batches
    # batch_idxs is a list of sample idxs of len batch_size
    for batch_idxs in tqdm(chunk_list(all_idxs, batch_size), total=math.ceil(N / batch_size), desc="Gathering gradients for batches"):
        
        # Now split batch_idxs into smaller slices of size minibatch_size
        for cur_subbatch_idxs in chunk_list(batch_idxs, subbatch_size):
            
            # get all samples of this current subbatch
            cur_subbatch_imgs_list, cur_subbatch_labels_list = [], []
            for i in cur_subbatch_idxs:
                cur_subbatch_imgs_list.append(dataset[i][0])
                cur_subbatch_labels_list.append(dataset[i][1])

            cur_subbatch_imgs = torch.stack(cur_subbatch_imgs_list, dim=0).to(device, non_blocking=True)
            cur_subbatch_labels = torch.tensor(cur_subbatch_labels_list, device=device)

            # Zero old grads
            model_copy.zero_grad(set_to_none=True)
            
            # shape: (subbatch_size, num_classes); or  (subbatch_size % num_subbatches, num_classes) if last subbatch
            logits = model_copy(cur_subbatch_imgs) # Forward pass for this sub-batch

            # WGAN real term = -mean(D(real)); not using mone as done in wgan train loop
            loss = criterion(logits, cur_subbatch_labels)
            loss.backward() # backprop to compute gradients

            # Flatten param grads
            param_keep_type = 'classifier' if not dr_config.use_backbone_gradients else None
            grads_flat = _flatten_param_grads(model_copy, param_keep_type=param_keep_type)

            subbatch_grads.append(grads_flat.detach().cpu())
            subbatch_idxs.append(cur_subbatch_idxs)

    # list -> tensor
    grads_matrix = torch.stack(subbatch_grads, dim=0) # shape: (N, param_dim)
    # sum grads across all samples
    grad_sum = grads_matrix.sum(dim=0) # shape: (param_dim,)

    # delete model copy
    del model_copy
    gc.collect()
    torch.cuda.empty_cache()
    return grads_matrix, grad_sum, subbatch_idxs

def _is_subset_epoch(epoch):
    return (epoch <= model_config.epochs and # if we haven't trained for the full spec'd num epochs and,
            epoch >= dr_config.num_warm_start_epochs and # if we have trained enough on the full data and,
            epoch % dr_config.epoch_selection_interval == 0) # we are at a selection interval epoch,  

def gradsel_reduce(
        model,
        base_dataset,
        criterion,
        epoch_idx,
        train_transform,
        val_transform,
        reduction_ratio,
        subbatch_size,
        device,
        prev_subset_idxs=None # for jaccard sim comparison
    ): 
    # ensure subbatch budget won't be 0, 
    # i.e. |_(reduction ratio * num_batches)_| > 0
    assert int(reduction_ratio * len(base_dataset)) >= 1, f"ERROR: Reduction ratio too small for dataset size and/or batch_size!"
    print(f"Will select subsets every {dr_config.epoch_selection_interval} epochs, warm start: {dr_config.num_warm_start_epochs}.")

    # TODO: integrate performance metrics
    metric = -1
        
    # update subset selections
    print("*** Performing data reduction ***")   
    t0_subset = time.time()  
    
    # gather gradients of Discriminator on real data (since that's what we are subsetting)
    grad_mat, grad_sum, subbatch_idxs = gather_gradients(
        model,
        base_dataset,
        criterion,
        device,
        subbatch_size,
        model_config.batch_size
    )
    
    # determine budget based on reduction ratio and subbatch size'
    # this is how many subbatches worth of samples we can take
    budget = int(reduction_ratio * grad_mat.shape[0])
    print(budget, grad_mat.size())
    print("Solving OMP system... (choosing best gradient approximating subsets)")
    
    # run Orthogonal Matching Pursuit to determine best idxs using gradient matrix and gradient sum
    new_subbatch_idxs = omp_select(grad_mat, grad_sum, budget, device)
    print(f"Chose: {len(new_subbatch_idxs)} subbatches ({len(new_subbatch_idxs) * subbatch_size}) out of {len(base_dataset)}")

    # get actual sample idxs from subbatch idxs
    new_idxs = []
    for subbatch_i in new_subbatch_idxs:
        # extend with sample idxs from idxs of chosen subsets
        new_idxs.extend(subbatch_idxs[subbatch_i])

    # Rebuild the subset/dataloader with the new indices
    current_subset = Subset(base_dataset, new_idxs)

    subset_sel_time = time.time() - t0_subset # track time for selecting subset of data
    print(f"New subset has {len(current_subset)} items.\nSubset selection took {subset_sel_time:.4f} seconds to compute")

    #TODO: Save computed subset idxs

    del grad_mat, grad_sum, subbatch_idxs
    gc.collect()
    torch.cuda.empty_cache()

    train_loader, val_loader = assemble_dataloaders(base_dataset, train_transform, val_transform, return_idxs=False, idxs_pool=current_subset.indices)

    # jaccard similarity compares how different set A (previously chosen subset) is from set B (newly chosen subset)
    if prev_subset_idxs is not None:
        jaccard_sim = compute_pairwise_jaccard_similarity(prev_subset_idxs, current_subset.indices)
        print(f"New subset is {jaccard_sim*100:.4f}% similar to previous subset")


    return train_loader, val_loader, chosen_idxs_pool

    return current_subset
