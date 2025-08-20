import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from PIL import Image
from pathlib import Path
import re
from typing import List, Dict, Optional, Any
import geopandas as gpd

class TreeDataset(Dataset):
    """
    X[i] : image tensor
    y[i] : species index (int)
    meta[i] : dict(tree_id, view, dset, file_idx, species_str, path, gpkg_row_idx)

    ex meta[i]: {
        'treeID': '00776',
        'species': 'ABCO',
        'dset': '0001_001435_001436',
        'view': 'oblique',
        'file_idx': '25',
        'path': PosixPath('/ofo-share/species-prediction-project/intermediate/cropped_trees/labelled/0001_001435_001436/treeID00776/treeID00776-speciesABCO-dset0001_001435_001436-viewoblique-25.png'),
        'label_idx': 0,
        'gpkg_row_idx': 776
    }
    """

    _PATTERN = re.compile( # regex pattern to grab relevant info from image filenames
        r"""
        treeID(?P<treeID>\d+)-              # treeID00613
        species(?P<species>[A-Za-z0-9]+)-   # speciesABCO/speciesCADE27
        dset(?P<dset>[\d_]+)-               # dset0001_001435_001436
        view(?P<view>[A-Za-z]+)-            # viewnadir
        (?P<file_idx>\d+)                   # 1.png -> file_idx = 1
        \.png$
        """,
        re.VERBOSE,
    ) # sample tree img file name: treeID00613-speciesABCO-dset0001_001435_001436-viewnadir-12

    def __init__(
        self,
        imgs_root: str | Path | List[str | Path],       # path(s) to root folder to grab images
        transform: Optional[torch.nn.Module] = None,    # torch transformations to apply
        img_exts: List[str] = ['.png'],                 # img exts to load
        gpkg_dir: Optional[str | Path] = None           # path to gpkg files, if None will not have row idxs of trees
    ):
        self.imgs_root = imgs_root
        self.img_exts = {ext.lower() for ext in img_exts}
        self.tree_id_col_name = 'unique_ID'
        
        # recursively get img paths from root dir
        self.img_paths = self.get_img_paths_from_root()
        self.parse_file_names() # assemble meta data dict from filename
        if not self.img_paths:
            raise RuntimeError("No image files found.")
        
        # map string species labels to int idxs, and have a reverse map to get labels back from idxs
        species_labels = sorted({m['species'] for m in self.meta})
        self.label2idx_map = {sp: i for i, sp in enumerate(species_labels)}
        self.idx2label_map = {i: sp for sp, i in self.label2idx_map.items()}

        for m in self.meta:
            m['label_idx'] = self.label2idx_map[m['species']]
            m['unique_treeID'] = f"{m['dset']}-{m['treeID']}"

        # get the row idx of each tree in the gpkg file
        if gpkg_dir is not None:
            self.gpkg_dir = Path(gpkg_dir)
            self.gpkg_LUT = self._build_gpkg_lookup_tables() # dset_path : {tree_id (str): gpkg_row_idx}
            #print(self.gpkg_LUT)
            for meta_dict in self.meta:
                tree_id_str = str(meta_dict['treeID'])
                gpkg_fp = gpkg_dir / f"{meta_dict['dset']}.gpkg"
                if not gpkg_fp.exists():
                    raise FileNotFoundError(f"Could not find: {gpkg_fp}")
                
                # TODO: Sometimes can't find the row index, says key error with some tree_id_strs
                try:
                    meta_dict['gpkg_row_idx'] = self.gpkg_LUT[meta_dict['dset']][tree_id_str]
                except:
                    pass
                    #print(meta_dict['dset'], tree_id_str)


        

        self.transform = (
            transform if transform is not None
            else T.Compose([T.ToTensor()])  # (0-255) -> (0-1) float tensor
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        meta = self.meta[idx]
        img = Image.open(meta["path"]).convert("RGB")
        img = self.transform(img)
        label_idx = int(meta["label_idx"])

        return img, label_idx, meta
    
    def get_img_paths_from_root(self):
        img_paths = []
        for path in self.imgs_root.rglob('*'):
            if path.suffix.lower() in self.img_exts:
                img_paths.append(path)
        
        return img_paths

    def parse_file_names(self):
        self.meta: List[Dict[str, Any]] = []
        species_strings = []
        for p in self.img_paths:
            m = self._PATTERN.search(p.name)
            if not m:
                raise ValueError(f"Filename does not match pattern: {p.name}")
            
            info = m.groupdict() # dict of matched subgroups from regex pattern
            info["path"] = p
            species_strings.append(info["species"])
            self.meta.append(info)

    def _build_gpkg_lookup_tables(self) -> dict[str, dict[str, int]]:
        """
        For each dset_id, open <dset_id>.gpkg (single layer assumed) and build a map:
            { tree_id (as str) -> pandas row index (int) }.
        Returns a dict keyed by dset_id so we can look up quickly later.
        """
        unique_dset_ids = set(m['dset'] for m in self.meta)
        dset_to_tree_map: dict[str, dict[str, int]] = {}

        for dset_id in unique_dset_ids:
            gpkg_path = self.gpkg_dir / f"{dset_id}.gpkg"
            if not gpkg_path.exists():
                raise FileNotFoundError(f"GPKG for dset '{dset_id}' not found: {gpkg_path}")

            gdf = gpd.read_file(gpkg_path)  # single layer assumed
            if self.tree_id_col_name not in gdf.columns:
                raise ValueError(f"Column '{self.tree_id_col_name}' not present in {gpkg_path}")

            # build mapping from ID column to the gdf's row index.
            # String keys avoid type mismatches ('42' vs 42).
            id_series = gdf[self.tree_id_col_name]
            tree_map = {id_str: int(idx) for idx, id_str in zip(gdf.index, id_series)}
            dset_to_tree_map[dset_id] = tree_map

        return dset_to_tree_map
    
def collate_batch(batch):
    imgs, labels, metas = zip(*batch)             # tuples of length B
    imgs   = torch.stack(imgs, dim=0)             # [B, C, H, W]
    labels = torch.as_tensor(labels, dtype=torch.long)  # [B]
    return imgs, labels, list(metas)   