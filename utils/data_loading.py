import logging
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import nibabel

# -----------------------------------------------------
# Chargement d'une image NIfTI en PIL 2D
# -----------------------------------------------------
def load_image(filename, slice_idx=None):
    """
    Charge un fichier .nii.gz avec nibabel et renvoie une image PIL 2D.
    """
    nii = nibabel.load(str(filename))
    data = nii.get_fdata()

    # Normalisation 0-1
    data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

    # Si volume 3D, prendre coupe centrale
    if data.ndim == 3:
        if slice_idx is None:
            slice_idx = data.shape[2] // 2
        data = data[:, :, slice_idx]

    img = Image.fromarray((data * 255).astype(np.uint8))
    return img

# -----------------------------------------------------
# Dataset PyTorch filtré et compatible CrossEntropy
# -----------------------------------------------------
class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, valid_ids: list, scale: float = 1.0, mask_values: list = None):
        """
        images_dir : dossier des images NIfTI (source)
        masks_dir  : dossier des masques NIfTI
        valid_ids  : liste des indices autorisés (train/test)
        scale      : facteur de redimensionnement
        mask_values: liste des valeurs uniques présentes dans les masques
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        # Liste des fichiers filtrée par valid_ids
        all_files = [f.name.replace('.nii.gz', '') for f in self.images_dir.glob('*.nii.gz')]
        self.ids = [f for f in all_files if int(f.split('-')[0]) in valid_ids]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir} matching valid_ids')

        logging.info(f'Creating dataset with {len(self.ids)} examples')

        # Valeurs uniques des masques (pour mapping multi-classes)
        if mask_values is None:
            # Détecter automatiquement les valeurs uniques
            mask_values_set = set()
            for idx in self.ids:
                mask_name = idx.replace('-src', '-mask')
                mask_path = self.masks_dir / f'{mask_name}.nii.gz'
                mask = load_image(mask_path)
                mask_np = np.array(mask)
                mask_values_set.update(np.unique(mask_np).tolist())
            self.mask_values = sorted(list(mask_values_set))
        else:
            self.mask_values = mask_values

        # Mapping {valeur réelle -> indice}
        self.mask_mapping = {v: i for i, v in enumerate(self.mask_values)}
        logging.info(f'Mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess_image(pil_img, scale):
        """Redimensionne et normalise l'image"""
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale too small'
        pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)
        img = np.asarray(pil_img, dtype=np.float32) / 255.0
        img = img[np.newaxis, ...]  # (1, H, W)
        return img

    def preprocess_mask(self, pil_mask):
        """Redimensionne le masque et mappe les valeurs réelles en indices [0..C-1]"""
        w, h = pil_mask.size
        newW, newH = int(self.scale * w), int(self.scale * h)
        assert newW > 0 and newH > 0, 'Scale too small'
        pil_mask = pil_mask.resize((newW, newH), resample=Image.NEAREST)
        mask_np = np.array(pil_mask, dtype=np.int64)

        # Mapping vers 0..C-1
        target = np.zeros_like(mask_np, dtype=np.int64)
        for v, i in self.mask_mapping.items():
            target[mask_np == v] = i
        return target

    def __getitem__(self, idx):
        name = self.ids[idx]
        img_path = self.images_dir / f'{name}.nii.gz'
        mask_name = name.replace('-src', '-mask')
        mask_path = self.masks_dir / f'{mask_name}.nii.gz'

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        img = load_image(img_path)
        mask = load_image(mask_path)

        assert img.size == mask.size, f'Image and mask {name} must have the same size'

        img = self.preprocess_image(img, self.scale)
        mask = self.preprocess_mask(mask)

        return {
            'image': torch.as_tensor(img.copy(), dtype=torch.float32).contiguous(),
            'mask': torch.as_tensor(mask.copy(), dtype=torch.long).contiguous()
        }
