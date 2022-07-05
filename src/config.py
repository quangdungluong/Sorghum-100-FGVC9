"""
Configuration
"""
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CFG:
    """Config
    """
    seed = 108
    backbone = 'resnet50_ibn_a'
    batch_size = 16
    img_size = 1024
    device = torch.device("cpu")

    """ArcFace parameter"""
    num_classes = 100
    embedding_size = 1024
    S, M = 30.0, 0.3  # S:cosine scale in arcloss. M:arg penalty
    EASY_MERGING, LS_EPS = False, 0.0

    """Model parameter"""
    model_path = "models/arcface_resnet50_ibna_1024x1024_fold0.pth"
    input_csv = "data/train_cultivar_mapping.csv"

    """Data transform"""
    data_transforms = A.Compose([
    A.CLAHE(clip_limit=40, tile_grid_size=(10, 10), p=1.0),
    A.Resize(img_size, img_size, p=1.0),
    A.Normalize(p=1.0),
    ToTensorV2(p=1.0),
    ])
