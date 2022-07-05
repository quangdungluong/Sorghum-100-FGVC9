"""Sorghum model
"""
import torch
from torch import nn

from src.config import CFG
from src.resnet_ibn import resnet50_ibn_a
from src.utils import ArcMarginProduct, se_block


class SorghumModel(nn.Module):
    """Sorghum Model
    """
    def __init__(self, model_name, embedding_size, num_classes):
        super().__init__()
        self.model_name = model_name
        self.model = resnet50_ibn_a(pretrained=True)
        in_features = 2048
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.enhance = se_block(channel=in_features, ratio=8)
        self.multiple_dropout = [nn.Dropout(0.25) for i in range(5)]
        self.embedding = nn.Linear(in_features, embedding_size)
        # bnnneck
        self.bottleneck = nn.BatchNorm1d(embedding_size)
        self.bottleneck.bias.requires_grad_(False)
        self.pr = nn.PReLU()
        self.fc = ArcMarginProduct(embedding_size, num_classes,
                                    CFG.S, CFG.M, CFG.EASY_MERGING, CFG.LS_EPS)

    def forward(self, images, labels):
        """Forward"""
        features = self.model(images)
        features = self.enhance(features)
        pooled_features = self.pooling(features).flatten(1)
        pooled_features_dropout = torch.zeros((pooled_features.shape)).to(CFG.device)
        for i in range(5):
            pooled_features_dropout += self.multiple_dropout[i](pooled_features)
        pooled_features_dropout /= 5
        embedding = self.embedding(pooled_features_dropout)
        embedding = self.bottleneck(embedding)
        embedding = self.pr(embedding)
        output = self.fc(embedding, labels)
        return output

    def extract(self, images):
        """Extract embedding"""
        features = self.model(images)
        features = self.enhance(features)
        pooled_features = self.pooling(features).flatten(1)
        embedding = self.embedding(pooled_features)
        embedding = self.bottleneck(embedding)
        embedding = self.pr(embedding)
        return embedding
