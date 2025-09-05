import torch
import torch.nn as nn
import torchvision.models as models

# Enhanced Vision Transformer specific for histopathology with 256x256 inputs
class HistoViT(nn.Module):
 
    def __init__(self, num_classes=9, pretrained=False):
        super().__init__()
        
        # Use DeiT (data-efficient transformer) as base
        # DeiT was trained with more augmentations, making it better for medical images
        self.vit = timm.create_model(
            'deit_base_patch16_224',  # DeiT base model
            pretrained=pretrained,
            img_size=256,  # Specify 256x256 input
            drop_path_rate=0.1  # Add regularization
        )
        
        # Get the hidden dimension
        hidden_size = self.vit.head.in_features
        self.vit.head = nn.Identity()
        
        # Add a specialized head for histopathology
        # This architecture is designed for tissue classification
        self.histo_head = nn.Sequential(
            nn.Linear(hidden_size, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(768, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(384, num_classes)
        )
    
    def forward(self, x):
        features = self.vit(x)
        return self.histo_head(features)
