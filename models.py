import torch
import torch.nn as nn
import timm
import torchvision.models as models
from torchvision.models import ResNet50_Weights, EfficientNet_B7_Weights

# ResNet Models
def get_resnet50(num_classes=9, pretrained=True):
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ViT Models - updated for 256x256 input
def get_vit_base(num_classes=9, pretrained=True):
    model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, img_size=256, num_classes=num_classes)
    return model


# Swin Transformer - updated for 256x256 input
def get_swin_base(num_classes=9, pretrained=True):
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained, img_size=256, num_classes=num_classes)
    return model

# ConvNeXt - already handles 256x256 input
def get_convnext(num_classes=9, pretrained=True):
    model = timm.create_model('convnext_base', pretrained=pretrained, num_classes=num_classes)
    return model

# EfficientNetB7 - already handles 256x256 input
def get_efficientnet(num_classes=9, pretrained=True):
    model = models.efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

# DeiT - updated for 256x256 input
def get_deit(num_classes=9, pretrained=True):
    model = timm.create_model('deit_base_patch16_224', pretrained=pretrained, img_size=256, num_classes=num_classes)
    return model