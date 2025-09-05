import torch
import torch.nn as nn
import timm

# Import custom models

from CNN_model import Model as CustomCNN
from HistoViT_model import HistoViT as CustomHistoViT
from config import MODEL_PATHS

def get_custom_cnn(num_classes=9, pretrained=False):
    model = CustomCNN(num_classes=num_classes)
    return model

def get_custom_histovit(num_classes=9, pretrained=False):
    model = CustomHistoViT(num_classes=num_classes)
    return model
