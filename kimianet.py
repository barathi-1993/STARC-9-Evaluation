import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights
from config import MODEL_PATHS

class KimiaNet(nn.Module):
    def __init__(self, num_classes=9, pretrained=True):
        super(KimiaNet, self).__init__()
        # Load DenseNet121 pretrained on ImageNet
        base_model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Separate features and classifier
        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Create a new classifier
        self.classifier = nn.Linear(1024, num_classes)
        
        # Initialize classifier weights
        nn.init.xavier_normal_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)
    
    def forward(self, x):
        # Extract features
        x = self.features(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        
        # Classification logits
        logits = self.classifier(features)
        
        # Return both features and logits
        return features, logits

def load_kimianet(model, weights_path=None):
    if weights_path:
        print(f"Loading KimiaNet weights from {weights_path}")
        try:
            # Load weights
            weights = torch.load(weights_path, map_location='cpu')
            
            # Check if weights are in a checkpoint format
            if isinstance(weights, dict) and 'state_dict' in weights:
                weights = weights['state_dict']
            
            # Load weights to features, ignoring the classifier
            # Create a new state dict with only the feature extractor keys
            features_state_dict = {}
            for key, value in weights.items():
                # Only copy weights for the features part
                if 'classifier' not in key and 'fc' not in key:
                    if key.startswith('features.'):
                        features_state_dict[key] = value
                    else:
                        # Handle case where keys don't have 'features.' prefix
                        features_state_dict[f'features.{key}'] = value
            
            # Load weights with strict=False to ignore missing keys (classifier)
            model.load_state_dict(features_state_dict, strict=False)
            print("KimiaNet weights loaded successfully")
        except Exception as e:
            print(f"Error loading KimiaNet weights: {e}")
            print("Using ImageNet pretrained DenseNet121 instead")
    else:
        print("No KimiaNet weights provided, using ImageNet pretrained DenseNet121")
    
    return model

def get_kimianet(num_classes=9):
    # Create model
    model = KimiaNet(num_classes=num_classes, pretrained=True)
    
    # Get path to KimiaNet weights
    kimianet_weights_path = MODEL_PATHS.get('kimianet', None)
    
    # Load weights
    model = load_kimianet(model, kimianet_weights_path)
    
    return model