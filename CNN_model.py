import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_classes=9):  # Default to 9 for multiclass classification
        super(Model, self).__init__()
        # Define convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )
        
        # Define classifier components
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)  # Output layer for 9 classes
        self.softmax = nn.Softmax(dim=1)  # For inference only

    def forward(self, x, return_features=False, feature_type=None):
        # Forward through conv layers
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        
        # If we need to return specific convolutional features
        if return_features and feature_type in ['conv4', 'conv5', 'conv6']:
            if feature_type == 'conv4':
                return x4.mean(dim=(2, 3))  # Global average pooling
            elif feature_type == 'conv5':
                return x5.mean(dim=(2, 3))
            elif feature_type == 'conv6':
                return x6.mean(dim=(2, 3))
        
        # Continue with classifier
        pooled = self.pool(x6)
        flattened = self.flatten(pooled)
        fc1_output = self.fc1(flattened)
        relu_output = self.relu(fc1_output)

        # Return features from the penultimate layer if requested
        if return_features and feature_type == 'penultimate':
            return relu_output

        # Final classification layer
        output = self.fc2(relu_output)
        
        # Return final output features if requested
        if return_features and feature_type == 'output':
            return output
            
        return output
        
    def get_features(self, x, feature_type):
        """Alternative method to get features from specific layers"""
        features = {}
        
        # Forward through conv layers
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        
        # Extract convolutional features
        features['conv4'] = x4.mean(dim=(2, 3))
        features['conv5'] = x5.mean(dim=(2, 3))
        features['conv6'] = x6.mean(dim=(2, 3))
        
        # Classifier features
        pooled = self.pool(x6)
        flattened = self.flatten(pooled)
        fc1_output = self.fc1(flattened)
        relu_output = self.relu(fc1_output)
        features['penultimate'] = relu_output
        
        # Final output
        output = self.fc2(relu_output)
        features['output'] = output
        
        if feature_type == 'all':
            return features
        return features[feature_type]
        
    def predict_with_confidence(self, x):
        """Returns class predictions with confidence scores"""
        outputs = self.forward(x)
        probabilities = self.softmax(outputs)
        conf_values, predicted_classes = torch.max(probabilities, dim=1)
        return predicted_classes, conf_values