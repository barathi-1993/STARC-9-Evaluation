import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
from config import (
    LABEL_MAP, IMG_SIZE, NORMALIZATION_MEAN, NORMALIZATION_STD,
    AUG_ROTATION, AUG_BRIGHTNESS, AUG_CONTRAST, AUG_SATURATION, AUG_HUE, 
    BATCH_SIZE, TEST_SPLIT
)

class CRCDataset(Dataset):
    def __init__(self, folder_path, transform=None, is_training=True):
        self.folder_path = folder_path
        self.image_files = []
        self.labels = []
        self.is_training = is_training
        
        # Define transformations based on training/testing
        if transform is None:
            if is_training:
                self.transform = transforms.Compose([
                    transforms.Resize((IMG_SIZE, IMG_SIZE)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(AUG_ROTATION),
                    transforms.ColorJitter(
                        brightness=AUG_BRIGHTNESS,
                        contrast=AUG_CONTRAST,
                        saturation=AUG_SATURATION,
                        hue=AUG_HUE
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((IMG_SIZE, IMG_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
                ])
        else:
            self.transform = transform
        
        # Use the label map from config
        self.label_map = LABEL_MAP

        # Walk through the directory to find all class folders
        for root, dirs, files in os.walk(folder_path):
            class_name = os.path.basename(root)
            if class_name in self.label_map:  # Only consider folders that match our classes
                label = self.label_map[class_name]
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                        self.image_files.append(os.path.join(root, f))
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
	
        img_path = self.image_files[idx]
	
        img = Image.open(img_path).convert("RGB")
	
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

def get_data_loaders(folder_path, batch_size=BATCH_SIZE):
    """Create train data loader using full dataset (no split)"""
    # Create full dataset for training
    train_dataset = CRCDataset(folder_path, is_training=True)
    
    # Create train data loader with full dataset
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False
    )
    
    # Return None for test_loader since you have separate test data
    return train_loader, None