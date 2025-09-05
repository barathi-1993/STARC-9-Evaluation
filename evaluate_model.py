#!/usr/bin/env python3
"""
Usage:
  python evaluate_model.py --model_name transpath(our_best_model) --batch_size 32
"""

import os
import shutil
import json
import argparse
from collections import Counter

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from config import LABEL_MAP, CLASS_NAMES, NUM_CLASSES, IMG_SIZE, NORMALIZATION_MEAN, NORMALIZATION_STD
from main import get_model

# Common test dataset root (subfolders per class)
COMMON_DATA_PATH = "/path/to/Validation_data/CURATED-TCGA-CRC-HE-20K-NORMALIZED" # TCGA_CRC validation  
                                or
COMMON_DATA_PATH = "/path/to/Validation_data/STANFORD-CRC-HE-VAL-LARGE"  # STANFORD-CRC validation

# Common base directory for evaluation outputs; each model will get its own subfolder
COMMON_SAVE_DIR = "/path/to/save_results"

# Map from model_name to its checkpoint path
MODEL_CONFIG = {
    "resnet50": "/absolute/path/to/your/best_resnet50.pth",
    
    "efficientnet": "/absolute/path/to/your/best_efficientnet.pth",
    
    "vit_base": "/absolute/path/to/your/best_vit_base.pth",
    
    "swin_base": "/absolute/path/to/your/best_swin_base.pth",
    
    "convnext": "/absolute/path/to/your/best_convnext.pth",
    
    "deit": "/absolute/path/to/your/best_deit.pth",
    
    "hipt": "/absolute/path/to/your/best_hipt.pth",
    
    "transpath": "/absolute/path/to/your/best_transpath.pth",
    
     "pathdino": "/absolute/path/to/your/best_pathdino.pth",
    
     "vim4path": "/absolute/path/to/your/best_vim4path.pth",
    
     "conch": "/absolute/path/to/your/best_conch.pth",
    
     "virchow": "/absolute/path/to/your/best_virchow.pth",
    
     "uni": "/absolute/path/to/your/best_uni.pth",
    
     "prov_gigapath": "/absolute/path/to/your/best_prov_gigapath.pth",
    
     "custom_cnn": "/absolute/path/to/your/best_custom_cnn.pth",
    
     "custom_histovit": "/absolute/path/to/your/best_custom_histovit.pth",
    
     "kimianet": "/absolute/path/to/your/best_kimianet.pth"
     
}

#DATASET 

class TestCRCDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.transform = transform or transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
        ])
        self.image_files = []
        self.labels      = []

        # now pick up BOTH "Adipose" and "ADI" (etc.)
        for folder_name, label_idx in LABEL_MAP.items():
            subdir = os.path.join(folder_path, folder_name)
            if not os.path.isdir(subdir):
                continue
            for fname in os.listdir(subdir):
                if fname.lower().endswith(('.png','.jpg','jpeg','tif','tiff')):
                    self.image_files.append(os.path.join(subdir, fname))
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = self.image_files[idx]
        img  = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx], path

#MAIN EVALUATION

def main():
    parser = argparse.ArgumentParser(description="Evaluate selected CRC model on common test set")
    parser.add_argument('--model_name', required=True,
                        choices=list(MODEL_CONFIG.keys()),
                        help='Name of the model to evaluate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    args = parser.parse_args()

    # Resolve checkpoint & output dirs
    checkpoint_path = MODEL_CONFIG[args.model_name]
    results_dir = os.path.join(COMMON_SAVE_DIR, args.model_name)
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(results_dir, cls), exist_ok=True)

    # Device and model loading
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Loading checkpoint for {args.model_name}: {checkpoint_path}")
    model = get_model(args.model_name, num_classes=NUM_CLASSES)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    # Handle DataParallel prefixes if needed
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()

    # Dataset & DataLoader
    dataset = TestCRCDataset(COMMON_DATA_PATH)
    total = len(dataset)
    print(f"Total test samples: {total}")
    if total == 0:
        raise RuntimeError(f"No images found under {COMMON_DATA_PATH}")

    counts = Counter(dataset.labels)
    print("Class distribution in test set:")
    for idx, cls in enumerate(CLASS_NAMES):
        print(f"  {cls:<25}: {counts.get(idx, 0)}")

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=4)

    # Inference + copy tiles
    all_labels, all_preds, all_probs = [], [], []
    print("Running inference and copying tiles...")
    with torch.no_grad():
        for images, labels, paths in tqdm(loader, desc="Batches", ncols=80):
            images = images.to(device)
            outputs = model(images)
            if isinstance(outputs, (tuple, list)):
                logits = outputs[-1]
            else:
                logits = outputs
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.append(probs)
            all_labels.extend(labels)
            all_preds.extend(preds.tolist())

            for pth, p in zip(paths, preds):
                dst = os.path.join(results_dir, CLASS_NAMES[p], os.path.basename(pth))
                shutil.copy(pth, dst)

    # Metrics calculations
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    probs = np.vstack(all_probs)

    acc        = accuracy_score(y_true, y_pred)
    prec_m     = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_m      = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_m       = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_Micro   = f1_score(y_true, y_pred, average='micro', zero_division=0)
    cm         = confusion_matrix(y_true, y_pred)
    prec_pc    = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_pc     = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_pc      = f1_score(y_true, y_pred, average=None, zero_division=0)
    acc_pc     = np.diag(cm) / (cm.sum(axis=1) + 1e-12)

    # Save JSON & CSV
    metrics = {
        'accuracy': acc,
        'precision_macro': prec_m,
        'recall_macro': rec_m,
        'f1_macro': f1_m,
        'f1_micro': f1_Micro,
        'confusion_matrix': cm.tolist(),
        'per_class': {
            cls: {
                'precision': float(prec_pc[i]),
                'recall':    float(rec_pc[i]),
                'f1':        float(f1_pc[i]),
                'accuracy':  float(acc_pc[i])
            } for i, cls in enumerate(CLASS_NAMES)
        }
    }
    with open(os.path.join(results_dir, f"{args.model_name}_evaluation.json"), 'w') as f:
        json.dump(metrics, f, indent=4)

    df = pd.DataFrame({
        'class':    CLASS_NAMES,
        'precision': prec_pc,
        'recall':    rec_pc,
        'f1_score':  f1_pc,
        'accuracy':  acc_pc
    })
    df.to_csv(os.path.join(results_dir, f"{args.model_name}_per_class_metrics.csv"), index=False)

    # Plots 
    # Confusion matrix
    plt.figure(figsize=(10, 15), dpi=300)
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.title('Confusion Matrix')
    ticks = np.arange(len(CLASS_NAMES))
    plt.xticks(ticks, CLASS_NAMES, rotation=45)
    plt.yticks(ticks, CLASS_NAMES)
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, cm[i, j], ha='center',
                 color='white' if cm[i, j] > thresh else 'black', fontsize=8)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{args.model_name}_confusion_matrix.png"))
    plt.close()

    # ROC Curves
    y_bin = label_binarize(y_true, classes=list(range(len(CLASS_NAMES))))
    plt.figure(figsize=(10, 15), dpi=300)
    for i, cls in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        auc_score   = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC={auc_score:.2f})")
    fpr_m, tpr_m, _ = roc_curve(y_bin.ravel(), probs.ravel())
    plt.plot(fpr_m, tpr_m, '--', label=f"micro (AUC={auc(fpr_m, tpr_m):.2f})")
    plt.plot([0,1], [0,1], linestyle=':', color='gray')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curves')
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{args.model_name}_roc_curve.png"))
    plt.close()

    # Precision-Recall Curves
    plt.figure(figsize=(10, 15), dpi=300)
    for i, cls in enumerate(CLASS_NAMES):
        p, r, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
        plt.plot(r, p, label=cls)
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curves')
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{args.model_name}_precision_recall.png"))
    plt.close()

    # Per-class bar plot
    x = np.arange(len(CLASS_NAMES))
    w = 0.25
    plt.figure(figsize=(12, 8), dpi=300)
    plt.bar(x - w, prec_pc, width=w, label='Precision')
    plt.bar(x,rec_pc, width=w, label='Recall')
    plt.bar(x + w,f1_pc, width=w, label='F1-Score')
    plt.xticks(x, CLASS_NAMES, rotation=45)
    plt.ylim(0,1); plt.ylabel('Score'); plt.title('Per-Class Metrics')
    plt.legend(); plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{args.model_name}_per_class_bar.png"))
    plt.close()

    # Console Summary 
    print(f"\nOverall: Accuracy={acc:.4f}, Precision={prec_m:.4f}, Recall={rec_m:.4f}, "
          f"F1-Macro={f1_m:.4f}, F1-Micro={f1_Micro:.4f}")
    print("\nPer-class metrics:")
    print(df.to_string(index=False))
    print(f"\nResults and plots saved under {results_dir}")

if __name__ == '__main__':
    main()
