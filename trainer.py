import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import json
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from config import SAVE_DIR, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, CLASS_NAMES, METRICS

import warnings
warnings.filterwarnings("ignore")


def train_model(model, criterion, optimizer, scheduler, train_loader, test_loader=None, 
                num_epochs=NUM_EPOCHS, device=None, save_dir=SAVE_DIR, model_name="model"):
    """
    Train the model on full dataset. Only training metrics are logged.
    Best model is saved based on training F1 score.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(save_dir, f"{model_name}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Save paths
    best_model_path = os.path.join(model_dir, f"best_{model_name}.pth")
    final_model_path = os.path.join(model_dir, f"final_{model_name}.pth")
    history_path = os.path.join(model_dir, f"history_{model_name}.csv")
    epoch_metrics_path = os.path.join(model_dir, f"epoch_metrics_{model_name}.csv")
    
    # Print model details
    print("\n" + "="*80)
    print(f"MODEL: {model_name}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print("Training on full dataset (no validation split)")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Device: {device}")
    print(f"Save directory: {model_dir}")
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("="*80 + "\n")
    
    # Initialize best metrics and history dictionary (only training metrics)
    best_train_f1 = 0.0
    history = {
        'epoch': [], 'train_loss': [], 'train_acc': [], 'train_f1': [],
        'train_precision': [], 'train_recall': [], 'train_f1_micro': [],
        'learning_rate': [], 'epoch_time': [], 'train_time': []
    }
    
    # Write CSV header for per-epoch metrics
    with open(epoch_metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_loss', 'train_acc', 'train_f1', 'train_precision', 
            'train_recall', 'train_f1_micro', 'learning_rate', 'epoch_time', 'train_time'
        ])
    
    start_time = time.time()
    epoch_pbar = tqdm(range(num_epochs), desc=f"Training {model_name}", position=0)
    
    # Training loop
    for epoch in epoch_pbar:
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        history['epoch'].append(epoch+1)
        history['learning_rate'].append(current_lr)
        
        # Training phase
        try:
            model.train()
            running_loss = 0.0
            all_preds = []
            all_labels = []
            train_start_time = time.time()
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)", 
                              position=1, leave=False)
            
            for inputs, labels in train_pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                if isinstance(outputs, (tuple, list)):
                    logits = outputs[-1]
                else:
                    logits = outputs

                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(logits, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            train_loss = running_loss / len(train_loader.dataset)
            train_acc = accuracy_score(all_labels, all_preds)
            train_f1 = f1_score(all_labels, all_preds, average='macro')
            train_f1_micro = f1_score(all_labels, all_preds, average='micro')
            train_precision = precision_score(all_labels, all_preds, average='macro')
            train_recall = recall_score(all_labels, all_preds, average='macro')
            train_time = time.time() - train_start_time
            
            # Store training metrics
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_f1'].append(train_f1)
            history['train_f1_micro'].append(train_f1_micro)
            history['train_precision'].append(train_precision)
            history['train_recall'].append(train_recall)
            history['train_time'].append(train_time)
            
        except Exception as e:
            print(f"Error during training epoch {epoch+1}: {e}")
            # Set dummy values
            train_loss, train_acc, train_f1, train_f1_micro = float('nan'), float('nan'), float('nan'), float('nan')
            train_precision, train_recall = float('nan'), float('nan')
            train_time = 0.0
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_f1'].append(train_f1)
            history['train_f1_micro'].append(train_f1_micro)
            history['train_precision'].append(train_precision)
            history['train_recall'].append(train_recall)
            history['train_time'].append(train_time)
        
        epoch_time = time.time() - epoch_start_time
        history['epoch_time'].append(epoch_time)
        
        # Step the scheduler
        try:
            scheduler.step()
        except Exception as e:
            print(f"Scheduler step error at epoch {epoch+1}: {e}")
        
        # Save model if training F1 improves
        if not np.isnan(train_f1) and train_f1 > best_train_f1:
            best_train_f1 = train_f1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'train_f1': train_f1,
                'train_precision': train_precision,
                'train_recall': train_recall,
                'train_f1_micro': train_f1_micro,
            }, best_model_path)
            epoch_pbar.set_postfix({
                "Best Train F1": f"{best_train_f1:.4f}",
                "Saved": "Yes"
            })
        else:
            epoch_pbar.set_postfix({
                "Best Train F1": f"{best_train_f1:.4f}",
                "Current F1": f"{train_f1:.4f}"
            })
        
        # Append epoch metrics to CSV
        with open(epoch_metrics_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, train_loss, train_acc, train_f1, train_precision, train_recall, train_f1_micro,
                current_lr, epoch_time, train_time
            ])
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1-Macro: {train_f1:.4f}, F1-Micro: {train_f1_micro:.4f}")
        print(f"       Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, Time: {train_time:.2f}s")
        print(f"Epoch Time: {epoch_time:.2f}s, Learning Rate: {current_lr:.6f}")
        print(f"Best Train F1: {best_train_f1:.4f}")
        print("-" * 80)
    
    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'final_train_loss': train_loss,
        'final_train_acc': train_acc,
        'final_train_f1': train_f1,
    }, final_model_path)
    
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTraining complete in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Best training F1: {best_train_f1:.4f}")
    print(f"Final model saved to: {final_model_path}")
    print(f"Best model saved to: {best_model_path}")
    
    history_df = pd.DataFrame(history)
    history_df.to_csv(history_path, index=False)
    
    # Load best model for return
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history_df, best_train_f1


def evaluate_model(model, test_loader, device=None, class_names=CLASS_NAMES):
    """
    Evaluate model on test dataset (kept for compatibility with existing code).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)

            if isinstance(outputs, (tuple, list)):
                logits = outputs[-1]
            else:
                logits = outputs
                
            probabilities = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average='macro')
    recall_macro = recall_score(all_labels, all_preds, average='macro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    
    precision_per_class = precision_score(all_labels, all_preds, average=None)
    recall_per_class = recall_score(all_labels, all_preds, average=None)
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    
    cm = confusion_matrix(all_labels, all_preds)
    
    results = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'precision_per_class': {class_names[i]: precision_per_class[i] for i in range(len(class_names))},
        'recall_per_class': {class_names[i]: recall_per_class[i] for i in range(len(class_names))},
        'f1_per_class': {class_names[i]: f1_per_class[i] for i in range(len(class_names))},
        'confusion_matrix': cm.tolist(),
    }
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision (Macro): {precision_macro:.4f}')
    print(f'Recall (Macro): {recall_macro:.4f}')
    print(f'F1 Score (Macro): {f1_macro:.4f}')
    print(f'F1 Score (Micro): {f1_micro:.4f}')
    
    return results, all_preds, all_labels, all_probs


def plot_training_history(history_df, save_path=None):
    """Plot training history (only training metrics)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training loss
    ax1.plot(history_df['epoch'], history_df['train_loss'], label='Train Loss', color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Training metrics
    ax2.plot(history_df['epoch'], history_df['train_acc'], label='Train Accuracy', color='green')
    ax2.plot(history_df['epoch'], history_df['train_f1'], label='Train F1-Macro', color='red', linestyle='--')
    ax2.plot(history_df['epoch'], history_df['train_f1_micro'], label='Train F1-Micro', color='orange', linestyle=':')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Metrics')
    ax2.set_title('Training Metrics')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox