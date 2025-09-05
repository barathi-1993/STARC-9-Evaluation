import os
import shutil
import csv
import argparse
from tqdm import tqdm
import time
import config
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Process, Queue, Lock, Manager, get_context

from main import get_model


CHECKPOINTS = {
    "model_name": /absolute/path/to/your/best_model.pth",  # ? UPDATE this to your trained-model .pth       
}

class WSIClassifier:
    def __init__(self, root_dir, output_dir, model, class_names, confidence_threshold, tumor_threshold=None, batch_size=32): 
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.model = model
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold
        self.tumor_threshold = tumor_threshold
        self.batch_size = batch_size
        
        # Image transformation
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a shared lock for writing to the global CSV file
        self.csv_lock = Manager().Lock()
        
        # Initialize the global CSV with headers
        self.global_csv_path = os.path.join(self.output_dir, 'class_counts.csv')
        with self.csv_lock:
            with open(self.global_csv_path, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                # Write header for your classes only
                header = ['WSI_Name', 'Subdirectory', 'Total_Images']
                header.extend([f'{class_name}_Count' for class_name in self.class_names])
                header.extend([f'{class_name}_Ratio' for class_name in self.class_names])
                csv_writer.writerow(header)
    
    def process_batch(self, batch_paths, device, class_dirs, tumor_idx=None): 
        batch_images = []
        valid_paths = []
        
        # Load and transform images
        for img_path in batch_paths:
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = self.transform(image)
                batch_images.append(image_tensor)
                valid_paths.append(img_path)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        
        if not valid_paths:
            return {}
        
        # Stack batch and move to device
        batch_tensor = torch.stack(batch_images).to(device)
        
        # Classify batch
        with torch.no_grad():
            logits = self.model(batch_tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        
        # Track classification results for this batch
        batch_class_counts = defaultdict(int)
        
        # Process each image result
        for i, (img_path, probs) in enumerate(zip(valid_paths, probabilities)):
            img_filename = os.path.basename(img_path)
            
            # Determine predicted class
            predicted_idx = np.argmax(probs)
            if tumor_idx is not None and predicted_idx == tumor_idx:
                tumor_prob = probs[tumor_idx]
                if tumor_prob < self.tumor_threshold:
                    temp_probs = probs.copy()
                    temp_probs[tumor_idx] = -1  # Exclude tumor class
                    predicted_idx = np.argmax(temp_probs)
            
            predicted_label = self.class_names[predicted_idx]
            
            # Copy image to appropriate class folder
            dest_path = os.path.join(class_dirs[predicted_label], img_filename)
            shutil.copy2(img_path, dest_path)
            
            # Update batch counts
            batch_class_counts[predicted_label] += 1
            
            # Extract patient ID from the path
            path_parts = os.path.normpath(img_path).split(os.sep)
            patient_id = "unknown"
            for part in path_parts:
                if part.startswith("P-"):
                    patient_id = part
                    break
            
            # Get the base output directory for the patient
            patient_output_dir = os.path.join(self.output_dir, patient_id)
            os.makedirs(patient_output_dir, exist_ok=True)
            
            # Save probability to CSV
            probabilities_csv_path = os.path.join(patient_output_dir, 'probabilities.csv')
            file_exists = os.path.isfile(probabilities_csv_path)
            with open(probabilities_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    header = ['tile_name', 'predicted_label', 'max_confidence', 'tumor_probability']
                    header.extend([f'prob_{cls}' for cls in self.class_names])
                    writer.writerow(header)
                
                max_prob = np.max(probs)
                tumor_prob = probs[tumor_idx] if tumor_idx is not None else None
                row = [img_filename, predicted_label, max_prob, tumor_prob]
                row.extend(probs)
                writer.writerow(row)
        
        return batch_class_counts
        
    def process_wsi_folder(self, wsi_folder, device, gpu_id, sub_directory): 
        # Prepare output directories
        wsi_name = os.path.basename(wsi_folder)
        wsi_output_dir = os.path.join(self.output_dir, wsi_name)
        os.makedirs(wsi_output_dir, exist_ok=True)
        
        # Create class-specific subdirectories
        class_dirs = {
            class_name: os.path.join(wsi_output_dir, class_name) 
            for class_name in self.class_names
        }
        for dir_path in class_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # Track classification results
        class_counts = defaultdict(int)
        
        # Process images in specified subdirectory
        sub_dir_path = os.path.join(wsi_folder, sub_directory)
        if not os.path.exists(sub_dir_path):
            print(f"Subdirectory not found: {sub_dir_path}")
            return None
        
        # Get tumor class index if tumor threshold is specified
        tumor_idx = None
        if self.tumor_threshold is not None and "TUM" in self.class_names:
            tumor_idx = self.class_names.index("TUM")
        
        # Get all images in the subdirectory
        image_files = [os.path.join(sub_dir_path, f) for f in os.listdir(sub_dir_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        # Process images in batches with progress bar
        total_batches = (len(image_files) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in tqdm(range(total_batches), 
                              desc=f"GPU:{gpu_id} | {wsi_name} | {sub_directory}",
                              leave=False):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(image_files))
            batch_paths = image_files[start_idx:end_idx]
            
            batch_counts = self.process_batch(
                batch_paths, 
                device, 
                class_dirs, 
                tumor_idx
            )
            
            for class_name, count in batch_counts.items():
                class_counts[class_name] += count
        
        total_images = sum(class_counts.values())
        
        # Prepare results
        results = {
            'wsi_name': wsi_name,
            'subdirectory': sub_directory,
            'class_counts': dict(class_counts),
            'total_images': total_images,
            'class_ratios': {
                class_name: class_counts.get(class_name, 0) / total_images if total_images > 0 else 0
                for class_name in self.class_names
            }
        }
        
        # Update CSVs
        self.update_wsi_class_counts_csv(wsi_name, {sub_directory: results})
        self.append_to_global_csv(results)
        
        return results
    
    def update_wsi_class_counts_csv(self, wsi_name, results_by_subdirectory):
        combined_counts = defaultdict(int)
        total_images = 0
        
        for sub_results in results_by_subdirectory.values():
            for class_name, count in sub_results['class_counts'].items():
                combined_counts[class_name] += count
            total_images += sub_results['total_images']
        
        wsi_output_dir = os.path.join(self.output_dir, wsi_name)
        class_counts_csv_path = os.path.join(wsi_output_dir, 'class_counts.csv')
        
        with open(class_counts_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Class', 'Count', 'Percentage'])
            
            for class_name in self.class_names:
                count = combined_counts.get(class_name, 0)
                percentage = (count / total_images * 100) if total_images > 0 else 0
                writer.writerow([class_name, count, percentage])
    
    def append_to_global_csv(self, result):
        with self.csv_lock:
            with open(self.global_csv_path, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                
                row = [
                    result['wsi_name'],
                    result['subdirectory'],
                    result['total_images']
                ]
                
                for class_name in self.class_names:
                    row.append(result['class_counts'].get(class_name, 0))
                
                for class_name in self.class_names:
                    row.append(result['class_ratios'].get(class_name, 0))
                
                csv_writer.writerow(row)
    
    def process_wsis_on_gpu(self, gpu_id, wsi_folders, sub_directories, shared_results): 
        device = torch.device(f'cuda:{gpu_id}' if gpu_id >= 0 else 'cpu')
        model_copy = self.model.to(device)
        model_copy.eval()
        
        results = []
        for wsi_folder in wsi_folders:
            wsi_name = os.path.basename(wsi_folder)
            for sub_directory in sub_directories:
                print(f"Processing {wsi_name}/{sub_directory} on GPU:{gpu_id}")
                result = self.process_wsi_folder(wsi_folder, device, gpu_id, sub_directory)
                if result:
                    results.append(result)
                    shared_results.append(result)
        return results
    
    def process_all_wsis_multi_gpu(self, sub_directories, selected_wsis=None): 
        if isinstance(selected_wsis, str) and selected_wsis != "None":
            if ',' in selected_wsis:
                selected_wsis = [w.strip() for w in selected_wsis.split(',')]
            else:
                selected_wsis = [selected_wsis.strip()]
        
        wsi_folders = [
            os.path.join(self.root_dir, d) 
            for d in os.listdir(self.root_dir) 
            if os.path.isdir(os.path.join(self.root_dir, d))
        ]
        
        if selected_wsis and selected_wsis != ["None"]:
            wsi_folders = [f for f in wsi_folders if os.path.basename(f) in selected_wsis]
            if not wsi_folders:
                print(f"Warning: No matching WSI folders found for {selected_wsis}")
                return
        
        device_ids = list(range(torch.cuda.device_count()))
        num_gpus = len(device_ids) or 1
        if num_gpus == 0:
            device_ids = [-1]
        
        print("------------------------------------")
        print(f"Using {num_gpus} {'GPU' if num_gpus>1 else 'GPU'}: {device_ids}")
        print(f"Batch size: {self.batch_size}")
        print("------------------------------------")
        
        wsi_split = [[] for _ in range(num_gpus)]
        for idx, folder in enumerate(wsi_folders):
            wsi_split[idx % num_gpus].append(folder)
        
        for gpu_id in range(num_gpus):
            names = [os.path.basename(f) for f in wsi_split[gpu_id]]
            print(f"GPU {device_ids[gpu_id]} will process {len(names)} WSIs: {', '.join(names)}")
        print("------------------------------------")
        
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        
        manager = Manager()
        shared_results = manager.list()
        processes = []
        start_time = time.time()
        
        for gpu_id in range(num_gpus):
            p = Process(
                target=self.process_wsis_on_gpu,
                args=(device_ids[gpu_id], wsi_split[gpu_id], sub_directories, shared_results)
            )
            processes.append(p)
            p.start()
        
        for p in processes:
            p.join()
        
        elapsed = time.time() - start_time
        print(f"Processing completed in {elapsed:.2f} seconds")
        print(f"Results saved to {self.output_dir}")

if __name__ == "__main__":
    # --- UPDATE THESE TO YOUR FOLDERS --------------------------------------------
    # Folder that contains one subfolder per WSI, each with your extracted tiles
    root_dir    = "/absolute/path/to/extracted_tiles"             
    # Where you want the classified tiles & CSVs to be written
    output_dir  = "/absolute/path/to/save_classified_tiles"      
    # -----------------------------------------------------------------------------

    confidence_threshold = 0.0  # this can be adjusted between 0.0 to 1.0 according to the use case of the readers

    tumor_threshold      = 0.0   # this can be adjusted between 0.0 to 1.0 according to the use case of the readers

    batch_size           = 256

    # --- OPTIONALLY: specify which tile-subfolder under each WSI to process ---
    # If your folder structure is root_dir/P-0001/Tiles, set sub_directories="Tiles"
    # Otherwise leave as " " to process everything in root_dir/P-0001
    sub_directories = "Tiles"     #  UPDATE to your tile folder name (or leave as " ")
    # -----------------------------------------------------------------------------

    # --- OPTIONALLY: process only a subset of WSIs -----------------------------
    # Comma-separated list of WSI folder names (e.g. "P-0001,P-0023"), or None for all
    selected_wsis   = None        #  set to e.g. "P-0071,P-0123" or leave None
    # -----------------------------------------------------------------------------

    num_classes = len(config.CLASS_NAMES)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model       = get_model("model_name", num_classes=num_classes)
    # load checkpoint
    ckpt        = torch.load(CHECKPOINTS["model_name"], map_location=device)
    state       = ckpt.get("model_state_dict", ckpt)
    state       = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    class_names = config.CLASS_NAMES

    classifier = WSIClassifier(
        root_dir=root_dir,
        output_dir=output_dir,
        model=model,
        class_names=class_names,
        confidence_threshold=confidence_threshold,
        tumor_threshold=tumor_threshold,
        batch_size=batch_size
    )
    classifier.process_all_wsis_multi_gpu(sub_directories, selected_wsis)
