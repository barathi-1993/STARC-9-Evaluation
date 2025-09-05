import numpy as np
import cv2, csv
from sklearn.metrics import (
    jaccard_score, 
    f1_score, 
    matthews_corrcoef, 
    cohen_kappa_score, 
    normalized_mutual_info_score,
    adjusted_rand_score
)
import matplotlib.pyplot as plt
import os, re
from scipy.spatial.distance import directed_hausdorff
from skimage import measure
from tqdm import tqdm

def advanced_segmentation_metrics(ground_truth, predicted_mask): 
    # Flatten arrays for metric calculations
    y_true = ground_truth.flatten()
    y_pred = predicted_mask.flatten()
    
    # Basic Performance Metrics
    metrics = {
        # Traditional Metrics (in percentage)
        'iou': jaccard_score(y_true, y_pred, average='binary') * 100,
        'dice': f1_score(y_true, y_pred, average='binary') * 100,
        'f1_score': f1_score(y_true, y_pred, average='binary') * 100,
        
        # Classification Metrics (in percentage)
        'matthews_corr_coef': matthews_corrcoef(y_true, y_pred) * 100,
        'cohen_kappa': cohen_kappa_score(y_true, y_pred) * 100,
        
        # Information Theoretic Metrics (in percentage)
        'normalized_mutual_info': normalized_mutual_info_score(y_true, y_pred) * 100,
        'adjusted_rand_index': adjusted_rand_score(y_true, y_pred) * 100
    }
    
    # Confusion Matrix Based Metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)
    metrics.update({
        'true_positive': tp,
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn,
        
        # Precision and Recall Variants (in percentage)
        'precision': (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0,
        'recall': (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0,
        'specificity': (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0,
        
        # Pixel-level Accuracy (in percentage)
        'pixel_accuracy': (tp + tn) / (tp + tn + fp + fn) * 100,
        'balanced_accuracy': ((tp / (tp + fn) + tn / (tn + fp)) / 2) * 100
    })
    
    # Morphological Analysis
    metrics.update(morphological_analysis(ground_truth, predicted_mask))
    
    return metrics

def confusion_matrix(y_true, y_pred): 
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    
    return tn, fp, fn, tp

def morphological_analysis(ground_truth, predicted_mask): 
    # Contour detection
    gt_contours = measure.find_contours(ground_truth, 0.5)
    pred_contours = measure.find_contours(predicted_mask, 0.5)
    
    # Contour-based metrics
    contour_metrics = {
        'gt_contour_count': len(gt_contours),
        'pred_contour_count': len(pred_contours),
    }
    
    # Hausdorff distance for contour similarity
    try:
        hausdorff_distances = [
            min(
                directed_hausdorff(gt_contour, pred_contour)[0] 
                for pred_contour in pred_contours
            )
            for gt_contour in gt_contours
        ]
        contour_metrics.update({
            'mean_hausdorff_distance': np.mean(hausdorff_distances) if hausdorff_distances else 0,
            'max_hausdorff_distance': np.max(hausdorff_distances) if hausdorff_distances else 0
        })
    except Exception as e:
        print(f"Error in Hausdorff distance calculation: {e}")
        contour_metrics.update({
            'mean_hausdorff_distance': 0,
            'max_hausdorff_distance': 0
        })
    
    return contour_metrics

def compare_masks(ground_truth_path, predicted_mask_path, output_dir, mode):  
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract filenames (without extension)
    ground_truth_filename   = os.path.splitext(os.path.basename(ground_truth_path))[0]
    predicted_mask_filename = os.path.splitext(os.path.basename(predicted_mask_path))[0]
    
    # Load masks as grayscale
    qupath_mask    = cv2.imread(ground_truth_path,   cv2.IMREAD_GRAYSCALE)
    predicted_mask = cv2.imread(predicted_mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Convert to binary (0=background, 1=tissue)
    qupath_binary    = (qupath_mask    >= 128).astype(np.uint8)
    predicted_binary = (predicted_mask >= 128).astype(np.uint8)
    
    # Calculate metrics
    metrics = advanced_segmentation_metrics(qupath_binary, predicted_binary)
    
    # Build a color overlay
    h, w = qupath_binary.shape
    comparison = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Ground-truth only ? Magenta
    gt_only = (qupath_binary == 1) & (predicted_binary == 0)
    comparison[gt_only] = [216, 0, 115] 
    
    # Predicted only ? Blue
    pred_only = (predicted_binary == 1) & (qupath_binary == 0)
    comparison[pred_only] =[0, 0, 255]
    
    # Overlap ? Lime 
    overlap = (qupath_binary == 1) & (predicted_binary == 1)
    comparison[overlap] = [164, 196, 0]
    
    # Save simple overlay TIFF
    comparison_path = os.path.join(
        output_dir,
        f"{predicted_mask_filename}_{ground_truth_filename}_comparison.tif"
    )
    cv2.imwrite(comparison_path, comparison)
    
    # Write metrics to text file
    metrics_path = os.path.join(
        output_dir,
        f"{predicted_mask_filename}_{ground_truth_filename}_metrics.txt"
    )
    with open(metrics_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.2f}\n")
    
    # Optionally print metrics
    if mode == 'single':
        for key, value in metrics.items():
            print(f"{key}: {value:.2f}")
    
    # Detailed side-by-side figure
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(qupath_binary * 255, cmap='gray')
    plt.title('Ground Truth (GT) Mask')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(predicted_binary * 255, cmap='gray')
    plt.title('Predicted (Pred) Mask')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(comparison)
    plt.title('Yellow=GT, Blue=Pred, Orange=Overlap')
    plt.axis('off')
    
    plt.tight_layout()
    detailed_comparison_path = os.path.join(
        output_dir,
        f"{predicted_mask_filename}_{ground_truth_filename}_detailed.tif"
    )
    plt.savefig(detailed_comparison_path, dpi=300)
    plt.close()
    
    return metrics

def batch_segmentation_analysis(ground_truth_dir, predicted_masks_dir, output_dir, model):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare for batch analysis
    all_metrics = []
    
    # Get matching files
    ground_truth_pattern = re.compile(r'^P-\d{4}_purple_grid_mask\.tiff$')
    if model == 'AutoEncoder' or model == 'HistoQC':
        generated_mask_pattern = re.compile(r'^P-\d{4}_remapped_part1\.tiff$')
    elif model == 'CLAM':
        generated_mask_pattern = re.compile(r'^P-\d{4}_purple_mask_compressed\.tiff$')
    else:  
        generated_mask_pattern = re.compile(r'^P-\d{4}_remapped_informative_part\.tiff$') 
    
    # Filter and sort ground truth files
    ground_truth_files = sorted([
        f for f in os.listdir(ground_truth_dir)
        if os.path.isfile(os.path.join(ground_truth_dir, f)) and ground_truth_pattern.match(f)
    ])
    
    # Filter and sort generated mask files
    predicted_mask_files = sorted([
        f for f in os.listdir(predicted_masks_dir)
        if os.path.isfile(os.path.join(predicted_dir, f)) and predicted_mask_pattern.match(f)
    ])
    
    # Prepare CSV file for writing results
    csv_path = os.path.join(output_dir, f'{model}_segmentation_metrics.csv')
    
    # Metrics to be extracted for CSV
    metric_keys = [
        'iou', 'dice', 'f1_score', 'matthews_corr_coef', 'cohen_kappa',
        'precision', 'recall', 'specificity', 'pixel_accuracy', 'balanced_accuracy'
    ]
    
    # Write CSV header
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Header with slide ID and metrics
        header = ['Slide_ID'] + metric_keys
        csv_writer.writerow(header) 
        
        # Match and process files with progress bar
        for gt_file, gen_file in tqdm(zip(ground_truth_files, predicted_mask_files), 
                                       total=len(ground_truth_files), 
                                       desc="Processing Masks"):
            try: 
                # Extract slide ID (assumes P-XXXX format)
                slide_id = re.search(r'(P-\d{4})', gt_file).group(1)
                
                # Construct full paths
                gt_path = os.path.join(ground_truth_dir, gt_file)
                gen_path = os.path.join(predicted_masks_dir, gen_file)
                
                # Perform comparison
                metrics = compare_masks(gt_path, gen_path, output_dir, mode='batch')
                
                # Prepare row for CSV
                row = [slide_id]
                row.extend([f"{metrics.get(key, 'N/A'):.2f}" for key in metric_keys])
                
                # Write individual image metrics
                csv_writer.writerow(row)
                
                # Store for summary statistics
                all_metrics.append(metrics) 
                    
            except Exception as e:
                print(f"Error processing {gt_file} and {gen_file}: {e}")
        
        # Calculate and write summary statistics
        if all_metrics:
            # Prepare summary row
            summary_row = ['Summary']
            
            # Calculate mean and std for each metric
            for key in metric_keys:
                values = [m.get(key, 0) for m in all_metrics if key in m]
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    summary_row.append(f'{mean_val:.2f} ± {std_val:.2f}')
                else:
                    summary_row.append('N/A')
            
            # Write summary row
            csv_writer.writerow(summary_row)
    
    print(f"Batch analysis complete. Results written to {csv_path}")
    return csv_path
 
# Example usage:
if __name__ == "__main__":
       
    ground_truth_path = '/path/to/2048_patsize_ground_truth_mask.tif'
    predicted_mask_path = '/path/to/save_predicted_mask.tif'
    output_dir = '/path/to/save_mask_comparison_output'
    print("\nAnalysis started. It might take several minutes...!!!\n") 
    compare_masks(ground_truth_path, predicted_mask_path, output_dir, mode='single')
    print(f"\nAnalysis completed. The results are stored in {output_dir}!!!\n")
        