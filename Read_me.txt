README.txt for STARC-9 Dataset & Benchmark Code
===============================================

Dataset_folder_structure(Hugging_face)

Path2AI/STARC-9
     |---Training_data_normalized
     	 |---ADI
         |---LYM
         |---MUS
         |---MUC
	 |---BLD
	 |---TUM
	 |---NOR
	 |---NCS
	 |---FCT

     |---Validation_data
	 |---CURATED-TCGA-CRC-HE-20K-NORMALIZED
		|---ADI
         	|---LYM
         	|---MUS
         	|---MUC
	 	|---BLD
	 	|---TUM
	 	|---NOR
	 	|---NCS
	 	|---FCT         
         |---STANFORD-CRC-HE-VAL-LARGE
		|---ADI
         	|---LYM
         	|---MUS
         	|---MUC
	 	|---BLD
	 	|---TUM
	 	|---NOR
	 	|---NCS
	 	|---FCT

===================================================

1. Setup Environment
--------------------
# Create conda env with Python 3.12

conda create -n starc9 python=3.12

conda activate starc9

# Install required packages

pip install torch torchvision timm pandas numpy matplotlib seaborn scikit-learn umap-learn tqdm pillow transformers
=======================================================

2. Organize Directory Structure
-------------------------------
Place all source files in a single project directory:

config.py 

dataset.py

models.py

custom_models.py

foundation_models.py

CNN_model.py

HistoViT_model.py

Kimianet.py

trainer.py

main.py

run_benchmark.py

evaluate_model.py
=========================================================

3.Training Individual Models
----------------------------

To train a single model:(refer config.py for additional argument passing)

python main.py --model modelname --epochs 10 --batch_size 32 --multi_gpu(optional)
(eg: python main.py --model transpath --epochs 10 --batch_size 32 --multi_gpu)

=========================================================

4. Train & evaluate all models
------------------------------
To run the full benchmark across all models:

python run_benchmark.py --epochs 10 --batch_size 32 --multi_gpu

This will:

Train and evaluate each model
ave per-model results under SAVE_DIR/benchmark_<timestamp>/
Generate summary CSV (benchmark_summary.csv)
Produce comparison plots in plots/

==========================================================

5. Evaluate the model on the validation/Test set (both internal/external)
------------------------------------------------------------------------

python evaluate_model.py --model modelname --batch_size --data_path <path_to_validation_data>

Outputs:

Per-class metrics JSON

Confusion matrix PNG

(Optional)copies of misclassified tiles into class folders

============================================================================================================================================

To perform/Evaluate downstream task (Tumor Segmentation - Tile/Patch level of size 2048)

        step 1: Classify tiles extracted from a patch and Normalize(Macenko or any).

	step 2: Remap predicted tumor classified tiles to the patch that intersects with the groundtruth mask(2048).

	step 3: Segmentation evaluation and compare the results.

6.Classify extracted tiles (inferencing) from a patch (2048) with best trained model weights
---------------------------------------------------------------------------------------------

python Classifiy_extracted_tiles_from_a_wsi_with_best_trained_model_weights.py (this also applies for WSI level as well)


7.Remap predicted tumor classified tiles to the patch that intersects with the groundtruth mask(2048)
-----------------------------------------------------------------------------------------------------

python Remap_tumor_patch_segmentation.py

8.Segmentation evaluation and comparison results
------------------------------------------------

python Segmentation_evaluation.py 




