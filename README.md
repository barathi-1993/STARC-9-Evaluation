# STARC-9 Dataset & Benchmark Code

The **STARC-9** dataset is a curated colorectal cancer (CRC) histopathology benchmark with **nine tissue classes**:

- **ADI** — Adipose tissue  
- **LYM** — Lymphoid tissue  
- **MUS** — Muscle tissue  
- **MUC** — Mucin  
- **BLD** — Blood  
- **TUM** — Tumor  
- **NOR** — Normal colon mucosa  
- **NCS** — Necrosis  
- **FCT** — Fibroconnective tissue  

---

## 📂 Dataset Folder Structure

<pre>
📂 Path2AI/STARC-9
├── 📁 Training_data_normalized/
│   ├── 📁 ADI/
│   ├── 📁 LYM/
│   ├── 📁 MUS/
│   ├── 📁 MUC/
│   ├── 📁 BLD/
│   ├── 📁 TUM/
│   ├── 📁 NOR/
│   ├── 📁 NCS/
│   └── 📁 FCT/
└── 📁 Validation_data/
    ├── 📁 CURATED-TCGA-CRC-HE-20K-NORMALIZED/
    │   ├── 📁 ADI/
    │   ├── 📁 LYM/
    │   ├── 📁 MUS/
    │   ├── 📁 MUC/
    │   ├── 📁 BLD/
    │   ├── 📁 TUM/
    │   ├── 📁 NOR/
    │   ├── 📁 NCS/
    │   └── 📁 FCT/
    └── 📁 STANFORD-CRC-HE-VAL-LARGE/
        ├── 📁 ADI/
        ├── 📁 LYM/
        ├── 📁 MUS/
        ├── 📁 MUC/
        ├── 📁 BLD/
        ├── 📁 TUM/
        ├── 📁 NOR/
        ├── 📁 NCS/
        └── 📁 FCT/
</pre>

---

## ⚙️ 1. Setup Environment

Create a new conda environment with Python 3.12:

```bash
conda create -n starc9 python=3.12
conda activate starc9
```
Install required packages:

```bash
pip install torch torchvision timm pandas numpy matplotlib seaborn scikit-learn umap-learn tqdm pillow transformers
```
## 2. Organize Project Files

Place all source files in a single project directory:

```bash
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
```
## 🏋️ 3. Train an Individual Model

Train a single model (see config.py for additional arguments):

```bash
python main.py --model modelname --epochs 10 --batch_size 32 --multi_gpu
```
Example:

```bash
python main.py --model transpath --epochs 10 --batch_size 32 --multi_gpu
```
## 📊 4. Train & Evaluate All Models

Run the full benchmark across all models:

```bash
python run_benchmark.py --epochs 10 --batch_size 32 --multi_gpu
```
This will:

```bash
- Train and evaluate each model
- Save per-model results under SAVE_DIR/benchmark_<timestamp>/
- Generate summary CSV: benchmark_summary.csv
- Produce comparison plots in plots/
```
## ✅ 5. Evaluate on Validation/Test Sets

Evaluate a trained model on validation or test data:

```bash
python evaluate_model.py --model modelname --batch_size <bs> --data_path <path_to_validation_data>
```
Outputs:

```bash
- Per-class metrics JSON
- Confusion matrix PNG
- (Optional) misclassified tiles copied into class folders
```
## 🧩 6. Downstream Task: Tumor Segmentation (2048 Patch Level)

```bash
1) Classify tiles extracted from a 2048 patch and normalize (e.g., Macenko).
2) Remap predicted tumor tiles to the patch intersecting the ground-truth mask (2048).
3) Run segmentation evaluation and compare results.
```
## 🔎 7. Classify Extracted Tiles from a Patch

Use the best trained model weights (also applies for WSI-level classification):

```bash
python Classifiy_extracted_tiles_from_a_wsi_with_best_trained_model_weights.py
```
## 🔗 8. Remap Predicted Tumor Tiles to Patches

```bash
python Remap_tumor_patch_segmentation.py
```
## 📈 9. Segmentation Evaluation & Comparison

```bash
python Segmentation_evaluation.py
```



