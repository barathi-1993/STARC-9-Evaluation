import os

# Paths
TRAIN_FOLDER_PATH = "/path/to/Training_data_normalized/

SAVE_DIR = "/path/to/save the Results"


# 1) Original “full” class names indices
ORIG_LABEL_MAP = {
    "Adipose":                 0,
    "Lymphocyte":              1,
    "Mucin":                   2,
    "Muscle":                  3,
    "Necrotic_debris":         4,
    "Normal":                  5,
    "Red_blood_cells":         6,
    "Loose_connective_tissue": 7,
    "Tumor":                   8,
}

# 2) Short-code aliases those same canonical names
ALIASES = {
    "ADI": "Adipose",
    "LYM": "Lymphocyte",
    "MUC": "Mucin",
    "MUS": "Muscle",
    "NCS": "Necrotic_debris",
    "NOR": "Normal",
    "BLD": "Red_blood_cells",
    "FCT": "Loose_connective_tissue",
    "TUM": "Tumor",
}

# 3) Unified map so we can use either full names or short codes as folder keys
LABEL_MAP = {
    **ORIG_LABEL_MAP,
    **{ short: ORIG_LABEL_MAP[full] for short, full in ALIASES.items() }
}

# 4) Build the list of short codes in index order for display/plots/JSON keys
SHORT_CLASS_NAMES = []
for full_name, idx in sorted(ORIG_LABEL_MAP.items(), key=lambda kv: kv[1]):
    # find the corresponding short code
    for short, full in ALIASES.items():
        if full == full_name:
            SHORT_CLASS_NAMES.append(short)
            break

# sanity check
assert len(SHORT_CLASS_NAMES) == len(ORIG_LABEL_MAP) == 9

# 5) Export the values your evaluation scripts expect:
CLASS_NAMES          = SHORT_CLASS_NAMES   # ["ADI","LYM","MUC","MUS","NCS","NOR","BLD","FCT","TUM"]
NUM_CLASSES          = len(CLASS_NAMES)    # 9


# Training Parameters
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-5

# Data Parameters
IMG_SIZE = 256  # Using native patch size of 256 
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]  # ImageNet stats
NORMALIZATION_STD = [0.229, 0.224, 0.225]  # ImageNet stats

# Validation Parameters
VAL_SPLIT = 0.0
TEST_SPLIT = 0.0

# Augmentation Parameters
AUG_ROTATION = 20
AUG_BRIGHTNESS = 0.1
AUG_CONTRAST = 0.1
AUG_SATURATION = 0.1
AUG_HUE = 0.05

# Common Evaluation Metrics
METRICS = ['accuracy', 'precision', 'recall', 'f1_macro', 'f1_micro']


# Model Paths (pretrained weights)
MODEL_PATHS = {
    'kimianet': 'path/to/KimiaNetPyTorchWeights.pth'
    # Add paths to model checkpoints if you have them
}