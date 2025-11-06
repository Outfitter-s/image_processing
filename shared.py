import os

HERE = os.path.dirname(os.path.abspath(__file__))

# --- START TYPE DETECTION PATHS ---
TYPE_DETECTION_OUT_VERSION = "1.0.0"
TYPE_DETECTION_DIR = os.path.join(HERE, "type_detection")
TYPE_DETECTION_OUT_DIR = os.path.join(TYPE_DETECTION_DIR, "out")
TYPE_DETECTION_OUT_DATASET_DIR = os.path.join(TYPE_DETECTION_DIR, "fashion-dataset")

TYPE_DETECTION_OUT_CSV_PATH = os.path.join(TYPE_DETECTION_OUT_DATASET_DIR, "styles.csv")
TYPE_DETECTION_OUT_IMAGES_DIR = os.path.join(TYPE_DETECTION_OUT_DATASET_DIR, "images")
TYPE_DETECTION_OUT_OUTPUT_H5_PATH = os.path.join(TYPE_DETECTION_OUT_DIR, TYPE_DETECTION_OUT_VERSION+"-fashion_clothing_classifier.h5")
# --- END TYPE DETECTION PATHS ---
