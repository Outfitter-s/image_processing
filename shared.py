import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")
BASE_DIR = os.path.join(OUT_DIR, "fashion_product_images")

CSV_PATH = os.path.join(BASE_DIR, "styles.csv")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
OUTPUT_H5_PATH = os.path.join(OUT_DIR, "fashion_clothing_classifier.h5")
