import pandas as pd
import numpy as np
import os
import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from shared import OUT_DIR, BASE_DIR, CSV_PATH, IMAGES_DIR, OUTPUT_H5_PATH

if not os.path.exists(BASE_DIR):
    raise FileNotFoundError(f"Dataset directory '{BASE_DIR}' not found. Please download and extract the dataset using \"kaggle datasets download -d paramaggarwal/fashion-product-images-small\" and \"unzip fashion-product-images-small.zip\".")

def train_model_from_dataset():
  if os.path.exists(OUTPUT_H5_PATH):
    return

  # 1. Load dataset metadata
  df = pd.read_csv(CSV_PATH, on_bad_lines='skip', engine='python')
  df['image'] = df['id'].astype(str) + ".jpg"

  # 2. Define custom clothing categories
  CATEGORY_MAP = {
    'pants': [
      'Jeans', 'Track Pants', 'Trousers', 'Shorts', 'Capris', 'Leggings', 'Jeggings',
      'Lounge Pants', 'Lounge Shorts', 'Churidar', 'Salwar', 'Salwar and Dupatta',
      'Patiala', 'Rain Trousers', 'Capris', 'Trunk', 'Boxers'
    ],

    'sweater': [
      'Sweaters', 'Sweatshirts'
    ],

    'dress': [
      'Dresses', 'Jumpsuit', 'Rompers', 'Nightdress', 'Night suits', 'Baby Dolls'
    ],

    'jacket': [
      'Jackets', 'Blazers', 'Rain Jacket', 'Waistcoat', 'Nehru Jackets', 'Suits', 'Shrug'
    ],

    'shirt': [
      'Shirts', 'Tshirts', 'Tops', 'Kurtas', 'Kurta Sets', 'Kurtis', 'Tunics',
      'Camisoles', 'Lounge Tshirts', 'Shapewear', 'Innerwear Vests', 'Bra'
    ],

    'shoes': [
      'Casual Shoes', 'Formal Shoes', 'Sports Shoes', 'Sandals', 'Heels', 'Flats', 'Sneakers',
      'Shoes', 'Booties', 'Shoes Accessories', 'Sports Sandals', 'Flip Flops', 'Socks', 'Shoe Laces'
    ],

    'accessory': [
      # Fashion wear
      'Handbags', 'Wallets', 'Belts', 'Clutches', 'Backpacks', 'Duffel Bag', 'Messenger Bag',
      'Laptop Bag', 'Trolley Bag', 'Mobile Pouch', 'Waist Pouch', 'Rucksacks',

      # Jewelry
      'Watches', 'Bracelet', 'Ring', 'Pendant', 'Necklace and Chains', 'Earrings',
      'Jewellery Set', 'Bangle', 'Ties', 'Cufflinks', 'Ties and Cufflinks',

      # Head/Face accessories
      'Sunglasses', 'Caps', 'Hats', 'Headband', 'Scarves', 'Mufflers', 'Stoles', 'Hair Accessory',
      'Hair Colour', 'Gloves', 'Mask and Peel', 'Key chain',
    ],
  }

  def map_category(article_type):
      for target, keywords in CATEGORY_MAP.items():
          if any(keyword.lower() in str(article_type).lower() for keyword in keywords):
              return target
      return None  # ignore irrelevant items

  df['label'] = df['articleType'].apply(map_category)
  df = df.dropna(subset=['label'])   # keep only mapped images

  # 3. Split data
  train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

  # 4. Data pipeline with augmentation
  BATCH_SIZE = 32
  IMG_SIZE = (224, 224)

  datagen = ImageDataGenerator(
      rescale=1./255,
      validation_split=0.2
  )

  train_gen = datagen.flow_from_dataframe(
      train_df,
      directory=IMAGES_DIR,
      x_col='image',
      y_col='label',
      target_size=IMG_SIZE,
      batch_size=BATCH_SIZE,
      class_mode='categorical',
      subset='training'
  )

  val_gen = datagen.flow_from_dataframe(
      train_df,
      directory=IMAGES_DIR,
      x_col='image',
      y_col='label',
      target_size=IMG_SIZE,
      batch_size=BATCH_SIZE,
      class_mode='categorical',
      subset='validation'
  )

  # 5. Build model with Transfer Learning (MobileNetV2)
  base_model = tf.keras.applications.MobileNetV2(
      input_shape=(*IMG_SIZE, 3),
      include_top=False,
      weights='imagenet'
  )
  base_model.trainable = False

  model = tf.keras.Sequential([
      base_model,
      tf.keras.layers.GlobalAveragePooling2D(),
      tf.keras.layers.Dense(256, activation='relu'),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(len(CATEGORY_MAP), activation='softmax')
  ])

  model.compile(
      optimizer='adam',
      loss='categorical_crossentropy',
      metrics=['accuracy']
  )

  # 6. Train model
  callbacks = [
      ModelCheckpoint(OUTPUT_H5_PATH, save_best_only=True, monitor="val_loss"),
      EarlyStopping(patience=6, restore_best_weights=True, monitor="val_loss"),
      ReduceLROnPlateau(factor=0.5, patience=3, monitor="val_loss"),
  ]
  history = model.fit(
      train_gen,
      validation_data=val_gen,
      epochs=50,
      callbacks=callbacks,
      workers=6,
      use_multiprocessing=True,
  )

  # 7. Save model
  model.save(OUTPUT_H5_PATH)
  print("✅ Model saved as h5")

def load_model_from_file():
  model = tf.keras.models.load_model(OUTPUT_H5_PATH)
  print("✅ Model loaded from h5")
  return model

def convert_model_to_tfjs(model):
  tfjs_target_dir = os.path.join(OUT_DIR, "clothing_classification_model")
  tfjs.converters.save_keras_model(model, tfjs_target_dir)
  print("✅ Model converted to TensorFlow.js format")

if __name__ == "__main__":
  train_model_from_dataset()
  model = load_model_from_file()
  convert_model_to_tfjs(model)
