# 🧠 CT Scan Image Classification using CNN

This project implements a **Convolutional Neural Network (CNN)** to classify CT scan images into two categories. It involves data preprocessing, model building, training with callbacks, performance visualization, and evaluation using classification metrics.

---

## 📚 Table of Contents

1. [Introduction](#-introduction)
   - 1.1 Brief explanation of the project
   - 1.2 Objective of the CNN model
   - 1.3 Overview of the dataset
2. [Importing Necessary Libraries](#-importing-necessary-libraries)
3. [Dataset Setup](#-dataset-setup)
   - 3.1 Paths to dataset directories
   - 3.2 Folder structure explanation
4. [Visualizing the Dataset](#-visualizing-the-dataset)
   - 4.1 Class distribution
   - 4.2 Sample image previews
5. [Data Preprocessing and Augmentation](#-data-preprocessing-and-augmentation)
6. [Building the CNN Model](#-building-the-cnn-model)
7. [Callbacks for Training](#-callbacks-for-training)
8. [Training the Model](#-training-the-model)
9. [Visualizing Training Performance](#-visualizing-training-performance)
10. [Model Evaluation](#-model-evaluation)
11. [Conclusion](#-conclusion)

---

## 🔍 Introduction

### 1.1 Brief Explanation

This project builds a CNN to classify CT scan images, typically distinguishing between categories such as 'Normal' and 'Stone'.

### 1.2 Objective

To develop a deep learning model capable of accurately classifying medical images using supervised learning.

### 1.3 Dataset Overview

The images are organized into folders under `Train` and `Test` directories representing two categories.

---

## 📦 Importing Necessary Libraries

```python
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
warnings.filterwarnings("ignore")
```

---

## 📁 Dataset Setup

### 3.1 Directory Paths

```python
train_dir = './CT_Images/Train'
validation_dir = './CT_Images/Test'
```

### 3.2 Folder Structure

```
CT_Images/
    Train/
        Normal/
        Stone/
    Test/
        Normal/
        Stone/
```

---

## 🖼️ Visualizing the Dataset

### 4.1 Class Distribution

```python
plot_class_distribution(train_dir, "Training Data Class Distribution")
plot_class_distribution(validation_dir, "Validation Data Class Distribution")
```

### 4.2 Sample Images

```python
show_sample_images(train_dir)
```

---

## ⚙️ Data Preprocessing and Augmentation

```python
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=32, class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir, target_size=(150, 150), batch_size=32, class_mode='binary'
)
```

---

## 🧱 Building the CNN Model

```python
model = Sequential([
    Input(shape=(150, 150, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

---

## ⏳ Callbacks for Training

```python
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
]
```

---

## 🚀 Training the Model

```python
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    steps_per_epoch=len(train_generator),
    validation_steps=len(validation_generator),
    callbacks=callbacks
)
```

---

## 📉 Visualizing Training Performance

```python
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()
```

---

## 🧪 Model Evaluation

```python
Y_pred = model.predict(validation_generator)
Y_pred = (Y_pred > 0.5).astype(int)
Y_true = validation_generator.classes

cm = confusion_matrix(Y_true, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Stone'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
```

### Metrics:

```python
print("Precision:", precision_score(Y_true, Y_pred))
print("Recall:", recall_score(Y_true, Y_pred))
print("F1-Score:", f1_score(Y_true, Y_pred))
```

---

## ✅ Conclusion

- Model achieved high training accuracy and generalization on validation set.
- The model can effectively distinguish between two CT scan image classes.
- Evaluation metrics confirm reliable classification performance.

---

> ⚠️ This project is intended for educational purposes. Accuracy in real medical applications must be verified by professionals.
