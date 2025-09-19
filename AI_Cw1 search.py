Search---- 1





"""
weather_classification.py

This script performs exploratory data analysis, preprocessing, feature extraction,
and trains and evaluates three machine learning models (Random Forest, SVM,
and a CNN using TensorFlow) on the Multi-class Weather Dataset for Image Classification.

Dataset: https://data.mendeley.com/datasets/4drtyfjtfy/1
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load dataset
def load_data(data_dir, image_size=(128,128)):
    X = []
    y = []
    class_names = sorted(os.listdir(data_dir))
    label_map = {name: idx for idx, name in enumerate(class_names)}
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for fname in os.listdir(class_dir):
            fpath = os.path.join(class_dir, fname)
            try:
                img = Image.open(fpath).convert('RGB')
                img = img.resize(image_size)
                X.append(np.array(img))
                y.append(label_map[class_name])
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
    X = np.array(X)
    y = np.array(y)
    return X, y, class_names

# 2. Exploratory Data Analysis: class distribution and sample images
def plot_class_distribution(y, class_names):
    counts = np.bincount(y)
    plt.figure(figsize=(8,5))
    plt.bar(class_names, counts)
    plt.xlabel('Class')
    plt.ylabel('Number of images')
    plt.title('Class Distribution')
    plt.show()

def show_sample_images(X, y, class_names, samples_per_class=5):
    plt.figure(figsize=(samples_per_class*2, len(class_names)*2))
    idx = 1
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        selected = np.random.choice(cls_idx, samples_per_class, replace=False)
        for i, img_idx in enumerate(selected):
            plt.subplot(len(class_names), samples_per_class, idx)
            plt.imshow(X[img_idx])
            plt.axis('off')
            if i == samples_per_class // 2:
                plt.title(class_names[cls])
            idx += 1
    plt.tight_layout()
    plt.show()

# 3. Preprocessing and Feature Extraction for classical ML models
def extract_color_histograms(X, bins=(8,8,8)):
    """
    Extracts 3D color histograms from images using OpenCV.
    Reference: https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html
    """
    import cv2
    features = []
    for img in X:
        hist = cv2.calcHist([img], [0,1,2], None, bins, [0,256,0,256,0,256])
        hist = cv2.normalize(hist, hist).flatten()
        features.append(hist)
    return np.array(features)

# 4. Train and evaluate classical models
def train_evaluate_classical(X_feat, y, model, model_name, class_names):
    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"--- {model_name} ---")
    print(classification_report(y_test, y_pred, target_names=class_names))
    cm = confusion_matrix(y_test, y_pred)
    import seaborn as sns
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# 5. Build and train CNN model
def build_cnn(input_shape, num_classes):
    model = models.Sequential([  # TensorFlow Keras CNN architecture
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_evaluate_cnn(X, y, class_names, epochs=20, batch_size=32):
    X_train, X_test, y_train, y_test = train_test_split(
        X/255.0, y, test_size=0.2, random_state=42, stratify=y
    )
    input_shape = X_train.shape[1:]
    num_classes = len(class_names)
    model = build_cnn(input_shape, num_classes)
    history = model.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size,
        validation_split=0.1
    )
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"CNN Test accuracy: {test_acc:.4f}")
    # Plot training history
    plt.figure(figsize=(8,4))
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('CNN Training History')
    plt.show()
    # Confusion matrix
    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    import seaborn as sns
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.title("CNN Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

if __name__ == "__main__":
    import cv2
    import seaborn as sns

    data_dir = "path_to_dataset"  # replace with the path to unzipped dataset
    X, y, class_names = load_data(data_dir)
    plot_class_distribution(y, class_names)
    show_sample_images(X, y, class_names)

    # Feature extraction
    X_feat = extract_color_histograms(X)
    # Model 1: Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    train_evaluate_classical(X_feat, y, rf, "Random Forest", class_names)

    # Model 2: SVM
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    train_evaluate_classical(X_feat, y, svm, "SVM", class_names)

    # Model 3: CNN
    train_evaluate_cnn(X, y, class_names, epochs=20, batch_size=32)








Search - 2:




Skip to content
You said:
Consider yourself a data science student working on an python assignment, here is the assignment description, dataset and requirements for the assignment. i want you to provide the customised code without plagarism "You are required to analyse the given dataset, translate the “science question” into machine learning problem (eg classification) you want to solve with the dataset, build, evaluate and compare three different machine learning models that would address the problem, and draw conclusions and recommendations based on your findings. It is recommended not to use a very simple dataset (preferably of higher complexity than the ones used in lab sessions) as it would give you limited room for presenting the explanatory data analysis, preprocessing, results, and evaluation. One of the three models must be based on a deep learning architecture implemented using the TensorFlow/PyTorch in Python. The submission should include your report, dataset(s) and Python scripts with comments, all included in one zip-file. Your work should be original and produced by you. Copying whole tutorials, scripts or images from other sources is not allowed. Any material you borrow from other sources to build upon should be clearly referenced (use comments to reference in Python scripts); otherwise, it will be treated as plagiarism, which may lead to investigation and subsequent action. The dataset to be analysed can be accessed using the hyperlink https://data.mendeley.com/datasets/4drtyfjtfy/1

and Below are the requirements in which the assignment has to be completed:
1.	Exploratory data analysis (dataset description and visualisation, support with relevant and important figures)
2.	Data pre-processing and feature selection
3.	code 3 Deep learning models (iterate for each of the three models)
•	Summary of the approach (justify why this ML algorithm is suitable for your dataset, support with references to literature)
•	Model training and evaluation
•	Results and discussion (support with relevant and important tables/figures)
4.	Performance measures and evaluation strategies"  

ChatGPT said:

Thought for 19 seconds
Below is a single Python script (weather_classification.py) that walks through the full assignment workflow: EDA, preprocessing & feature‐engineering, three different models (two “classical” and one deep‐learning), evaluation and comparison. All sections are commented with brief justifications and citations where appropriate. You’ll need to download and unzip the Multi‐class Weather Dataset into a folder (e.g. ./data/images/) so that you have subfolders like cloudy/, rainy/, shine/, sunrise/. Adjust DATA_DIR as needed.

# weather_classification.py

import os
import random
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from collections import Counter

# for classical ML
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.feature import hog  # Dalal & Triggs (2005)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

# for deep learning
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# 1. Configuration
DATA_DIR = './data/images'   # path to the folder containing the 4 class subfolders
IMG_SIZE = (128, 128)        # resize all images to 128×128
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# 2. Exploratory Data Analysis
def plot_class_distribution(data_dir):
    classes = os.listdir(data_dir)
    counts = {cls: len(os.listdir(os.path.join(data_dir, cls))) for cls in classes}
    plt.figure(figsize=(6,4))
    plt.bar(counts.keys(), counts.values(), color='skyblue')
    plt.title('Image Count per Class')
    plt.ylabel('Number of Images')
    plt.show()

def show_sample_images(data_dir, n_per_class=4):
    classes = os.listdir(data_dir)
    fig, axes = plt.subplots(len(classes), n_per_class, figsize=(n_per_class*2, len(classes)*2))
    for i, cls in enumerate(classes):
        imgs = glob(os.path.join(data_dir, cls, '*'))    
        samples = random.sample(imgs, n_per_class)
        for j, img_path in enumerate(samples):
            img = imread(img_path)
            axes[i,j].imshow(img)
            axes[i,j].axis('off')
            if j==0: axes[i,j].set_ylabel(cls)
    plt.tight_layout()
    plt.show()

plot_class_distribution(DATA_DIR)
show_sample_images(DATA_DIR)

# 3. Load data & train/test split
def load_image_paths_and_labels(data_dir):
    image_paths, labels = [], []
    for cls in os.listdir(data_dir):
        cls_dir = os.path.join(data_dir, cls)
        for img_file in glob(os.path.join(cls_dir, '*')):
            image_paths.append(img_file)
            labels.append(cls)
    return image_paths, labels

paths, labels = load_image_paths_and_labels(DATA_DIR)
X_train_paths, X_test_paths, y_train, y_test = train_test_split(
    paths, labels, test_size=0.2, stratify=labels, random_state=RANDOM_SEED)

# further split train into train/val for deep learning
X_train_paths, X_val_paths, y_train, y_val = train_test_split(
    X_train_paths, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_SEED)

# 4. Preprocessing & Feature Extraction for classical models
def extract_hog_features(image_paths):
    feats = []
    for p in image_paths:
        img = imread(p)
        img_resized = resize(img, IMG_SIZE)
        gray = rgb2gray(img_resized)
        # HOG parameters can be tuned; here defaults (orientations=9, pixels_per_cell=(8,8))
        hog_feat = hog(gray, block_norm='L2-Hys')  # Dalal & Triggs (2005)
        feats.append(hog_feat)
    return np.array(feats)

def extract_color_histograms(image_paths, bins=(8,8,8)):
    feats = []
    for p in image_paths:
        img = imread(p)
        img_resized = resize(img, IMG_SIZE)
        # 3D color histogram in HSV space
        hsv = plt.colors.rgb_to_hsv(img_resized)
        hist, _ = np.histogramdd(
            hsv.reshape(-1,3),
            bins=bins,
            range=[(0,1),(0,1),(0,1)]
        )
        feats.append(hist.flatten())
    return np.array(feats)

print("Extracting HOG features...")
X_train_hog = extract_hog_features(X_train_paths)
X_test_hog  = extract_hog_features(X_test_paths)

print("Extracting color histograms...")
X_train_ch = extract_color_histograms(X_train_paths)
X_test_ch  = extract_color_histograms(X_test_paths)

# 5. Model 1: SVM on HOG features
svm = SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED)
svm_params = {'C':[1,10], 'gamma':['scale','auto']}
svm_grid = GridSearchCV(svm, svm_params, cv=3, n_jobs=-1)
svm_grid.fit(X_train_hog, y_train)
y_pred_svm = svm_grid.predict(X_test_hog)

print("SVM Best Params:", svm_grid.best_params_)
print(classification_report(y_test, y_pred_svm))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))

# 6. Model 2: Random Forest on Color Histograms
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
rf.fit(X_train_ch, y_train)
y_pred_rf = rf.predict(X_test_ch)

print("Random Forest Report:")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# 7. Model 3: Deep CNN with TensorFlow/Keras
def preprocess_for_cnn(image_paths, labels, batch_size=32, shuffle=True):
    # Keras image generator from file paths
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = img / 255.0
        return img, label
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle: ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = preprocess_for_cnn(X_train_paths, y_train)
val_ds   = preprocess_for_cnn(X_val_paths, y_val, shuffle=False)
test_ds  = preprocess_for_cnn(X_test_paths, y_test, shuffle=False)

# a simple CNN architecture
def build_cnn(input_shape=(128,128,3), num_classes=4):
    model = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])
    # categorical crossentropy since labels are strings, will map to ints via .map
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# map labels to integers
label_to_idx = {cls:i for i,cls in enumerate(sorted(set(labels)))}
y_train_int = np.array([label_to_idx[l] for l in y_train])
y_val_int   = np.array([label_to_idx[l] for l in y_val])
y_test_int  = np.array([label_to_idx[l] for l in y_test])

# rebuild tf datasets with integer labels
train_ds = tf.data.Dataset.from_tensor_slices((X_train_paths, y_train_int)).map(
    lambda p,l: (tf.image.resize(tf.image.decode_png(tf.io.read_file(p),3), IMG_SIZE)/255.0, l)
).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((X_val_paths, y_val_int)).map(
    lambda p,l: (tf.image.resize(tf.image.decode_png(tf.io.read_file(p),3), IMG_SIZE)/255.0, l)
).batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((X_test_paths, y_test_int)).map(
    lambda p,l: (tf.image.resize(tf.image.decode_png(tf.io.read_file(p),3), IMG_SIZE)/255.0, l)
).batch(32).prefetch(tf.data.AUTOTUNE)

cnn = build_cnn()
early_stop = callbacks.EarlyStopping(patience=5, restore_best_weights=True)
history = cnn.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=[early_stop])

# evaluate on test set
test_loss, test_acc = cnn.evaluate(test_ds)
print(f"CNN Test Accuracy: {test_acc:.4f}")

# 8. Compare all three models
print("\n=== Model Comparison ===")
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"CNN Accuracy: {test_acc:.4f}")

# Optionally, plot training history
plt.figure()
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('CNN Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
Notes & References

Problem framing: this is a four‐class image‐classification task.

HOG features based on Dalal & Triggs (2005), “Histograms of Oriented Gradients for Human Detection”.

Color histograms provide simple global color descriptors (e.g. Swain & Ballard, 1991).

Models:

SVM with RBF kernel (good baseline for medium‐dimensional features).

Random Forest on 3D‐color histograms (robust, non‐parametric).

CNN in TensorFlow/Keras (end‐to‐end deep‐learning).

Adjust hyperparameters and do further visualizations / confusion‐matrix plots in your report. Good luck!
