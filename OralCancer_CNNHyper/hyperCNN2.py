import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras import layers, models, regularizers
import cv2
import csv
import matplotlib
matplotlib.use('Agg')  # Use non-interactive Agg backend
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import pandas as pd

# ============================================================
# Paths
# ============================================================
base_data_path = "C:/Users/mesho/PyCharmMiscProject/MedicalAI/OralCancer_Efficient/dataset"
train_path = os.path.join(base_data_path, "train")
val_path = os.path.join(base_data_path, "val")
test_path = os.path.join(base_data_path, "test")

base_save_path = "C:/Users/mesho/PyCharmMiscProject/MedicalAI/OralCancer_training/models"
os.makedirs(base_save_path, exist_ok=True)

results_csv_path = os.path.join(base_save_path, "results.csv")
if not os.path.exists(results_csv_path):
    with open(results_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_name", "filters", "dense_units", "dropout_rate", "lr", "l2_lambda",
                         "test_acc", "test_prec", "test_rec", "test_auc"])

# ============================================================
# Parameters
# ============================================================
img_size = (128, 128)
batch_size = 16
epochs = 50

# ============================================================
# Preprocessing: Gradient Image Conversion
# ============================================================
def gradient_preprocess(img):
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    grad_rgb = cv2.cvtColor(grad, cv2.COLOR_GRAY2RGB)
    return grad_rgb / 255.0

# ============================================================
# Data Generators
# ============================================================
datagen = ImageDataGenerator(preprocessing_function=gradient_preprocess)

train_gen = datagen.flow_from_directory(
    train_path, target_size=img_size, class_mode="categorical", color_mode="rgb", batch_size=batch_size, shuffle=True
)

val_gen = datagen.flow_from_directory(
    val_path, target_size=img_size, class_mode="categorical", color_mode="rgb", batch_size=batch_size, shuffle=False
)

test_gen = datagen.flow_from_directory(
    test_path, target_size=img_size, class_mode="categorical", color_mode="rgb", batch_size=batch_size, shuffle=False
)

num_classes = train_gen.num_classes
class_labels = list(train_gen.class_indices.keys())

# ============================================================
# Build CNN Model
# ============================================================
def build_cnn(filters=32, dense_units=256, dropout_rate=0.5, lr=1e-3, l2_lambda=0.01):
    regularizer = regularizers.l2(l2_lambda)

    model = models.Sequential([
        layers.Conv2D(filters, (3, 3), activation="relu", padding="same",
                      kernel_regularizer=regularizer, input_shape=(img_size[0], img_size[1], 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),

        layers.Conv2D(filters * 2, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizer),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),

        layers.Conv2D(filters * 4, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizer),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),

        layers.Flatten(),
        layers.Dense(dense_units, activation="relu", kernel_regularizer=regularizer),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation="softmax", kernel_regularizer=regularizer)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )
    return model

# ============================================================
# Grad-CAM Utilities
# ============================================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
    return heatmap.numpy()

def save_gradcam_images(test_gen, model, run_dir, num_images=5):
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break
    gradcam_dir = os.path.join(run_dir, "gradcam_samples")
    os.makedirs(gradcam_dir, exist_ok=True)
    x_batch, _ = next(test_gen)
    for i in range(min(num_images, len(x_batch))):
        img = x_batch[i]
        img_array = np.expand_dims(img, axis=0)
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(np.uint8(img * 255), 0.6, heatmap, 0.4, 0)
        cv2.imwrite(os.path.join(gradcam_dir, f"gradcam_{i}.png"), superimposed_img)

# ============================================================
# ROC Curve Utility
# ============================================================
def save_roc_curves(y_true, y_pred_probs, run_dir, class_labels):
    plt.figure(figsize=(10, 8))
    for i, class_label in enumerate(class_labels):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_label} (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, "roc_curve.png"))
    plt.close()

# ============================================================
# Hyperparameter Search
# ============================================================
param_grid = {
    "filters": [32, 64],
    "dense_units": [128, 256],
    "dropout_rate": [0.3, 0.5],
    "lr": [1e-3, 5e-4],
    "l2_lambda": [0.001, 0.01]
}

for filters in param_grid["filters"]:
    for dense_units in param_grid["dense_units"]:
        for dropout_rate in param_grid["dropout_rate"]:
            for lr in param_grid["lr"]:
                for l2_lambda in param_grid["l2_lambda"]:

                    model = build_cnn(filters, dense_units, dropout_rate, lr, l2_lambda)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    run_name = f"cnn_run_{timestamp}_f{filters}_d{dense_units}_dr{dropout_rate}_lr{lr}_l2{l2_lambda}"
                    run_dir = os.path.join(base_save_path, run_name)
                    os.makedirs(run_dir, exist_ok=True)

                    weights_path = os.path.join(run_dir, "best_weights.h5")
                    model_json_path = os.path.join(run_dir, "model.json")
                    final_model_path = os.path.join(run_dir, "final_model.h5")
                    history_log_path = os.path.join(run_dir, "history.csv")
                    plot_path = os.path.join(run_dir, "training_curves.png")
                    cm_path = os.path.join(run_dir, "confusion_matrix.csv")
                    report_path = os.path.join(run_dir, "classification_report.csv")

                    with open(model_json_path, "w") as json_file:
                        json_file.write(model.to_json())

                    callbacks = [
                        ModelCheckpoint(weights_path, monitor="val_loss", save_best_only=True, save_weights_only=True),
                        CSVLogger(history_log_path)
                    ]

                    print(f"Starting run: {run_name}")
                    history = model.fit(
                        train_gen,
                        validation_data=val_gen,
                        epochs=epochs,
                        callbacks=callbacks,
                        verbose=1
                    )

                    model.save(final_model_path)

                    # Plot training curves
                    plt.figure(figsize=(12, 8))
                    for key in ["loss", "accuracy", "precision", "recall", "auc"]:
                        if key in history.history:
                            plt.plot(history.history[key], label=f"train_{key}")
                        val_key = f"val_{key}"
                        if val_key in history.history:
                            plt.plot(history.history[val_key], label=f"val_{key}")
                    plt.title("Training & Validation Metrics")
                    plt.xlabel("Epochs")
                    plt.ylabel("Value")
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(plot_path)
                    plt.close()

                    # Evaluate on test set
                    y_true = test_gen.classes
                    y_pred_probs = model.predict(test_gen)
                    y_pred = np.argmax(y_pred_probs, axis=1)

                    test_loss, test_acc, test_prec, test_rec, test_auc = model.evaluate(test_gen, verbose=1)
                    print(f"Test results for {run_name}: Acc={test_acc:.4f}, Prec={test_prec:.4f}, Rec={test_rec:.4f}, AUC={test_auc:.4f}")

                    # Confusion matrix + classification report
                    cm = confusion_matrix(y_true, y_pred)
                    pd.DataFrame(cm, index=class_labels, columns=class_labels).to_csv(cm_path)

                    report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
                    pd.DataFrame(report).transpose().to_csv(report_path)

                    # Save ROC curves
                    save_roc_curves(y_true, y_pred_probs, run_dir, class_labels)

                    # Save Grad-CAM samples
                    save_gradcam_images(test_gen, model, run_dir, num_images=5)

                    with open(results_csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([run_name, filters, dense_units, dropout_rate, lr, l2_lambda,
                                         test_acc, test_prec, test_rec, test_auc])