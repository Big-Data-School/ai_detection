import os
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import (
    CSVLogger,
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)

from models.cnn import mini_XCEPTION
from utils.datasets import DataManager, split_data
from utils.preprocessor import preprocess_input

# ────────────────────────────────────────────────
# Accélérations Apple Silicon
# ────────────────────────────────────────────────
mixed_precision.set_global_policy("mixed_float16")        # float16 sur GPU Metal

# Batch-size auto-scalé selon la VRAM dispo
if tf.config.list_physical_devices("GPU"):
    gmem = tf.config.experimental.get_memory_info("GPU:0")["current"]
else:
    gmem = 0
batch_size = 128 if gmem < 6e9 else 256

# ────────────────────────────────────────────────
# Hyper-paramètres généraux
# ────────────────────────────────────────────────
num_epochs        = 10_000
input_shape       = (64, 64, 1)
validation_split  = 0.2
verbose           = 1
num_classes       = 7
patience          = 50
base_path         = os.getcwd() + "/trained_models/emotion_models/"

# ────────────────────────────────────────────────
# Construction / compilation du modèle
# ────────────────────────────────────────────────
model = mini_XCEPTION(input_shape, num_classes)
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,                     # XLA sur Metal
)
model.summary()

# ────────────────────────────────────────────────
# Boucle sur les jeux de données
# ────────────────────────────────────────────────
datasets = ["fer2013"]
for dataset_name in datasets:
    print(f"Training dataset : {dataset_name}")

    # Callbacks
    log_file_path        = base_path + dataset_name + "_emotion_training.log"
    csv_logger           = CSVLogger(log_file_path, append=False)
    early_stop           = EarlyStopping("val_loss", patience=patience)
    reduce_lr            = ReduceLROnPlateau("val_loss", factor=0.1,
                                             patience=patience // 4, verbose=1)
    trained_models_path  = base_path + dataset_name + "_mini_XCEPTION"
    model_names          = trained_models_path + ".{epoch:02d}-{val_accuracy:.2f}.keras"
    model_checkpoint     = ModelCheckpoint(model_names, monitor="val_loss",
                                           verbose=1, save_best_only=True)
    callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

    # Chargement et pré-traitement des données
    data_loader = DataManager(dataset_name, image_size=input_shape[:2])
    faces, emotions = data_loader.get_data()
    faces = preprocess_input(faces)

    (train_faces, train_emotions), (val_faces, val_emotions) = split_data(
        faces, emotions, validation_split
    )

    # ────────────────────────────────────────────
    # Pipelines tf.data (CPU ↔ GPU asynchrones)
    # ────────────────────────────────────────────
    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_faces, train_emotions))
        .shuffle(10_000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((val_faces, val_emotions))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # ────────────────────────────────────────────
    # Entraînement
    # ────────────────────────────────────────────
    model.fit(
        train_ds,
        epochs=num_epochs,
        callbacks=callbacks,
        validation_data=val_ds,
        verbose=verbose,
    )
