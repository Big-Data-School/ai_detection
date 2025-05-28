import os
import tensorflow as tf
from tensorflow.keras import mixed_precision  # Changement ici
from tensorflow.keras.callbacks import (      # Changement ici
    CSVLogger,
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)
import logging

from models.cnn import mini_XCEPTION
from utils.datasets import DataManager, split_data
from utils.preprocessor import preprocess_input

# ────────────────────────────────────────────────
# Callback personnalisé pour logguer l'arrêt anticipé
# ────────────────────────────────────────────────
class LoggingEarlyStopping(EarlyStopping):
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            logging.info(f"Early stopping déclenché à l'époque {self.stopped_epoch} : {self.monitor} n'a pas progressé pendant {self.patience} époques.")
        else:
            logging.info("Entraînement terminé sans early stopping.")

# ────────────────────────────────────────────────
# Initialisation du logging
# ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("training_debug.log", mode="w"),
        logging.StreamHandler()
    ]
)
logging.info("Démarrage du script d'entraînement.")

# ────────────────────────────────────────────────
# Accélérations Apple Silicon
# ────────────────────────────────────────────────
mixed_precision.set_global_policy("mixed_float16")        # float16 sur GPU Metal
logging.info(f"Politique de précision mixte : {mixed_precision.global_policy()}")

# Détection GPU Apple Silicon
physical_gpus = tf.config.list_physical_devices("GPU")
if physical_gpus:
    logging.info(f"GPU détecté : {physical_gpus}")
    try:
        # Pour TensorFlow 2.17+, utiliser get_memory_usage au lieu de get_memory_info
        gmem = tf.config.experimental.get_memory_usage("GPU:0")
    except:
        # Fallback si la fonction n'existe pas
        gmem = 6e9  # Valeur par défaut
else:
    logging.warning("Aucun GPU détecté, entraînement sur CPU !")
    gmem = 0
batch_size = 128 if gmem < 6e9 else 256
logging.info(f"Batch size utilisé : {batch_size}")

# ────────────────────────────────────────────────
# Hyper-paramètres généraux
# ────────────────────────────────────────────────
num_epochs        = 100
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
logging.info("Modèle compilé avec jit_compile=True.")

# ────────────────────────────────────────────────
# Boucle sur les jeux de données
# ────────────────────────────────────────────────
datasets = ["fer2013"]
for dataset_name in datasets:
    logging.info(f"Début d'entraînement sur le dataset : {dataset_name}")
    print(f"Training dataset : {dataset_name}")

    # Callbacks
    log_file_path        = base_path + dataset_name + "_emotion_training.log"
    csv_logger           = CSVLogger(log_file_path, append=False)
    early_stop           = LoggingEarlyStopping(monitor="val_loss", patience=patience)
    reduce_lr            = ReduceLROnPlateau("val_loss", factor=0.1,
                                             patience=patience // 4, verbose=1)
    trained_models_path  = base_path + dataset_name + "_mini_XCEPTION"
    model_names          = trained_models_path + ".{epoch:02d}-{val_accuracy:.2f}.keras"
    model_checkpoint     = ModelCheckpoint(model_names, monitor="val_loss",
                                           verbose=1, save_best_only=True)
    callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

    # Chargement et pré-traitement des données
    logging.info("Chargement et prétraitement des données...")
    data_loader = DataManager(dataset_name, image_size=input_shape[:2])
    faces, emotions = data_loader.get_data()
    faces = preprocess_input(faces)
    logging.info(f"Taille des données : faces={faces.shape}, emotions={emotions.shape}")

    (train_faces, train_emotions), (val_faces, val_emotions) = split_data(
        faces, emotions, validation_split
    )
    logging.info(f"Split : train={train_faces.shape[0]}, val={val_faces.shape[0]}")

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
    logging.info("Début de l'entraînement...")
    try:
        model.fit(
            train_ds,
            epochs=num_epochs,
            callbacks=callbacks,
            validation_data=val_ds,
            verbose=verbose,
        )
        logging.info("Entraînement terminé avec succès.")
    except Exception as e:
        logging.error(f"Erreur lors de l'entraînement : {e}")
        raise