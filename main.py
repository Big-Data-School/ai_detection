# ''' Main file for the project. '''
# ────────────────────────────────────────────────
# TESTING for GPU on Apple Silicon
# ────────────────────────────────────────────────
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
from tensorflow.keras import mixed_precision
print(mixed_precision.global_policy())

# ────────────────────────────────────────────────
# ────────────────────────────────────────────────

