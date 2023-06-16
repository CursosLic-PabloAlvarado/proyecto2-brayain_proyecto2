
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)
tf.config.list_physical_devices('GPU')
#tf.test.is_gpu_available()
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

