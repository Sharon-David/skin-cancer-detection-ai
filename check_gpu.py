import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print("GPUs available:", gpus)
if gpus:
    print("GPU is working!")
else:
    print("No GPU detected.")