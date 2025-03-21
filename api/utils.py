import tensorflow as tf
import numpy as np
from google.cloud import storage

def prepare_image(image, height, width):
    """
    Prepare image datasets from directories

    Args:
        Image

    Returns:
        Numpy Array
    """

    # Convert the image to RGB
    image = image.convert("RGB")

    # Resize the image
    image = image.resize((width, height))

    # Convert to NumPy array and normalize
    image_array = np.array(image) / 255.0

    # Expand dimensions (1, height, width, channels)
    image_array = np.expand_dims(image_array, axis=0)


    return image_array

def load_model_from_gcp():
    # Download model on startup
    MODEL_BUCKET = "eyesense_model"
    MODEL_PATH = "Xception-01.keras"
    LOCAL_MODEL_PATH = "/tmp/model.keras"
    
    # Download the model from GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(MODEL_BUCKET)
    blob = bucket.blob(MODEL_PATH)
    blob.download_to_filename(LOCAL_MODEL_PATH)
    
    # Load the model
    global model
    model = tf.keras.models.load_model(LOCAL_MODEL_PATH)
    return model