import tensorflow as tf
import tensorflow_hub as hub

print("Loading model...")
loaded_model = tf.keras.models.load_model(
    "cat_dog_classifier_model.keras",
    custom_objects={'KerasLayer': hub.KerasLayer},
    safe_mode=False
)
print("Loaded successfully!")
