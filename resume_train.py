import tensorflow as tf

MODEL_PATH = "models/plant_disease_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(224,224),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(224,224),
    batch_size=32
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3
)

model.save(MODEL_PATH)

print("Saved successfully")