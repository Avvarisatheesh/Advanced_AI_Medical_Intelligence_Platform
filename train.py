import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

# Dataset paths
train_dir = "dataset/train"
val_dir = "dataset/val"
test_dir = "dataset/test"

# Image settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_data = test_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# CNN Model
from tensorflow.keras.layers import Input

inputs = Input(shape=(224, 224, 3))

x = Conv2D(32, (3, 3), activation="relu", name="conv2d")(inputs)
x = MaxPooling2D()(x)

x = Conv2D(64, (3, 3), activation="relu", name="conv2d_1")(x)
x = MaxPooling2D()(x)

x = Conv2D(128, (3, 3), activation="relu", name="conv2d_2")(x)
x = MaxPooling2D()(x)

x = Flatten()(x)

x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)

outputs = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

print("Training Started...")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=[early_stop]
)

os.makedirs("model", exist_ok=True)

model.save("model/xray_model.keras")

loss, accuracy = model.evaluate(test_data)

print(f"Test Accuracy : {accuracy*100:.2f}%")

print("Model Saved Successfully.")

