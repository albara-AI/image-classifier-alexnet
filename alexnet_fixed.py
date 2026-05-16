import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization, Activation
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# FIX 1: Eager mode — keep for debugging,
#         remove in production for speed
# ─────────────────────────────────────────
tf.config.run_functions_eagerly(True)

# ─────────────────────────────────────────
# CONFIG — change paths here only
# ─────────────────────────────────────────
DATA_PATH   = r'YOUR_DATASET_PATH_HERE'   # ← change this
IMAGE_SIZE  = (227, 227)                  # AlexNet standard
BATCH_SIZE  = 32
EPOCHS      = 25
LEARNING_RATE = 1e-4                      # FIX 3: lowered from 0.001 → 0.0001

# ─────────────────────────────────────────
# Data generators
# ─────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]    # FIX 2: added missing comma in original
)

train_generator = train_datagen.flow_from_directory(
    DATA_PATH,
    target_size=IMAGE_SIZE,        # FIX 1: was (224,224), now matches AlexNet (227,227)
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATA_PATH,
    target_size=IMAGE_SIZE,        # FIX 1: same fix here
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

num_classes = train_generator.num_classes
print(f"✅ Classes found: {num_classes}")
print(f"✅ Training samples: {train_generator.samples}")
print(f"✅ Validation samples: {val_generator.samples}")

# ─────────────────────────────────────────
# AlexNet architecture
# ─────────────────────────────────────────
model = Sequential([
    # Block 1
    Conv2D(96, (11, 11), strides=(4, 4), padding="valid",
           input_shape=(227, 227, 3)),
    Activation("relu"),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    BatchNormalization(),

    # Block 2
    Conv2D(256, (5, 5), padding="same"),
    Activation("relu"),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    BatchNormalization(),

    # Block 3
    Conv2D(384, (3, 3), padding="same"),
    Activation("relu"),
    BatchNormalization(),

    # Block 4
    Conv2D(384, (3, 3), padding="same"),
    Activation("relu"),
    BatchNormalization(),

    # Block 5
    Conv2D(256, (3, 3), padding="same"),
    Activation("relu"),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    BatchNormalization(),

    # Fully Connected
    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.5),                  # FIX 4: 0.5 is more standard than 0.6
    BatchNormalization(),

    Dense(4096, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),

    Dense(num_classes, activation='softmax')
])

model.summary()

# ─────────────────────────────────────────
# Compile
# ─────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ─────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7,               # FIX 5: added min_lr floor
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(   # FIX 6: save best epoch only
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ─────────────────────────────────────────
# Train
# ─────────────────────────────────────────
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks
)

# ─────────────────────────────────────────
# Save
# ─────────────────────────────────────────
model.save("alexnet_final.keras")
print("✅ Model saved.")

# ─────────────────────────────────────────
# Plot
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(history.history['accuracy'],     label='Train Accuracy',  marker='o')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy',    marker='x')
axes[0].set_title('Accuracy Curve')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['loss'],     label='Train Loss',  marker='o')
axes[1].plot(history.history['val_loss'], label='Val Loss',    marker='x')
axes[1].set_title('Loss Curve')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("training_metrics_plot.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot saved.")
