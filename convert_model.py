import keras
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Input, Flatten, Dropout, Dense
import os

# --- This architecture is based on the code you provided ---

IMAGE_SIZE = 128

# IMPORTANT: Set the number of output classes.
# Your code used len(os.listdir(train_dr)).
# For the brain tumor dataset, this is typically 4 (glioma, meningioma, pituitary, no_tumor).
# Please verify and change this number if it's different for you.
NUM_CLASSES = 4

# 1. Re-create the base model exactly as before
base_model = VGG16(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet')

# 2. Set the layer trainability status exactly as before
for layer in base_model.layers:
  layer.trainable = False

base_model.layers[-2].trainable = True
base_model.layers[-3].trainable = True
base_model.layers[-4].trainable = True

# 3. Re-create the final model exactly as before
model = Sequential([
    Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    base_model,
    Flatten(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(NUM_CLASSES, activation='softmax')
])

print("✅ Model architecture created successfully.")

# 4. Load the weights from the old file
print("⌛ Loading weights from best_model.h5...")
model.load_weights('models/best_model.h5')
print("✅ Weights loaded successfully.")

# 5. Save the complete, fixed model in the new .keras format
print("💾 Saving new model to best_model.keras...")
model.save('models/best_model.keras')
print("✅ Conversion complete! Your new model is 'models/best_model.keras'")