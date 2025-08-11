from PIL import Image
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dropout, Dense
from tensorflow.keras.models import Sequential
import sys
import matplotlib.pyplot as plt

# --- Configuration ---
MODEL_WEIGHTS_PATH = 'brain_tumor_model_weights_complete.weights.h5'
IMG_SIZE = (128, 128)

# --- 1. Recreating Your EXACT Model Architecture ---
print("--- 1. Recreating model architecture precisely as trained ---")

base_model = VGG16(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
for layer in base_model.layers:
    layer.trainable = False
base_model.layers[-2].trainable = True
base_model.layers[-3].trainable = True
base_model.layers[-4].trainable = True

# built the model and giving each important layer a unique name
model = Sequential([
    base_model,
    Flatten(name="flatten"),
    Dropout(0.3, name="dropout_1"),
    Dense(128, activation='relu', name="dense_1"),
    Dropout(0.2, name="dropout_2"),
    Dense(4, activation='softmax', name="dense_2_predictions")
], name="brain_tumor_classifier")
print("✅ Model architecture created successfully.")

# 2. Load the Weights
print("\n--- 2. Loading trained weights ---")
model.load_weights(MODEL_WEIGHTS_PATH)
print(f"✅ Weights loaded successfully from {MODEL_WEIGHTS_PATH}")

# 3. Define Helper Functions
print("\n--- 3. Defining helper functions ---")


def preprocess_image(img_path):
    """Loads and preprocesses an image for the model."""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# This is the robust Grad-CAM function
def get_gradcam_heatmap(img_array, full_model):
    """
    Computes Grad-CAM by manually performing the forward pass,
    avoiding the creation of new, buggy models.
    """
    # We get handles to the layers we need from the working model
    vgg16_base = full_model.get_layer('vgg16')
    last_conv_layer = vgg16_base.get_layer('block4_conv3')

    # We create a temporary model ONLY to define the classifier path
    # This is a list of all layers AFTER the VGG16 base
    classifier_layers = full_model.layers[1:]

    with tf.GradientTape() as tape:
        # 1. Get the feature maps from the base model
        feature_maps = vgg16_base(img_array)
        tape.watch(feature_maps)  # Watch these maps

        # 2. Manually pass the feature maps through the classifier layers
        x = feature_maps
        for layer in classifier_layers:
            x = layer(x)
        predictions = x

        # 3. Get the score for the top predicted class
        top_pred_index = tf.argmax(predictions[0])
        top_class_channel = predictions[:, top_pred_index]

    # 4. Calculate the gradient of the top class score w.r.t. the feature maps
    grads = tape.gradient(top_class_channel, feature_maps)

    # 5. Pool gradients and compute heatmap
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    feature_maps = feature_maps[0]
    heatmap = tf.reduce_sum(tf.multiply(feature_maps, pooled_grads), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)

    return heatmap, predictions.numpy()


# 4. Run the Process
TEST_IMAGE_PATH = 'test_images/Te-piTr_0003.jpg'

# Make sure the file exists before running
try:
    with open(TEST_IMAGE_PATH, 'rb') as f:
        pass
except FileNotFoundError:
    print(f"❌ Error: Image file not found at '{TEST_IMAGE_PATH}'")
    print("Please edit the script to provide the correct path.")
    sys.exit()


print(f"\n--- 4. Running Grad-CAM on: {TEST_IMAGE_PATH} ---")
preprocessed_img = preprocess_image(TEST_IMAGE_PATH)
heatmap, predictions = get_gradcam_heatmap(preprocessed_img, model)
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
confidence = np.max(predictions[0])

print(f" > Predicted Class: {predicted_class} ({confidence:.2%} confidence)")
print(f" > Heatmap generated with shape: {heatmap.shape}")

# 5.Visualize and Superimpose the Heatmap (IMPROVED) ---
original_img = image.load_img(TEST_IMAGE_PATH, target_size=IMG_SIZE)
original_img = image.img_to_array(original_img)

heatmap = np.uint8(255 * heatmap)
jet = plt.get_cmap("jet")
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[heatmap]

jet_heatmap = image.array_to_img(jet_heatmap)
jet_heatmap = jet_heatmap.resize((original_img.shape[1], original_img.shape[0]))
jet_heatmap = image.img_to_array(jet_heatmap)

superimposed_img = jet_heatmap * 0.4 + original_img
superimposed_img = image.array_to_img(superimposed_img)


# Get the base name of the input file (e.g., 'Te-gl_0015.jpg')
base_name = os.path.basename(TEST_IMAGE_PATH)
# Remove the extension to get just the name (e.g., 'Te-gl_0015')
file_name_no_ext = os.path.splitext(base_name)[0]
# Create a new, unique output path
output_path = f"gradcam_{file_name_no_ext}.jpg"


superimposed_img.save(output_path)
print(f"\n✅ Grad-CAM visualization saved to: {output_path}")