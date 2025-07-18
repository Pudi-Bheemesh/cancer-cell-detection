import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Load the saved model
model_path = "./model"  # Specify the path where the model is saved
model = load_model(model_path)

# Folder containing the images for prediction
prediction_folder = "./images"

# Class labels (ensure they match the classes used in training)
class_labels = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

# Image size (should match the size used during training)
img_size = 128

# Function to preprocess images
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(img_size, img_size))  # Resize image to match the Training size
    image_array = img_to_array(image)  # Convert to array
    image_array = image_array / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Iterate over images in the folder
image_files = [f for f in os.listdir(prediction_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

print("Predictions:")
for image_file in image_files:
    image_path = os.path.join(prediction_folder, image_file)
    image_array = preprocess_image(image_path)

    # Predict using the model
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = class_labels[predicted_class]

    # Display the result
    print(f"Image: {image_file}, Predicted Label: {predicted_label}")

    # Optionally, display the image with its prediction
    plt.imshow(load_img(image_path))
    plt.title(f"Prediction: {predicted_label}")
    plt.axis('off')
    plt.show()
