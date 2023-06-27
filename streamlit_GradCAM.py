import streamlit as st
import keras
import numpy as np
# import tensorflow as tf
# from tensorflow import keras

# Display
from IPython.display import Image, display
from PIL import Image as pil_img
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import ast
def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.8):
    # Load the original image
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    st.image(((cam_path)))
    
    
def get_prediction_and_heatmap(model,image,loaded_dict):
    # Prepare image
    img_array = preprocess_image(image)

    # Make model
    #model = keras.models.load_model('VGG19_transfer_learning_animals10.h5')

    # Remove last layer's softmax
    model.layers[-1].activation = None

    # Print what the top predicted class is
    prediction = model.predict(img_array)

    # Get the predicted class label
    predicted_class = np.argmax(prediction)
    #print(predicted_class)
    animal_name = get_keys_by_value(loaded_dict, predicted_class)
    # Print the predicted class label
    #print("Predicted class:", animal_name)

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(img_array, model, 'block5_conv4')

    # Display heatmap
    # plt.matshow(heatmap)
    # plt.show()
    return heatmap, animal_name

def get_keys_by_value(dictionary, value):
    return [key for key, val in dictionary.items() if val == value]


def preprocess_image(input_image):
    image = pil_img.open(input_image)
    image = image.resize((224, 224))
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    # Convert the image to a numpy array
    image_array = np.array(image)

    # Normalize the pixel values
    image_array = image_array / 255.0

    # Expand dimensions to match the model input shape
    input_image = np.expand_dims(image_array, axis=0)
    
    return input_image


def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.utils.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    import tensorflow as tf
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def load_dict_from_csv(filename):
    dictionary = {}
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip header row
        for row in reader:
            key = row[0]
            value = ast.literal_eval(row[1])  # Convert value back to its original data type
            dictionary[key] = value
    return dictionary

def load_model():
    model = keras.models.load_model('VGG19_transfer_learning_animals10.h5')
    return model
def demo():
    model = load_model()
    loaded_dict = load_dict_from_csv('classes_dictionary.csv')
    # Upload image
    uploaded_image = st.file_uploader("Upload an image of one of the Animals 10 (dog, horse, elephant, butterfly, chicken, cat, cow, sheep, squirrel, dog, horse, elephant, butterfly, chicken, cat, cow, spider, squirrel)", type=["png", "jpg", "jpeg"])

# Check if image is uploaded
    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
        
        heatmap, predicted_class = get_prediction_and_heatmap(model , uploaded_image,loaded_dict)
        save_and_display_gradcam(uploaded_image, heatmap)
        trimmed_animal_name = str(predicted_class[0]).strip("[]")
        st.write("Predicted class:", trimmed_animal_name)
    
if __name__ == '__main__':
    demo()