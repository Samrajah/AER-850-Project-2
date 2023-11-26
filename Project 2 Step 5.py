from keras.models import load_model
from keras.preprocessing import image
import numpy as np

def load_and_preprocess_image(image_path):
    # Load and preprocess the test image
    test_image = image.load_img(image_path, target_size=(100, 100))
    test_image_array = image.img_to_array(test_image)
    test_image_array = np.expand_dims(test_image_array, axis=0)
    test_image_array /= 255.  # Normalize the image
    return test_image_array

def make_prediction(model, test_image_array):
    # Make a prediction using the trained model
    prediction = model.predict(test_image_array)

    # Get the predicted class
    predicted_class = np.argmax(prediction)
    return predicted_class

def main():
    # Define the path to the test image
    test_image_path = r'C:\Users\samby\OneDrive\Documents\GitHub\AER-850-Project-2\Project 2 Data\Data\Test\Medium'  # Replace with the actual path

    # Load the trained model
    model = load_model('Project_2_model.h5')

    # Load and preprocess the test image
    test_image_array = load_and_preprocess_image(test_image_path)

    # Make a prediction
    predicted_class = make_prediction(model, test_image_array)

    # Display the prediction
    print(f'The predicted class is: {predicted_class}')

if __name__ == "__main__":
    main()