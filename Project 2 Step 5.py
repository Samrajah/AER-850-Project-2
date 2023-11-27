import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

def load_and_preprocess_image(image_path):
    # Load and preprocess the test image
    test_image = image.load_img(image_path, target_size=(100, 100))
    test_image_array = image.img_to_array(test_image)
    test_image_array = np.expand_dims(test_image_array, axis=0)
    test_image_array /= 255.0  # Normalize the image
    return test_image_array

def make_prediction(model, test_image_array):
    # Make a prediction using the trained model
    prediction = model.predict(test_image_array)

    # Get the predicted class
    predicted_class = np.argmax(prediction)
    return predicted_class, prediction.flatten()

def main():
    # Define the paths to the test images
    test_image_paths = [
        r'C:\Users\samby\OneDrive\Documents\GitHub\AER-850-Project-2\Project 2 Data\Data\Test\Medium\Crack__20180419_06_19_09,915.bmp',
        r'C:\Users\samby\OneDrive\Documents\GitHub\AER-850-Project-2\Project 2 Data\Data\Test\Large\Crack__20180419_13_29_14,846.bmp'
    ]

    # Load the trained model
    model = load_model('Project_2_model.h5')

   
    class_names = {
        0: 'Large Crack',
        1: 'Medium Crack',
        2: 'Small Crack',
        3: 'No Crack'
    }

    # Iterate through the specified test images
    for test_image_path in test_image_paths:
        # Load and preprocess the test image
        test_image_array = load_and_preprocess_image(test_image_path)

        # Make a prediction
        predicted_class, class_probabilities = make_prediction(model, test_image_array)

        # Display the prediction
        print(f'\nImage: {os.path.basename(test_image_path)}')
        print(f'Predicted Class: {class_names[predicted_class]}')

        # Display the percentages for each class
        for class_label, probability in enumerate(class_probabilities):
            print(f'Percentage for {class_names[class_label]}: {probability * 100:.2f}%')

if __name__ == "__main__":
    main()