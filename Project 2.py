# Step 1: Data Processing
from keras.preprocessing.image import ImageDataGenerator

# Define input image shape
input_shape = (100, 100, 3)

# Define data directories
train_data_dir = r'C:\Users\samby\OneDrive\Documents\GitHub\AER-850-Project-2\Project 2 Data\Data\Test'
validation_data_dir = r'C:\Users\samby\OneDrive\Documents\GitHub\AER-850-Project-2\Project 2 Data\Data\Train'
test_data_dir = r'C:\Users\samby\OneDrive\Documents\GitHub\AER-850-Project-2\Project 2 Data\Data\Validation'

# Set up data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Rescaling to normalize pixel values
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Rescale validation data (no data augmentation for validation)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
batch_size = 32

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

# Step 2: Neural Network Analysis
import tensorflow as tf
from tensorflow.keras import layers, models

# Define input image shape
input_shape = (100, 100, 3)

# Build the neural network
model = models.Sequential()

# Convolutional layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten layer
model.add(layers.Flatten())

# Fully connected layers
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))  # Dropout layer to reduce overfitting

# Output layer
model.add(layers.Dense(4, activation='softmax'))  # Assuming 4 output classes

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Adjust loss function based on your task
              metrics=['accuracy'])

# Display the model summary
model.summary()

# Step 3: Neural Network Analysis
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)

# Print the evaluation results
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')
