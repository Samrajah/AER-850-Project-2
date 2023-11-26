
from keras.preprocessing.image import ImageDataGenerator

# Define input image shape
input_shape = (100, 100, 3)

# Define data directories
train_data_dir = 'Project 2 Data.zip/Data/train'
validation_data_dir = 'Project 2 Data.zip/Data/validation'
test_data_dir = 'Project 2 Data.zip/Data/test'

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