import pathlib
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
import splitfolders
import matplotlib.pyplot as plt
import tensorflow as tf

splitfolders.ratio("Data", output="TestingData", seed=42, ratio=(.7, .3), group_prefix=None, move=False)

train_data_dir = pathlib.Path('TestingData/train')
print(train_data_dir)

test_data_dir = pathlib.Path('TestingData/val')
print(test_data_dir)

train_data_gen = ImageDataGenerator(rescale=1. / 255)
validation_data_gen = ImageDataGenerator(rescale=1. / 255)

# Preprocess all training images
train_generator = train_data_gen.flow_from_directory(
    'TestingData/train',
    target_size=(128, 128),
    batch_size=32,
    color_mode="grayscale",
    class_mode='categorical',
    seed=42)

# Preprocess all testing images
validation_generator = validation_data_gen.flow_from_directory(
    'TestingData/val',
    target_size=(128, 128),
    batch_size=32,
    color_mode="grayscale",
    class_mode='categorical',
    seed=42)

emotion_model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),

    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(29, activation='softmax')
])

emotion_model.compile(optimizer='adam',
                      loss="categorical_crossentropy",
                      metrics=['accuracy'])

emotion_model.summary()
# Train the neural network/model
emotion_model_info = emotion_model.fit(train_generator, epochs=10, validation_data=validation_generator)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
# Plot training & validation accuracy values
axes[0].plot(emotion_model_info.history['accuracy'])
axes[0].plot(emotion_model_info.history['val_accuracy'])
axes[0].set_title('Model accuracy')
axes[0].set_ylabel('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
axes[1].plot(emotion_model_info.history['loss'])
axes[1].plot(emotion_model_info.history['val_loss'])
axes[1].set_title('Model loss')
axes[1].set_ylabel('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend(['Train', 'Validation'], loc='upper left')

plt.show()

filename = "hGDetection.h5"
tf.keras.models.save_model(emotion_model, filepath=filename)