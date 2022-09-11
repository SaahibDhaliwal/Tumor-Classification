import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import os
import math
import cv2
import imghdr
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import load_model

# Preprocess data
def preprocess ():
    data_loc = 'tumor_data'

    dir_list = os.listdir(data_loc)

    extension_list = ['jpeg', 'jpg', 'bmp', 'png']

    # This loop will go through all of the images and reformat them and ensure they are good to use
    for image_file in dir_list:
            for image in os.listdir(os.path.join(data_loc, image_file)):
                image_path = os.path.join (data_loc, image_file, image)
                try:
                    img = cv2.imread(image_path)
                    extension = imghdr.what(image_path)
                    if extension not in extension_list:
                        os.remove(image_path)
                except Exception as e:
                    print("Image issue")

    # Set the data

    # This function will create batches of data, standard is 32 images and labels in each
    data = tf.keras.utils.image_dataset_from_directory(data_loc, labels='inferred')

    # corresponding class to tumor type
    # Class 0 is giloma tumor
    # Class 1 is meningioma tumor
    # Class 2 is no tumor
    # Class 3 is pituitary tumor

    # Scale data

    data = data.map(lambda x, y: (x/255, y))

    # Split data into training set, validation set, and testig set

    train_size = 60;
    validate_size = 20;
    test_size = 10;

    train = data.take(train_size)
    validation = data.skip(train_size).take(validate_size)
    test = data.skip(train_size + validate_size).take(test_size)

# Develop the Convolutional Neural Network
def create_network():
    model = Sequential()
    model.add(Conv2D(16, (3,3), 1, activation = 'relu', input_shape = (256,256,3)))
    model.add(MaxPooling2D())

    model.add(Conv2D(32, (3,3), 1, activation = 'relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, (3,3), 1, activation = 'relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3,3), 1, activation = 'relu'))
    model.add(MaxPooling2D())

    model.add(Dropout(0.2)) # trying to stop overfitting with these dropout layers

    model.add(Flatten())

    model.add(Dense(units=128, activation = 'relu'))
    model.add(Dropout(0.2))

    model.add(Dense(4, activation='softmax')) # 4 output neurons because we have 4 classifications

    model.summary()

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


    # Create logs for training history

    logdir = 'logs'
    tensor_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    history = model.fit(train, epochs=10, validation_data=validation, callbacks=[tensor_callback])

    # Test data with the data set aside for testing
    for batch in test.as_numpy_iterator():
        X, y = batch
        model.evaluate (X,y)


    # Saving my model
    save_name = "tumorclassificationv2.h5"
    model.save(os.path.join('models', save_name))

# Create a testing loop to test all the images from my testing folder
def testing_loop ():
    test_loc = "Testing"
    directory = os.listdir(test_loc)
    for image_file in directory:
        for image in os.listdir(os.path.join(test_loc, image_file)):
            image_path = os.path.join (test_loc, image_file, image)
            try:
                test_img = cv2.imread(image_path)
                resize_test = tf.image.resize(test_img, (256, 256))
                plt.imshow(resize_test.numpy().astype(int))
                plt.show()
                prediction =  np.argmax(new_model.predict(np.expand_dims(resize_test/255,0)))
                if prediction == 0:
                     print("This is a glioma tumor")
                elif prediction == 1:
                     print("This is a meningioma tumor")
                elif prediction == 2:
                     print("There is no tumor")
                elif prediction == 3:
                     print("This is a pituitary tumor") 
            except Exception as e:
                print("There was an image error")


# Loading a model
def main ():
    model_name = 'tumorclassificationv2.h5'
    new_model = load_model(os.path.join('models', model_name))
    testing_loop()
    
if __name__ == "__main__":
    main()
    