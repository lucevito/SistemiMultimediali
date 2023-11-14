import os
import cv2
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def fiximage(data):
    """
    Converts images to a specific format and resolves format issues.

    Args:
        data (str): Path of the directory containing the images to process.
    """
    for dir in os.listdir(data):
        for dir2 in os.listdir(os.path.join(data, dir)):
            if "conjunctive" in dir2:
                for dir3 in os.listdir(os.path.join(data, dir, dir2)):
                    for file in os.listdir(os.path.join(data, dir, dir2, dir3)):
                        img = cv2.imread(os.path.join(data, dir, dir2, dir3, file), cv2.IMREAD_UNCHANGED)
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                        cv2.imwrite(os.path.join(data, dir, dir2, dir3, file), img)


def create_train_data_generator(train_dir, target_size=(150, 150), batch_size=16):
    """
    Create an ImageDataGenerator object for training data.

    Args:
        train_dir (str): Directory path containing the training data.
        target_size (tuple, optional): Target size of the images (default is (150, 150)).
        batch_size (int, optional): Batch size for training (default is 16).

    Returns:
        train_d_gen: ImageDataGenerator object configured for training data.

    Description:
    This function creates an ImageDataGenerator object with specified augmentation parameters
    such as rotation, shear, width and height shift, horizontal and vertical flip, zoom, and fill mode.
    It then generates a flow from the directory containing training data, setting class mode to 'categorical'
    for multi-class classification, and returns the configured ImageDataGenerator object.
    """
    train_data_gen = ImageDataGenerator(
        rescale=1. / 255.,
        rotation_range=40,
        shear_range=.2,
        width_shift_range=.2,
        height_shift_range=.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )

    train_d_gen = train_data_gen.flow_from_directory(
        train_dir,
        class_mode='categorical',
        target_size=target_size,
        batch_size=batch_size
    )

    return train_d_gen


def create_validation_data_generator(validation_dir, target_size=(150, 150), batch_size=2):
    """
    Create an ImageDataGenerator object for validation data.

    Args:
        validation_dir (str): Directory path containing the validation data.
        target_size (tuple, optional): Target size of the images (default is (150, 150)).
        batch_size (int, optional): Batch size for validation (default is 2).

    Returns:
        val_d_gen: ImageDataGenerator object configured for validation data.

    Description:
    This function creates an ImageDataGenerator object with rescaling and generates a flow
    from the directory containing validation data, setting class mode to 'categorical'
    for multi-class classification. It returns the configured ImageDataGenerator object.
    """
    val_data_gen = ImageDataGenerator(rescale=1. / 255.)

    val_d_gen = val_data_gen.flow_from_directory(
        validation_dir,
        class_mode='categorical',
        target_size=target_size,
        batch_size=batch_size
    )

    return val_d_gen


def create_pre_trained_model(input_shape=(150, 150, 3), dense_units=100, dropout_rate=0.2):
    """
    Create a pre-trained InceptionV3-based model for binary classification.

    Args:
    input_shape (tuple): The shape of the input images (height, width, channels).
                          Default is (150, 150, 3).
    dense_units (int): Number of units in the densely connected layer. Default is 100.
    dropout_rate (float): Dropout rate for regularization in the densely connected layer.
                           Should be a float between 0 and 1. Default is 0.2.

    Returns:
      pre_trained_model (tf.keras.models.Model): A pre-trained InceptionV3-based model
                                                 with a custom dense layer for binary classification.
    """
    inception_v3 = InceptionV3(include_top=False, weights=None, input_shape=input_shape)
    inception_v3.load_weights('inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

    for layer in inception_v3.layers:
        layer.trainable = False

    last_layer = inception_v3.get_layer('mixed7')
    last_output = last_layer.output

    x = tf.keras.layers.Flatten()(last_output)
    x = tf.keras.layers.Dense(dense_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    pre_trained_model = tf.keras.models.Model(inputs=inception_v3.input, outputs=x)

    return pre_trained_model
