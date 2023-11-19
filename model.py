import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.applications import InceptionV3


def create_model(input_shape=(150, 150, 3)):
    """
    Creates a convolutional neural network (CNN) model for image classification.

    Args:
        input_shape (tuple, optional): The shape of the input images. Defaults to (150, 150, 3).

    Returns:
        model (tf.keras.models.Sequential): The compiled CNN model.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.summary()
    return model


def compile_model(model):
    """
    Compiles a given CNN model with binary cross-entropy loss, Adam optimizer, and accuracy metric.

    Args:
        model (tf.keras.models.Sequential): The CNN model to be compiled.

    Returns:
        model (tf.keras.models.Sequential): The compiled CNN model.
    """
    model.compile(loss=tf.keras.losses.binary_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-05),
                  metrics=['accuracy'])
    return model


def train_model(model, train_generator, val_generator,
                epochs=20):
    """
    Trains the compiled CNN model on the provided training and validation data generators.

    Args:
        model (tf.keras.models.Sequential): The compiled CNN model to be trained.
        train_generator: The training data generator.
        val_generator: The validation data generator.
        epochs (int, optional): Number of training epochs. Defaults to 20.

    Returns:
        history (tf.keras.callbacks.History): Training history containing loss and accuracy.
    """
    history = model.fit(train_generator, validation_data=val_generator, epochs=epochs)
    return history


def plot_learning_curves(history):
    """
    Plots the learning curves (training and validation loss, training and validation accuracy)
    based on the training history.

    Args:
        history (tf.keras.callbacks.History): Training history containing loss and accuracy.

    Side Effect:
        Displays a matplotlib plot of the learning curves.
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 6))
    plt.plot(loss, 'b', label='training loss')
    plt.plot(val_loss, 'orange', label='validation loss')
    plt.legend()
    plt.show()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(acc, 'b', label='training acc')
    plt.plot(val_acc, 'orange', label='val acc')
    plt.legend()
    plt.show()


def evaluate_model(model, validation):
    """
    Evaluate the performance of a machine learning model on a validation dataset.

    Parameters:
    - model: The machine learning model to be evaluated.
    - validation: The validation dataset used to assess the model's performance.

    Returns:
    None
    """
    print(model.evaluate(validation))
    model.evaluate(validation)
    prediction = model.predict(validation)
    print(prediction)
    binary_predictions = (prediction > 0.5).astype(int)
    true_labels = validation.classes
    precision = precision_score(binary_predictions, true_labels, average='weighted')
    recall = recall_score(binary_predictions, true_labels, average='weighted')
    f1 = f1_score(binary_predictions, true_labels, average='weighted')
    print(f'Precision: {precision:.4f} ')
    print(f'Recall: {recall:.4f} ')
    print(f'F1 Score: {f1:.4f} ')


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
