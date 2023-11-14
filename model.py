import tensorflow as tf
import matplotlib.pyplot as plt


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
    Evaluates the trained CNN model on the provided data generator and prints the evaluation results.

    Args:
        model (tf.keras.models.Sequential): The trained CNN model.
        validation: The data for evaluation and prediction.

    Side Effect:
        Prints the evaluation results.
    """
    print(model.evaluate(validation))
    model.evaluate(validation)
    prediction = model.predict(validation)
    print(prediction)
