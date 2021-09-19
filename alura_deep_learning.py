import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import numpy as np
import random
import matplotlib.pyplot as plt


# Generate the Model using Keras
def generate_model(train_data, test_data, train_label, test_label, epochs):
    # Create the Multilayer Network
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(.2),
        keras.layers.Dense(64, activation='sigmoid'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # Fit model and get training history data
    history_model = model.fit(train_data, train_label, epochs=epochs, validation_split=0.2)
    plot_accuracy(history_model.history, epochs)
    plot_loss(history_model.history, epochs)
    # Evaluate model metrics
    test_loss, test_acc = model.evaluate(test_data, test_label, verbose=2)
    print(f'Mean Accuracy:{test_acc*100}\nMean Loss:{test_loss*100}\n')
    return model


def plot_accuracy(history, epoch):
    accuracy = [acc * 100 for acc in history['accuracy']]
    val_accuracy = [val_acc * 100 for val_acc in history['val_accuracy']]
    plt.figure()
    plt.plot(accuracy)
    plt.plot(val_accuracy)
    plt.title('Accuracy by Epochs')
    plt.xlabel('Epochs')
    plt.xticks(range(epoch))
    plt.ylabel('Accuracy(%)')
    plt.legend(['Train', 'Validation'])
    plt.show()


def plot_loss(history, epoch):
    loss = [loss * 100 for loss in history['loss']]
    val_loss = [val_loss * 100 for val_loss in history['val_loss']]
    plt.figure()
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Loss by Epochs')
    plt.xlabel('Epochs')
    plt.xticks(range(epoch))
    plt.ylabel('Loss (%)')
    plt.legend(['Train', 'Validation'])
    plt.show()


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]), color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
    plt.xticks(range(10), class_names, rotation=35, fontsize=8)


print(f'TensorFlow Version: {tf.__version__}')

# Loading the fashion_mnist Dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Data Info
print(40 * "=")
print(f'Fashion Mnist Dataset')
print(f'Train Image Dimensions:{train_images.shape}')
print(f'Test Image Dimensions:{test_images.shape}')
print(40 * "=")

# Plot Images
plt.figure()
plt.imshow(train_images[random.randint(0, len(train_images))])
plt.colorbar()
plt.grid(False)
plt.show()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

for image in range(0, 10):
    plt.subplot(2, 5, image + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[image], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[image]])
plt.show()

# no_normalize_model = generate_model(train_data=train_images, test_data=test_images,
#                                    train_label=train_labels, test_label=test_labels, epochs=5)
# predictions = no_normalize_model.predict(test_images)

# Normalize Data
train_images = train_images / 255.0
test_images = test_images / 255.0
normalize_model = generate_model(train_data=train_images, test_data=test_images,
                                 train_label=train_labels, test_label=test_labels, epochs=5)
predictions = normalize_model.predict(test_images)

num_rows = 5
num_cols = 2
num_images = num_rows * num_cols
plt.figure(figsize=(10 * num_cols, 4 * num_rows))

for image_index in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * image_index + 1)
    plot_image(image_index, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * image_index + 2)
    plot_value_array(image_index, predictions, test_labels)
plt.show()

# Saving a model
normalize_model.save('mnist_model.h5')
save_model = load_model('mnist_model.h5')

tests = save_model.predict(test_images)
print(tests)
