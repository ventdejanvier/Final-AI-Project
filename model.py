import numpy as np 
import pandas as pd 
import matplotlib.pyplot as pltz

from keras.datasets import mnist
import tensorflow as tf

import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical, plot_model


(x_train, y_train),(x_test, y_test) = mnist.load_data()


unique, counts = np.unique(y_train, return_counts=True)
print("Train labels: ", dict(zip(unique, counts)))


unique, counts = np.unique(y_test, return_counts=True)
print("\nTest labels: ", dict(zip(unique, counts)))

num_labels = len(np.unique(y_train))
print("num_labels = ",num_labels)

y_train = to_categorical(y_train)

image_size = x_train.shape[1]
input_size = image_size * image_size

x_train = x_train.reshape(-1, 28 * 28) / 255.0
x_test = x_test.reshape(-1, 28 * 28) / 255.0


input_size = 784
hidden_size_1 = 256
hidden_size_2 = 128
output_size = 10
epochs = 10
lr = 0.01
batch_size = 200


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_size_1, activation='relu', input_shape=(input_size,)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(hidden_size_2, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(output_size, activation='softmax')
])


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])


history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)


def predict_labels(model, x_test):
    y_pred = model.predict(x_test)
    predicted_labels = np.argmax(y_pred, axis=1)
    return predicted_labels


def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_samples = len(y_true)
    accuracy = correct_predictions / total_samples
    return accuracy

train_losses = history.history['loss']


plt.plot(train_losses)
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

print("----Model summary----- ")
model.summary()



if len(y_test.shape) == 1:
    y_test_onehot = np.zeros((len(y_test), output_size))
    y_test_onehot[np.arange(len(y_test)), y_test] = 1
else:
    y_test_onehot = y_test


y_test_pred = predict_labels(model, x_test)


test_accuracy = calculate_accuracy(np.argmax(y_test_onehot, axis=1), y_test_pred)

print(f'Test Accuracy: {test_accuracy:.4f}')




conf_matrix = confusion_matrix(np.argmax(y_test_onehot, axis=1), y_test_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=True, yticklabels=True)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

model.save("Model.h5")


try:
    model = tf.keras.models.load_model("Model.h5")
    print("Model loaded successfully")
    epochs = 1
except Exception as e:
    print(f"Error loading the model: {e}")
