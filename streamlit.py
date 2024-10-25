
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

def model():
    input_size = 784
    hidden_size_1 = 256
    hidden_size_2 = 128
    output_size = 10

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_size_1, activation='relu', input_shape=(input_size,)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(hidden_size_2, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(output_size, activation='softmax')
    ])

    return model

model = model()
model.load_weights('Model.h5')

st.title('Handwritten Digit Recognition')
st.write('This is a simple handwritten digit recognition web app using a neural network.')

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = tf.image.decode_image(uploaded_file.read(), channels=1)
    image = tf.image.resize(image, [28, 28])
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, (1, 28 * 28))
    prediction = model.predict(image)
    prediction = np.argmax(prediction)
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write(f'The predicted digit is: {prediction}')