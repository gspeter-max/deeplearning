from tensorflow.keras.models import Model
from tensorflow.keras import layers
import tensorflow as tf

input_shape = (16, 112, 112, 3)  # Correct input shape for each video clip
inputs = layers.Input(shape=input_shape)

# 3D CNN Layers
x = layers.Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(x)

x = layers.Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)

x = layers.Conv3D(filters=256, kernel_size=(3, 3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)

# Reshape for RNN
shape = x.shape
x = layers.Reshape((shape[1], shape[2] * shape[3] * shape[4]))(x)

# BiLSTM for Temporal Features
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

# Attention Layer
attention = layers.Attention()([x, x])
x = layers.GlobalAveragePooling1D()(attention)

# Fully Connected Layers
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(101, activation='softmax')(x)

# Build Model
model = Model(inputs, outputs)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Model Summary
model.summary()
