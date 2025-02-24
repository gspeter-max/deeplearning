from tensorflow.keras.models import Model
from tensorflow.keras import layers 
import tensorflow as tf

input_shape = (None, 16, 112, 112, 3) 

input = layers.Input(shape = input_shape)

x = layers.Conv3D(filters = 64, kernel_size = (3,3,3),padding = 'same')(input)
x = layers.BatchNormalization()(x)
x = layers.activation('relu')(x)
x = layers.MaxPooling3D(pool_size = (1,2,2), strides = (1,2,2))

x = layers.Conv3D(filters = 128, kernel_size = (3,3,3), padding = 'same')(x)
x = layers.BatchNormalization()(x)
x = layers.activation('relu')(x)
x = layers.MaxPooling3D(pool_size = (2,2,2), strides = (2,2,2))

x = layers.Conv3D(filters = 256, kernel_size = (3,3,3), padding = 'same')(x)
x = layers.BatchNormalization()(x)
x = layers.activation('relu')
x = layers.MaxPooling3D(pool_size = (2,2,2), strides = (2,2,2))

shape = x.shape
x = tf.reshape(x, (-1, shape[1], shape[2]*shape[3]*shape[4]))

x = layers.Bidirectional(layers.LSTM(128, return_sequence = True))(x)

x = layers.Attention()([x,x])
x = tf.reduce_mean(axis = 1)(x)

x = layers.Dense(512, activation = 'relu')
x = layers.Dropout(0.5)
output = layers.Dense(101, activation = 'softmax')(x)

model = Model(input, output)
model.compile(
    optimizer = 'Adam', 
    loss = 'sparse_categorical_crossentropy', 
    metrics = ['accuracy']
)
model.summary() 
