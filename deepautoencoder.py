from tensorflow.keras.models import Model
from tensorflow.keras import layers 
import tensorflow as tf

class custom_loss(tf.keras.losses.Loss): 
    def call(self,y_true, y_pre): 
        first_error = tf.reduce_mean(tf.square(y_true - y_pre), axis = 1) 

        return tf.reduce_mean(tf.square(first_error))


class DeepAutoencoder(Model): 

    def __init__(self, input_dim):
        super(DeepAutoencoder, self).__init__() 

        self.encoder = tf.keras.Sequential([
            layers.Dense(128), 
            layers.LayerNormalization(), 
            layers.Activation('relu'), 
            layers.Dropout(0.2), 

            layers.Dense(64), 
            layers.LayerNormalization(), 
            layers.Activation('relu'), 
            layers.Dropout(0.2), 

            layers.Dense(32), 
            layers.LayerNormalization(), 
            layers.Activation('relu')
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(64), 
            layers.LayerNormalization(), 
            layers.Activation('relu'), 
            layers.Dropout(0.2), 

            layers.Dense(128), 
            layers.LayerNormalization(), 
            layers.Activation('relu'), 
            layers.Dropout(0.2), 

            layers.Dense(input_dim, activation = 'softmax')
        ])

    def call(self, input): 
        
        encoder = self.encoder(input)
        decoder = self.decoder(encoder)
        return decoder 


input_dim = 122 
model = DeepAutoencoder(input_dim)
model.compile(
    optimizer = 'Adam', 
    loss = custom_loss() , 
    metrics = ['mse']

)

import numpy as np 

x_train = np.random.randn(10000,input_dim).astype(np.float32)
x_test = np.random.randn(100,input_dim).astype(np.float32)

model.fit(x_train, x_train,epochs= 50, batch_size = 256, validation_split = 0.2) 

reconstract_error = model.predict(x_test)
reconstract_error = np.mean(np.square(x_test - reconstract_error), axis = 1)

threshold = np.percentile(reconstract_error, 95)
detected_anomalies = reconstract_error > threshold 
rate_anomalies = np.mean(detected_anomalies)

print(f"rate_anomalies  : {rate_anomalies}") 

_, model_accuracy = model.evaluate(x_test,x_test)
print(model_accuracy)
