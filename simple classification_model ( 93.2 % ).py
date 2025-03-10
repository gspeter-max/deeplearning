import numpy as np
import pandas as pd

file_path = "/content/drive/MyDrive/large_classification_data.csv"
df =  pd.read_csv(file_path)
x_temp  = df.drop(columns = 'target')
y = df['target']
print(df.shape)

from sklearn.decomposition import PCA

reduction = PCA(n_components= 10, random_state = 42)
x = reduction.fit_transform(x_temp)

from sklearn.model_selection import train_test_split
x_train, x_test , y_train , y_test = train_test_split( x, y,test_size = 0.1 , random_state = 42)

from imblearn.combine import SMOTETomek

sampler = SMOTETomek(random_state = 42)
x_train_sampler ,y_train_sampler   = sampler.fit_resample(x_train, y_train)

x_training,validation_x,  y_training, validation_y = train_test_split(x_train_sampler, y_train_sampler , test_size = 0.1, random_state = 42)
shape = x_training.shape
import tensorflow as tf

model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(80, input_shape  = (x_training.shape[-1],), activation = 'relu'),
        tf.keras.layers.BatchNormalization() ,
        tf.keras.layers.Dropout(0.2 ) ,

        tf.keras.layers.Dense(100, activation = 'relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2) ,

        tf.keras.layers.Dense(150, activation = 'relu') ,
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(5, activation = 'softmax' )
    ])

model.compile(
        optimizer = 'Adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
        )

model.fit(x_training ,y_training, epochs = 10, batch_size = 125 ,validation_steps=(validation_x, validation_y ) )
''' 
961/961 ━━━━━━━━━━━━━━━━━━━━ 5s 5ms/step - accuracy: 0.9252 - loss: 0.2391
''' 
