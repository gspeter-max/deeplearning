from  tensorflow.keras.utils import to_categorical 
from tensorflow.keras  import layers , models 
import numpy as np
from tensorflow.keras.datasets import cifar10

(x_train, x_test), (y_train, y_test) = cifar10.load_data() 
x_train, x_test = x_train / 255.0 , x_test / 255.0

y_train,y_test = to_categorical(y_train) ,  to_categorical(y_test)
model = models.Sequential([
    layers.Conv2D(filters= 32, kernel_size = (3,3), activation= 'relu', input_shape = (32,32,3)) , 
    layers.MaxPool2D(pool_size = (2,2)), 
    layers.Conv2D(filters = 64, kernel_size = (3,3), activation= 'relu'), 
    layers.MaxPool2D( pool_size= (2,2)), 
    layers.Conv2D(filters= 32, kernel_size = (2,2), activation=  'relu'),

    layers.Flatten(), 
    layers.Dense(64, activation = 'relu'), 
    layers.Dense(32, activation= 'sigmoid')
])

model.compile(
    optimizer = 'Adam', 
    loss = 'categorical_crossentropy', 
    metrics = ['accuracy']
)

model.fit(x_train, y_train, epochs = 100, batch_size = 20, validation_data = (x_test, y_test), verbose = 2) 

model_loss, model_accuracy = model.evaluate(x_test, y_test,verbose='auto') 
print(model_loss, model_accuracy)
