import numpy as np
import tensorflow as tf
 

class SimpleRNN:
    def __init__(self, sentence , seq_len ):
        self.sentence = sentence
        self.char = sorted(list(set(sentence)))
        self.seq_len = seq_len 

    def make_data(self): 
        self.index_to_char = {i : char for i , char in enumerate(self.char)}
        self.char_to_index = {char : i for i , char in enumerate(self.char)}
        train_sen = [] 
        train_label = [] 

        for i  in range(len(self.sentence) - self.seq_len):

            train_char = self.sentence[i : i + self.seq_len]
            train_label_char = self.sentence[i + self.seq_len]
            
            train_sen.append([self.char_to_index[train_chars] for train_chars in train_char])
            train_label.append(self.char_to_index[train_label_char])
        x = np.array(train_sen)
        y = np.array(train_label) 

        x_one_hot = tf.one_hot(x, len(self.char))
        y_one_hot = tf.one_hot(y,len(self.char))

        return x_one_hot , y_one_hot 

    
    def make_model(self): 
        model = tf.keras.models.Sequential([
            # tf.keras.layers.Input(shape=(self.seq_len, len(self.char))),
            tf.keras.layers.SimpleRNN(128,input_shape = (self.seq_len, len(self.char)), activation = 'relu'), 
            tf.keras.layers.Dense(len(self.char), activation = 'sigmoid')
        ])
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        x_train , y_train = self.make_data()
        model.fit(x_train, y_train, epochs = 100)

        return model
    

        

text = """
Deep learning is a subset of machine learning that uses neural networks with multiple layers to automatically learn patterns from data. 
A neural network consists of interconnected layers of nodes (neurons) that process information. 

Key concepts:
1. Neural Network: A model inspired by the human brain, consisting of layers of neurons.
2. Activation Function: Introduces non-linearity to help the network learn complex patterns.
3. Backpropagation: An optimization algorithm that updates weights by minimizing the error using gradients.
4. Gradient Descent: A technique to optimize model parameters by minimizing the loss function.
5. Loss Function: A metric that measures the difference between predictions and actual values.

Types of Neural Networks:
1. ANN (Artificial Neural Network): Basic deep learning model with fully connected layers.
2. CNN (Convolutional Neural Network): Specialized for image processing using convolution layers.
3. RNN (Recurrent Neural Network): Used for sequential data, remembers past inputs.
4. LSTM (Long Short-Term Memory): A type of RNN that overcomes short-term memory issues.
5. GAN (Generative Adversarial Network): Used for generating synthetic data.
6. Transformer: State-of-the-art model for NLP, the foundation of GPT and BERT.

Popular Deep Learning Frameworks:
1. TensorFlow: Developed by Google, supports both training and deployment.
2. PyTorch: Developed by Facebook, popular for research and flexibility.
3. Keras: High-level API for TensorFlow, easy to use and deploy models.
4. MXNet: Efficient framework for scalable deep learning models.
5. Caffe: Optimized for deep learning in computer vision applications.

Optimizers in Deep Learning:
1. SGD (Stochastic Gradient Descent): Updates weights using a small batch of data.
2. Adam: Adaptive learning rate optimization combining momentum and RMSProp.
3. RMSProp: Adjusts learning rates dynamically to improve training stability.
4. Adagrad: Adapts learning rate per parameter based on past gradients.

Applications of Deep Learning:
1. Image Recognition: Identifying objects, faces, or scenes in images.
2. Natural Language Processing: Understanding and generating human language.
3. Speech Recognition: Converting spoken language into text.
4. Autonomous Driving: Deep learning for self-driving car perception and decision-making.
5. Recommendation Systems: Personalized content recommendations in e-commerce and streaming platforms.
"""
sl = 5 

model = SimpleRNN(text, sl)
trained_model = model.make_model() 

test_text = ' Deep learnign is '
genrated_text = test_text 

for _ in range(500): 
    x = np.array([[model.char_to_index[char] for char in genrated_text[-model.seq_len:]]])
    x_one_hot = tf.one_hot(x,len(model.char))
    predict = trained_model.predict(x_one_hot)
    best_index = np.argmax(predict)
    genrated_text += model.index_to_char[best_index]

print(genrated_text)

