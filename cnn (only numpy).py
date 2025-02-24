import numpy as np

input_image = np.array([
    [1, 2, 1, 0, 2],
    [0, 1, 3, 1, 1],
    [2, 1, 0, 2, 3],
    [1, 0, 1, 2, 1],
    [3, 1, 2, 1, 0]
])

filter = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

def convolve2d(image, kernel):
    kernel_size = kernel.shape[0]
    output_size = image.shape[0] - kernel_size + 1
    output = np.zeros((output_size, output_size))
    
    for i in range(output_size):
        for j in range(output_size):
            region = image[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = np.sum(region * kernel)
    return output

convolved_output = convolve2d(input_image, filter)

def relu(x):
    return np.maximum(0, x)

activated_output = relu(convolved_output)

def max_pooling(feature_map, size=2, stride=2):
    output_size = (feature_map.shape[0] - size) // stride + 1
    pooled_output = np.zeros((output_size, output_size))
    
    for i in range(0, output_size):
        for j in range(0, output_size):
            region = feature_map[i*stride:i*stride+size, j*stride:j*stride+size]
            pooled_output[i, j] = np.max(region)
    return pooled_output

pooled_output = max_pooling(activated_output)

def flatten(feature_map):
    return feature_map.flatten()

flattened_output = flatten(pooled_output)

weights = np.random.randn(flattened_output.shape[0], 2)
biases = np.random.randn(2)

def dense_layer(inputs, weights, biases):
    return np.dot(inputs, weights) + biases

dense_output = dense_layer(flattened_output, weights, biases)

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

final_output = softmax(dense_output)

true_labels = np.array([1, 0])

def cross_entropy_loss(predicted, actual):
    return -np.sum(actual * np.log(predicted + 1e-15)) 

loss = cross_entropy_loss(final_output, true_labels)


print("Convolved Output:\n", convolved_output)
print("Activated Output (ReLU):\n", activated_output)
print("Pooled Output:\n", pooled_output)
print("Flattened Output:\n", flattened_output)
print("Dense Layer Output:\n", dense_output)
print("Final Output (Softmax Probabilities):\n", final_output)
print("Loss:\n", loss)

learning_rate = 0.01
grad = final_output - true_labels  


weights -= learning_rate * np.outer(flattened_output, grad)
biases -= learning_rate * grad
