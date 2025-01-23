import math
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split  # Import train_test_split
from tensorflow.keras.utils import to_categorical
from sympy import *
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import json
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical

# Example usage:
def DoG_OOCS(x, y, center, gamma, radius):

    # compute sigma from radius of the center and gamma(center to surround ratio)
    sigma = (radius / (2 * gamma)) * (math.sqrt((1 - gamma ** 2) / (-math.log(gamma))))
    excite = (1 / (gamma ** 2)) * math.exp(-1 * ((x - center) ** 2 + (y - center) ** 2) / (2 * ((gamma * sigma) ** 2)))
    inhibit = math.exp(-1 * ((x - center) ** 2 + (y - center) ** 2) / (2 * (sigma ** 2)))

    return excite , inhibit
def On_Off_Center_filters(radius, gamma, in_channels, out_channels):

    # size of the kernel
    kernel_size = int((radius/gamma)*2-1)
    # center node index
    centerX = int((kernel_size+1)/2)

    posExcite = 0
    posInhibit = 0
    negExcite = 0
    negInhibit = 0

    for i in range(kernel_size):
        for j in range(kernel_size):
            excite, inhibit = DoG_OOCS(i+1,j+1, centerX, gamma, radius)
            if excite > inhibit:
                posExcite += excite
                posInhibit += inhibit
            else:
                negExcite += excite
                negInhibit += inhibit

    # Calculating A-c and A-s, with requiring the positive vlaues sum up to 1 and negative vlaues to -1
    x, y = symbols('x y')
    solution = solve((x * posExcite + y * posInhibit - 1, negExcite * x + negInhibit * y + 1), x, y)
    A_c, A_s = float(solution[x].evalf()), float(solution[y].evalf())

    # making the On-center and Off-center conv filters
    kernel = np.zeros([kernel_size, kernel_size, in_channels, out_channels])

    for i in range(kernel_size):
        for j in range(kernel_size):
            excite, inhibit = DoG_OOCS(i+1,j+1, centerX, gamma, radius)
            weight = excite*A_c + inhibit*A_s

            kernel[i][j] = tf.fill([in_channels, out_channels], weight)

    return kernel.astype(np.float32)
def create_DOG_kernels(gamma, in_channels, out_channels):
    #format kernel for the network
    radius = 2*gamma #to assure that the kernelsize is always 3
    excitatory = On_Off_Center_filters(radius, gamma, in_channels, out_channels)
    excitatory = excitatory.squeeze(axis=-1)  # Removes the last dimension
    inhibitory = excitatory*-1
    return excitatory,inhibitory

# So that we dont have to calculate the create_DOG_kernels() function each time we
# change a layer I will calculate it before iterating over the layers. It will be
# calculated twice(inhibitory and excitatory) and the following function will chose
# one randomly

def kernel_chooser(excitatory,inhibitory):
  x = random.randint(0, 1)
  if x == 0:
    return excitatory
  else:
    return inhibitory


in_channels = 1
out_channels = 1
# Removes the last dimension

def model_initializer(gamma,  ratio):

  model = MobileNetV2(weights=None, include_top=False, input_shape=(32, 32, 3))
  #def take_layer_and_change_kernel_subset(percent_changed_kernels,layer):

  #generate Kernels
  excitatory, inhibitory = create_DOG_kernels(gamma, in_channels, out_channels)


  # Modify specific DepthwiseConv2D layers
  for layer in model.layers:
      if type(layer)==tf.keras.layers.DepthwiseConv2D:
          weights = layer.get_weights()  # [kernel, bias] if `use_bias=True`
          kernel = weights[0]          # as weights can have biases etc

          #define which layers to change
          numbers = list(range(kernel.shape[2]))
          random_subset = random.sample(numbers, int(ratio *kernel.shape[2]))

          for i in random_subset:
              kernel[:, :, i, :] = kernel_chooser(excitatory,inhibitory)

          weights[0] = kernel
          layer.set_weights(weights)
  return model

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# Initialize ranges for gamma and radius
# adjust parameter grid here
gamma_values = np.linspace(0.1, 0.7, 7) 
ratio_values = [0.3,0.45,0.6,0.75,0.9]
epochs =  20

# Results storage
results = []
# Iterate over gamma and ratio values
for gamma in np.linspace(0.1, 0.7, 7):
    print(gamma)
    for ratio in ratio_values:
        base_model = model_initializer(gamma,ratio)
    # Add classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)  # Pooling layer
        x = Dense(128, activation="relu")(x)  # Fully connected layer
        outputs = Dense(10, activation="softmax")(x)  # Output layer for 10 classes

        model_X = Model(inputs=base_model.input, outputs=outputs)

        # Compile the model
        model_X.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        history = model_X.fit(
        x_train, y_train,validation_data=(x_val, y_val),
        epochs=epochs,batch_size=64,verbose=1)               # Adjust the number of epochs based on your dataset and hardware


        test_loss, test_accuracy = model_X.evaluate(x_test, y_test, verbose=0)
        print(f"Test Accuracy for gamma={gamma:.2f}: {test_accuracy:.4f}")
        results.append((gamma, ratio, test_accuracy))
        print(results)
        results_array = np.array(results)

        # save results in json
        file_full_path = f"/Users/zehra/Desktop/results_gamma_ratio_{gamma}_{ratio}.json"
        # Convert the array to a list of dictionaries for JSON serialization
        results_dict_list = [
        {"gamma": float(row[0]), "ratio": float(row[1]), "accuracy": float(row[2])}
        for row in results_array]

        # Save to a JSON file
        try:
            with open(file_full_path, 'w') as json_file:
                json.dump(results_dict_list, json_file, indent=4)
            print(f"JSON file successfully saved at: {file_full_path}")
        except Exception as e:
            print(f"Error saving JSON file: {e}")

    print("finished")

print("done")
