# libraries
from sklearn.datasets import fetch_mldata
import numpy as np
from sklearn.model_selection import train_test_split

# load data 
mnist = fetch_mldata('MNIST original', data_home='E:/y_elk/Documents/PROJETS_GIT/ML_python/mnist')

#
#print(mnist.data.shape)

# sampling
sample = np.random.randint(70000, size=5000)
data = mnist.data[sample]
target = mnist.target[sample]

#
#print(data.shape)

# create training and test set
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.8)