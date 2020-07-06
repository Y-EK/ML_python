# libraries
from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import neighbors

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

#model = neighbors.KNeighborsClassifier(n_neighbors=3)
#model.fit(x_train, y_train)
#print(model.predict([x_test[3]]))
#print(y_test[3])
# Percent error of the model applied to test data
#print(1 - model.score(x_test, y_test))

# Plot percent error in terms of number of neighbors
# Set x axis label 
plt.xlabel('n_neighbors')
# Set y axis label 
plt.ylabel('Percent error')

errors = []
for k in range(2,15):
    knn = neighbors.KNeighborsClassifier(k)
    errors.append(100*(1 - knn.fit(x_train, y_train).score(x_test, y_test)))
plt.plot(range(2,15), errors, 'o-')
plt.show()