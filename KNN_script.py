# libraries
from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import neighbors

# load data 
#mnist = fetch_mldata('MNIST original', data_home='E:/y_elk/Documents/PROJETS_GIT/ML_python/mnist')
mnist = fetch_mldata('MNIST original', data_home='./mnist')

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

# Get the value of n_neighbors corresponding to a minimal error 
ymin = min(errors)
xpos = errors.index(ymin)
xmin = range(2,15)[xpos]

# Get the best performing classifier
knn = neighbors.KNeighborsClassifier(xmin)
knn.fit(x_train, y_train)

# Get the predictions of the test data
predicted = knn.predict(x_test)

# We resize the data in the form of images
images = x_test.reshape((-1, 28, 28))

# Select a sample of 12 random images
select = np.random.randint(images.shape[0], size=12)

# Display the images with the associated prediction
fig,ax = plt.subplots(3,4)

for index, value in enumerate(select):
    plt.subplot(3,4,index+1)
    plt.axis('off')
    plt.imshow(images[value],cmap=plt.cm.gray_r,interpolation="nearest")
    plt.title('Predicted: {}'.format( predicted[value]) )

plt.show()
