# Import required librairies 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

house_data = pd.read_csv('house.csv')
house_data = house_data[house_data['loyer']<10000]

# house_data.shape
# print(house_data[:10])

#plt.plot(house_data['surface'], house_data['loyer'], 'ro', markersize=4)
#plt.show()

X = np.matrix([np.ones(house_data.shape[0]), house_data['surface'].values]).T
Y = np.matrix(house_data['loyer']).T
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)


plt.xlabel('Surface')
plt.ylabel('Loyer')

plt.plot([0,250], [theta.item(0),theta.item(0) + 250 * theta.item(1)], linestyle='--', c='#000000')
plt.plot(house_data['surface'], house_data['loyer'], 'ro', markersize=4)
plt.show()