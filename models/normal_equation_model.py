"""

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# download the data from given URL and with given columns
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
columns = ['symbolying','normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location'
,'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system'
,'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']


#loading the dataset using pandas and replacing "?" with NA values
raw_data = pd.read_csv(url,names=columns, na_values="?")
#We ignore 'symboling' column
raw_data.pop("symbolying")

# drop all rows with missing values
dataset = raw_data.copy()
dataset =dataset.dropna()

#Now we assign a one-hot encoding for eatch categorical feature
dataset = pd.get_dummies(dataset,columns=["num-of-cylinders","num-of-doors","make","fuel-type","aspiration","body-style","drive-wheels"
                        ,"engine-location","engine-type","fuel-system"],
                        prefix=["num-of-cylinders","num-of-doors","make","fuel-type","aspiration","body-style","drive-wheels"
                        ,"engine-location","engine-type","fuel-system"],prefix_sep='_')


# We set 80% of the available data for training and the rest for testing
train_dataset = dataset.sample(frac = 0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)


train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('normalized-losses')
test_labels = test_features.pop('normalized-losses')



train_features = train_features.to_numpy()
test_features = test_features.to_numpy()

train_labels = train_labels.to_numpy()
test_labels = test_labels.to_numpy()

#since our target variable is always positive, we can log scale it and exponentiate after prediction. it works well with regression models.
log_labels = np.log(train_labels)


#first we get the identity matrix
identity_size = train_features.shape[1]
identity_matrix= np.zeros((identity_size, identity_size))
np.fill_diagonal(identity_matrix, 1)

# we set a regularization parameter labmda to 1
lamb = 1
xTx = train_features.T.dot(train_features) + lamb * identity_matrix
xTx_inv = np.linalg.inv(xTx)
xTx_inv_xT = xTx_inv.dot(train_features.T)
theta = xTx_inv_xT.dot(log_labels)

#since we took the log of the lables, we must exponentiate after each prediction
prediction = np.exp(test_features.dot(theta))


mse = (np.square(prediction - test_labels)).mean()
percentage = np.mean(np.abs(prediction - test_labels)/(test_labels))
print("mean squared error is {} and the percentage is {}".format(mse,percentage))
#Our normal equation model achieved a mean squared error of 279.0 (16.67 RMSE) and a percentage Error of 11.75%
plt.plot(test_labels)
plt.plot(prediction)
plt.show()