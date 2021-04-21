import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
import numpy as np
import os 

# We set seeds for reproduciability

tf.random.set_seed(1)
np.random.seed(1)

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
# download the data from given URL and with given columns
columns = ['symbolying','normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location'
        ,'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system'
        ,'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

#loading the dataset using pandas and replacing "?" with NA values
raw_data = pd.read_csv(url,names=columns,na_values="?")

#We ignore 'symboling' column
raw_data.pop('symbolying')

#drop all rows with missing values
dataset = raw_data.dropna().copy()

# we perform min-max normalization as the following
norm_data = dataset.loc[:,["wheel-base","length","width","height","curb-weight","engine-size","bore","stroke","compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg","price"]].copy()
norm_data_mins = norm_data.min()
norm_data_maxs = norm_data.max()
normalized_features =(norm_data-norm_data_mins)/(norm_data_maxs -  norm_data_mins)
dataset.loc[:,["wheel-base","length","width","height","curb-weight","engine-size","bore","stroke","compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg","price"]] = normalized_features.loc[:,["wheel-base","length","width","height","curb-weight","engine-size","bore","stroke","compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg","price"]]

dataset = pd.get_dummies(dataset,columns=["num-of-cylinders","num-of-doors","make","fuel-type","aspiration","body-style","drive-wheels"
                        ,"engine-location","engine-type","fuel-system"],
                        prefix=["num-of-cylinders","num-of-doors","make","fuel-type","aspiration","body-style","drive-wheels"
                        ,"engine-location","engine-type","fuel-system"],prefix_sep='_')

# We set 80% of the available data for training and the rest for testing
train_dataset = dataset.sample(frac = 0.8, random_state=1)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('normalized-losses')
test_labels = test_features.pop('normalized-losses')
# Working with such small dataset, it is better to train the model sample by sample for it to converge quickly
batch_size = 1
train_ds = tf.data.Dataset.from_tensor_slices((np.array(train_features),np.log(np.array(train_labels)))).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((np.array(test_features),np.log(np.array(test_labels)))).batch(batch_size)

class Regression_Model(Model):

    def __init__(self):
        super(Regression_Model,self).__init__()
        self.dense1 = Dense(64, activation='relu' )
        self.dense2 = Dense(32, activation='relu' )
        self.dense3 = Dense(16, activation='relu' )
        self.final = Dense(1)


    def call(self,x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.final(x)


class Trainer:

    def __init__(self):
        self.model:Regression_Model = Regression_Model()
        
        self.loss = self.get_loss()
        self.optimizer = self.get_optimizer("SGD")
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')


    def get_optimizer(self,opt="adam"):
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.1,decay_steps=10000,decay_rate=1,staircase=False)
        if opt == 'adam':
            return tf.keras.optimizers.Adam(0.001)
        elif opt == 'SGD':
            return tf.keras.optimizers.SGD(lr_schedule)
        else:
            raise "This optimizer does not exist"

    def get_loss(self,loss='MSE'):
        if loss == 'MSE':
            return tf.keras.losses.MSE
        if loss == 'MAE':
            return tf.keras.losses.MAE
        else:
            raise "error"
    def predict(self,features):
        return self.model.predict(features)

    @tf.function
    def train_step(self,features,values):
        with tf.GradientTape() as tape:
            predictions = self.model(features,training = True)
            loss = self.loss(values,predictions)
            gradients = tape.gradient(loss,self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients,self.model.trainable_variables))
            self.train_loss(loss)
    
    @tf.function
    def test_step(self,features,values):
        predictions = self.model(features,training=False)
        loss = self.loss(values,predictions)
        self.test_loss(loss)

    def train(self):
        
        for epoch in range(100):
            self.train_loss.reset_states()
            self.test_loss.reset_states()

            for features,values in train_ds:
                self.train_step(features,values)

            for features,values in test_ds:
                self.test_step(features,values)
            
            print(
                f'Epoch {epoch + 1}, '
                f'Loss: {self.train_loss.result()}, '
                f'Test Loss: {self.test_loss.result()}, '
            )

# Now we reset the random seeds for reproduciability and start the training!
os.environ['PYTHONHASHSEED']=str(1)
tf.random.set_seed(1)
np.random.seed(1)
trainer = Trainer()
trainer.train()

# lets see th esummary of the trained model
trainer.model.summary()

# Now we test the model on the test set
predictions = np.exp(np.reshape(trainer.model.predict(np.array(test_features)),(np.shape(test_features)[0],)))
mse = (np.square(predictions - test_labels)).mean()
percentage = np.mean(np.abs(predictions - test_labels)/(test_labels))
print("mean squared error is {} and the percentage is {}".format(mse,percentage))

#Our deep neural ntwork model achieved a mean squared error of 226.68 (15.05 RMSE) and a percentage Error of 9.35%.

plt.plot(predictions)
plt.plot(np.array(test_labels))
plt.legend(labels = ["predictions","labels"])
plt.show()