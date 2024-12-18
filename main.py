import random
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


''' CREATEING DATASET '''

dataset = []
testing_data = []
training_data = []

for i in range(0, 100):
    if i >= 3:
        sequence = (i - 3, i - 2, i - 1)
        output_data = i
        dataset.append((sequence, output_data))


''' Spliting the data set into training and testing data 80/20%'''

# shuffle the datset
random.shuffle(dataset) 

split_count = int(len(dataset) * 0.8)

training_data = dataset[:split_count]
testing_data = dataset[split_count:]


''' Divide the input and output from the '''


x_train = [data[0] for data in training_data]
y_train = [data[1] for data in training_data]


x_test = [data[0] for data in testing_data]
y_test = [data[1] for data in testing_data]


''' Normalizing the Dataset '''

scaler = MinMaxScaler()

x_train_scaled = scaler.fit_transform(x_train)
y_train_scaled = scaler.fit_transform([[i] for i in y_train])

x_test_scaled = scaler.fit_transform(x_test)
y_test_scaled = scaler.fit_transform([[i] for i in y_test])


''' Creating model for prediction '''

x_train = np.array(x_train_scaled)
y_train = np.array(y_train_scaled)

x_test = np.array(x_test_scaled)
y_test = np.array(y_test_scaled)

''' Model '''

model = Sequential([
    layers.Dense(64, activation='relu', input_shape=(3,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

# model.summary()

''' Model Compiling '''

model.compile(
    optimizer = 'adam',
    loss = 'mse',
    metrics = ['mae']
)

''' Model Training '''

trained = model.fit(
    x_train, y_train,
    validation_data = (x_test, y_test),
    epochs = 100,
    batch_size = 32,
    verbose = 1
)



''' Model Evaluation '''

loss,  mae = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}, Test MAE: {mae}")




