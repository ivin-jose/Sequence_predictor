import random
from sklearn.preprocessing import MinMaxScaler


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

# print(len(training_data), len(testing_data))

''' Divide the input and output from the '''


# print(training_data)
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

