# Creating the Dataset

testing_data = []
training_data = []

for i in range(0, 100):
    if i >= 3:
        sequence = (i - 3, i - 2, i - 1)
        output_data = i
        training_data.append((sequence, output_data))

print(training_data)


