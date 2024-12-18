y_min = 0
y_max = 100

# Input sequence
input_sequence = np.array([[5, 6, 7]])
''' Normalize input'''
input_sequence = (input_sequence - y_min) / (y_max - y_min)
input_sequence = input_sequence.reshape((1, 3))

prediction = model.predict(input_sequence)
# De-normalize


prediction_original = prediction * (y_max - y_min) + y_min
print("De-Normalized Predictions: ", np.round(prediction_original))
