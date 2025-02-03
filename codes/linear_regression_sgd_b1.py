import numpy as np

# Example data (inputs and targets)
inputs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
targets = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

# Initialize weights and bias
a = 1.0
b = 0.0

# Learning rate
learning_rate = 0.0025

# Number of epochs
num_epochs = 5

# Training loop
for epoch in range(num_epochs):
    for i in range(len(inputs)):
        # Forward pass: compute predicted output
        prediction = a * inputs[i] + b

        # Compute the loss (squared error)
        loss = 0.5 * (prediction - targets[i]) ** 2

        # Compute gradients
        da = (prediction - targets[i]) * inputs[i]
        db = (prediction - targets[i])

        # Update weights and bias
        a -= learning_rate * da
        b -= learning_rate * db

    # Compute the loss for the whole dataset with current parameters
    total_loss = 0
    for k in range(len(inputs)):
        prediction = a * inputs[k] + b
        total_loss += 0.5 * (prediction - targets[k]) ** 2
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}')
    # Print the learned parameters
    print("Learned parameters:")
    print(f'Weight (a): {a:.4f}')
    print(f'Bias (b): {b:.4f}')
    test_input = 6.0
    predicted_output = a * test_input + b
    print(f'Predicted output for input {test_input}: {predicted_output:.4f}')
    print("")