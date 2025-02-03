import numpy as np

# Example data (inputs and targets)
inputs = np.array([[1.0, 2.0],  # Each row is a feature vector
                   [2.0, 3.0],
                   [3.0, 4.0],
                   [4.0, 5.0],
                   [5.0, 6.0]])  # Shape: (5, 2)

targets = np.array([4.0, 6.0, 8.0, 10.0, 12.0])  # Shape: (5,)

# Initialize weights and bias
W = np.ones(inputs.shape[1])  # One weight per feature
b = 0.0  # Bias

# Learning rate
learning_rate = 0.0025

# Number of epochs
num_epochs = 5

# Training loop
for epoch in range(num_epochs):
    for i in range(len(inputs)):
        # Forward pass: compute predicted output
        prediction = np.dot(W, inputs[i]) + b  # W • x + b

        # Compute the loss (squared error)
        loss = 0.5 * (prediction - targets[i]) ** 2

        # Compute gradients
        dW = (prediction - targets[i]) * inputs[i]  # Gradient for weights
        db = (prediction - targets[i])  # Gradient for bias

        # Update weights and bias
        W -= learning_rate * dW
        b -= learning_rate * db

    # Compute total loss for the epoch
    total_loss = 0
    for k in range(len(inputs)):
        prediction = np.dot(W, inputs[k]) + b
        total_loss += 0.5 * (prediction - targets[k]) ** 2

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}')
    print("Learned parameters:")
    print(f'Weights (W): {W}')
    print(f'Bias (b): {b}')

    test_input = np.array([6.0, 7.0])  # Example test input
    predicted_output = np.dot(W, test_input) + b
    print(f'Predicted output for input {test_input}: {predicted_output:.4f}')
    print("")

