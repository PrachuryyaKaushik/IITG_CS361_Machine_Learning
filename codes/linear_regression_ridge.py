import numpy as np

# Initialize parameters
a = 0.0  # weight
b = 0.0  # bias

# Learning rate
learning_rate = 0.01

# Number of epochs
num_epochs = 5

# Regularization parameter (lambda)
lambda_reg = 0.1

# Training data
inputs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
targets = np.array([3.0, 5.0, 7.0, 9.0, 11.0])

# Training loop
for epoch in range(num_epochs):
    # Forward pass: compute predicted outputs
    predictions = a * inputs + b

    # Compute the loss (squared error with regularization)
    loss = 0.5 * np.sum((predictions - targets) ** 2) + 0.5 * lambda_reg * a ** 2

    # Compute gradients with regularization
    da = np.sum(2 * (predictions - targets) * inputs) + lambda_reg * a
    db = np.sum(2 * (predictions - targets))

    # Update weights and bias
    a -= learning_rate * da
    b -= learning_rate * db

    # Print the current state
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}')
    # Print the learned parameters
    print("Gradient for parameter a: {:.4f}".format(da))
    print("Gradient for parameter b: {:.4f}".format(db))
    print("Learned parameters:")
    print(f'Weight (a): {a:.4f}')
    print(f'Bias (b): {b:.4f}')
    test_input = 6.0
    predicted_output = a * test_input + b
    print(f'Predicted output for input {test_input}: {predicted_output:.4f}')
    print("")