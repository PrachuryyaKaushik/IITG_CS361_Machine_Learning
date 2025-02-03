import numpy as np

# Example multi-dimensional data (inputs and targets)
inputs = np.array([
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0],
    [4.0, 5.0],
    [5.0, 6.0]
])  # Shape: (5, 2)

targets = np.array([4.0, 6.0, 8.0, 10.0, 12.0])  # Shape: (5,)

# Initialize weights and bias
a = np.ones(inputs.shape[1])  # Shape: (2,)
b = 0.0

# Learning rate
learning_rate = 0.005

# Number of epochs
num_epochs = 5

# Batch size
batch_size = 2

# Training loop
for epoch in range(num_epochs):
    for i in range(0, len(inputs), batch_size):
        # Get the batch
        batch_inputs = inputs[i:i + batch_size]  # Shape: (batch_size, num_features)
        batch_targets = targets[i:i + batch_size]  # Shape: (batch_size,)

        # Forward pass: compute predictions
        predictions = np.dot(batch_inputs, a) + b  # Shape: (batch_size,)

        # Compute loss (squared error)
        loss = 0.5 * np.sum((predictions - batch_targets) ** 2)

        # Compute gradients
        error = predictions - batch_targets  # Shape: (batch_size,)
        da = np.dot(batch_inputs.T, error)  # Shape: (num_features,)
        db = np.sum(error)  # Scalar

        # Update weights and bias
        a -= learning_rate * da
        b -= learning_rate * db

    # Compute total loss for the whole dataset
    total_predictions = np.dot(inputs, a) + b
    total_loss = 0.5 * np.sum((total_predictions - targets) ** 2)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}')
    print("Learned parameters:")
    print(f'Weights (a): {a}')
    print(f'Bias (b): {b:.4f}')

