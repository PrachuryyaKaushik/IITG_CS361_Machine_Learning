import numpy as np

# Hardcoded multi-dimensional input (4 samples, 2 features)
X = np.array([
    [1.0, 2.0], 
    [2.0, 3.0], 
    [3.0, 4.0], 
    [4.0, 5.0]
])  # Shape: (4, 2)

# Target outputs (same number of samples, 1D output)
y = np.array([2.0, 6.0, 10.0, 12.0])  # Shape: (4,)

# Initialize weights (vector) and bias (scalar)
W = np.ones(X.shape[1])  # Initialize weights as ones, shape: (2,)
b = 0.0  # Scalar bias

# Learning rate
learning_rate = 0.01

# Number of epochs
num_epochs = 5

# Training loop
for epoch in range(num_epochs):
    # Forward pass: compute predicted outputs
    predictions = np.dot(X, W) + b  # Matrix multiplication (X @ W) + b

    # Compute the loss (squared error)
    loss = 0.5 * np.sum((predictions - y) ** 2)

    # Compute gradients
    dW = np.dot((predictions - y), X)  # Gradient for W
    db = np.sum(predictions - y)  # Gradient for b

    # Update weights and bias
    W -= learning_rate * dW
    b -= learning_rate * db

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}')
    print("Learned parameters:")
    print(f'Weights (W): {W}')
    print(f'Bias (b): {b:.4f}')
    
    test_input = np.array([6.0, 7.0])  # Example test input (same feature dimension)
    predicted_output = np.dot(test_input, W) + b
    print(f'Predicted output for input {test_input}: {predicted_output:.4f}')
    print("")

# Test the model with a new input
test_input = np.array([6.0, 7.0])  # New test input with 2 features
predicted_output = np.dot(test_input, W) + b
print(f'Predicted output for input {test_input}: {predicted_output:.4f}')
print(f'Expected (approximate) output for input {test_input}: ? (depends on training)')

