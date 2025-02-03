import numpy as np

# Hardcoded input and output data
inputs = np.array([1.0, 2.0, 3.0, 4.0])
targets = np.array([2.0, 6.0, 10.0, 12.0])

# Initialize weights and bias
#a = np.random.randn()
#b = np.random.randn()

a = 1.0
b = 0.0

# Learning rate
learning_rate = 0.01

# Number of epochs
num_epochs = 5

# Training loop
for epoch in range(num_epochs):
    # Forward pass: compute predicted outputs
    predictions = a * inputs + b

    # Compute the loss (squared error)
    loss = 0.5 * np.sum((predictions - targets) ** 2)

    # Compute gradients
    da = np.sum(2 * (predictions - targets) * inputs)
    db = np.sum(2 * (predictions - targets))

    # Update weights and bias
    a -= learning_rate * da
    b -= learning_rate * db

    
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}')
    # Print the learned parameters
    print("Learned parameters:")
    print(f'Weight (a): {a:.4f}')
    print(f'Bias (b): {b:.4f}')
    test_input = 6.0
    predicted_output = a * test_input + b
    print(f'Predicted output for input {test_input}: {predicted_output:.4f}')
    print("")



# Test the model
test_input = 6.0
predicted_output = a * test_input + b
print(f'Predicted output for input {test_input}: {predicted_output:.4f}')
print(f'Actual output for input {test_input}: {17}')