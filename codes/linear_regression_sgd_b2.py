import numpy as np

# Example data (inputs and targets)
inputs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
targets = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

# Initialize weights and bias
a = 1.0
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
        batch_inputs = inputs[i:i + batch_size]
        batch_targets = targets[i:i + batch_size]

        # Initialize gradients
        da = 0
        db = 0

        # Compute gradients for the batch
        for j in range(len(batch_inputs)):
            # Forward pass: compute predicted output
            prediction = a * batch_inputs[j] + b

            # Compute the loss (squared error)
            loss = 0.5 * (prediction - batch_targets[j]) ** 2

            # Accumulate gradients
            da += (prediction - batch_targets[j]) * batch_inputs[j]
            db += (prediction - batch_targets[j])

        # Average gradients
        #da /= batch_size
        #db /= batch_size

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