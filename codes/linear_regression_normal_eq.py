import numpy as np

# Regularization parameter (lambda)
lambda_reg = 0.1

# Training data
inputs = np.array([1.0, 2.0, 3.0, 4.0])
targets = np.array([2.0, 6.0, 10.0, 12.0])

# Add a column of ones to the inputs to account for the bias term
X = np.vstack((np.ones(inputs.shape[0]), inputs)).T
y = targets

# Compute the analytical solution using the normal equation
XtX = X.T @ X
det = np.linalg.det(XtX)

if np.abs(det) > 1e-10:  # Check if the determinant is not close to zero
    theta = np.linalg.inv(XtX) @ X.T @ y
else:
    theta = np.linalg.pinv(X) @ y

# Extract the learned parameters
b = theta[0]
a = theta[1]

# Print the learned parameters
print("Learned parameters:")
print(f'Weight (a): {a:.4f}')
print(f'Bias (b): {b:.4f}')

# Test prediction
test_input = 6.0
predicted_output = a * test_input + b
print(f'Predicted output for input {test_input}: {predicted_output:.4f}')