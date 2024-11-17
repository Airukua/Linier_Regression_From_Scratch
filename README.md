# Linear Regression from Scratch

This project demonstrates how to implement a linear regression model from scratch using Python. The notebook walks through the essential steps, including data preparation, model definition, training, and visualisation of results.

## Features

- **Custom Linear Regression Model**: Implementation of linear regression without relying on pre-built machine learning libraries.
- **Gradient Descent Optimisation**: Training the model using gradient descent to minimise mean squared error (MSE).
- **Data Splitting**: Splitting data into training and testing sets using `scikit-learn`.
- **Visualisation**: Visualising model loss over iterations.

## Requirements

The following Python libraries are required to run the notebook:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## Implementation Overview

1. **Data Generation**: Randomly generated feature (`X`) and target (`y`) data are used for training.
2. **Custom Model Definition**:
    - Weights and bias are initialised randomly.
    - Prediction is calculated using the formula: \( y = XW + b \).
3. **Loss Calculation**: Mean Squared Error (MSE) is used as the loss function.
4. **Gradient Descent**: Optimisation of weights and bias using gradient descent.
5. **Training and Visualisation**: Training the model over multiple epochs and plotting the convergence of loss.

## Usage

1. Clone or download the repository.
2. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
3. Open the Jupyter Notebook and run the cells sequentially.

## Visualisation

The notebook includes a visualisation of the loss convergence over iterations, providing insight into the effectiveness of the gradient descent algorithm.

## Key Code Snippets

### Linear Regression Class
```python
class Linear_Regression:
    def __init__(self, X):
        self.num_feature = X.shape[1]
        self.weight = np.random.rand(self.num_feature, 1)
        self.bias = np.random.randn(1)
        self.X = X

    def predict(self):
        return np.dot(self.X, self.weight) + self.bias
```

### Gradient Descent
```python
# Training loop
for i in range(num_epoch):
    y_pred = model.predict()

    # Calculate loss using MSE
    loss = 1/(2 * len(y_train)) * np.sum((y_train - y_pred)**2)
    loss_history.append(loss)

    # Update weights and bias
    model.weight -= learning_rate * gradient_weights
    model.bias -= learning_rate * gradient_bias
```

## Results

- The model demonstrates the convergence of loss, as evidenced by the line plot showing MSE over epochs.

## Contributing

Feel free to contribute by improving the model, adding new features, or optimising the implementation.
