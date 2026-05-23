"""
Machine Learning Models implemented from scratch using NumPy
Includes: Logistic Regression, Linear Regression, Softmax Regression, MLP
"""

import numpy as np


class LogisticRegression:
    """
    Logistic Regression classifier implemented from scratch using NumPy.
    Uses sigmoid activation and binary cross-entropy loss.
    
    Forward Propagation: z = w·x + b → σ(z) = 1/(1+e^-z)
    Loss: L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
    Gradient: dw = (1/m)·X^T·(ŷ-y), db = (1/m)·Σ(ŷ-y)
    """
    
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def sigmoid(self, z):
        """Sigmoid activation function: σ(z) = 1/(1+e^-z)"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def forward(self, X):
        """Forward propagation: compute predictions"""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def compute_loss(self, y_pred, y_true):
        """Binary cross-entropy loss: L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]"""
        m = y_true.shape[0]
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def backward(self, X, y_pred, y_true):
        """Backpropagation: compute gradients"""
        m = X.shape[0]
        error = y_pred - y_true
        dw = np.dot(X.T, error) / m
        db = np.sum(error) / m
        return dw, db
    
    def fit(self, X, y):
        """Train the model using gradient descent"""
        m, n = X.shape
        self.weights = np.random.randn(n) * 0.01
        self.bias = 0
        
        for epoch in range(self.epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y_pred, y)
            self.loss_history.append(loss)
            
            # Backward pass
            dw, db = self.backward(X, y_pred, y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs} - Loss: {loss:.6f}")
    
    def predict(self, X):
        """Make predictions"""
        return (self.forward(X) >= 0.5).astype(int)
    
    def predict_proba(self, X):
        """Probability predictions"""
        return self.forward(X)


class LinearRegression:
    """
    Linear Regression implemented from scratch using NumPy.
    Uses Mean Squared Error (MSE) loss and gradient descent.
    
    Forward Propagation: ŷ = w·x + b
    Loss: L = (1/2m)·Σ(ŷ-y)²
    Gradient: dw = (1/m)·X^T·(ŷ-y), db = (1/m)·Σ(ŷ-y)
    """
    
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def forward(self, X):
        """Forward propagation: compute predictions"""
        return np.dot(X, self.weights) + self.bias
    
    def compute_loss(self, y_pred, y_true):
        """Mean Squared Error: L = (1/2m)·Σ(ŷ-y)²"""
        m = y_true.shape[0]
        mse = np.mean((y_pred - y_true) ** 2)
        return mse
    
    def backward(self, X, y_pred, y_true):
        """Backpropagation: compute gradients"""
        m = X.shape[0]
        error = y_pred - y_true
        dw = np.dot(X.T, error) / m
        db = np.sum(error) / m
        return dw, db
    
    def fit(self, X, y):
        """Train the model using gradient descent"""
        m, n = X.shape
        self.weights = np.random.randn(n) * 0.01
        self.bias = 0
        
        for epoch in range(self.epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y_pred, y)
            self.loss_history.append(loss)
            
            # Backward pass
            dw, db = self.backward(X, y_pred, y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs} - MSE Loss: {loss:.6f}")
    
    def predict(self, X):
        """Make predictions"""
        return self.forward(X)


class SoftmaxRegression:
    """
    Softmax Regression (Multi-class Logistic Regression) from scratch.
    Uses softmax activation and categorical cross-entropy loss.
    
    Forward Propagation: z = w·x + b → softmax(z) = e^z / Σe^z
    Loss: L = -(1/m)·Σ Σ y_ij·log(ŷ_ij)
    Gradient: dw = (1/m)·X^T·(ŷ-y), db = (1/m)·Σ(ŷ-y)
    """
    
    def __init__(self, learning_rate=0.01, epochs=100, num_classes=None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.num_classes = num_classes
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def softmax(self, z):
        """Softmax function: σ(z) = e^z / Σe^z"""
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X):
        """Forward propagation: compute softmax predictions"""
        z = np.dot(X, self.weights) + self.bias
        return self.softmax(z)
    
    def compute_loss(self, y_pred, y_true):
        """Categorical cross-entropy loss"""
        m = y_true.shape[0]
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss
    
    def backward(self, X, y_pred, y_true):
        """Backpropagation: compute gradients"""
        m = X.shape[0]
        error = y_pred - y_true
        dw = np.dot(X.T, error) / m
        db = np.sum(error, axis=0) / m
        return dw, db
    
    def one_hot_encode(self, y, num_classes):
        """One-hot encode labels"""
        one_hot = np.zeros((y.shape[0], num_classes))
        one_hot[np.arange(y.shape[0]), y] = 1
        return one_hot
    
    def fit(self, X, y):
        """Train the model using gradient descent"""
        if self.num_classes is None:
            self.num_classes = len(np.unique(y))
        
        m, n = X.shape
        self.weights = np.random.randn(n, self.num_classes) * 0.01
        self.bias = np.zeros((1, self.num_classes))
        
        # One-hot encode labels
        y_one_hot = self.one_hot_encode(y, self.num_classes)
        
        for epoch in range(self.epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y_pred, y_one_hot)
            self.loss_history.append(loss)
            
            # Backward pass
            dw, db = self.backward(X, y_pred, y_one_hot)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs} - Cross-Entropy Loss: {loss:.6f}")
    
    def predict(self, X):
        """Make class predictions"""
        proba = self.forward(X)
        return np.argmax(proba, axis=1)
    
    def predict_proba(self, X):
        """Probability predictions"""
        return self.forward(X)


class MultiLayerPerceptron:
    """
    Multi-Layer Perceptron (Deep Neural Network) implemented from scratch.
    Features:
    - ReLU activation for hidden layers
    - Sigmoid/Softmax for output layer
    - Full backpropagation with chain rule
    - Gradient descent optimization
    - Support for binary and multi-class classification
    """
    
    def __init__(self, hidden_layers=[128, 64], learning_rate=0.01, epochs=100, 
                 task='binary', num_classes=2):
        """
        Parameters:
        - hidden_layers: list of neuron counts per hidden layer
        - learning_rate: learning rate for gradient descent
        - epochs: training iterations
        - task: 'binary' or 'multiclass'
        - num_classes: number of output classes
        """
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.task = task
        self.num_classes = num_classes if task == 'multiclass' else 1
        self.loss_history = []
        self.layers = []
        self.cache = {}
    
    def relu(self, x):
        """ReLU activation: max(0, x)"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        """Sigmoid activation: 1/(1+e^-x)"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid"""
        sig = self.sigmoid(x)
        return sig * (1 - sig)
    
    def softmax(self, x):
        """Softmax activation"""
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def initialize_weights(self, X_shape):
        """Xavier initialization for weights and bias"""
        input_size = X_shape[1]
        layer_sizes = [input_size] + self.hidden_layers
        
        if self.task == 'multiclass':
            layer_sizes.append(self.num_classes)
        else:
            layer_sizes.append(1)
        
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * \
                np.sqrt(2.0 / layer_sizes[i])  # Xavier initialization
            b = np.zeros((1, layer_sizes[i+1]))
            self.layers.append({'w': w, 'b': b})
    
    def forward(self, X):
        """Forward propagation through all layers"""
        self.cache = {'a0': X}
        a = X
        
        # Hidden layers with ReLU
        for i, layer in enumerate(self.layers[:-1]):
            z = np.dot(a, layer['w']) + layer['b']
            a = self.relu(z)
            self.cache[f'z{i+1}'] = z
            self.cache[f'a{i+1}'] = a
        
        # Output layer
        z = np.dot(a, self.layers[-1]['w']) + self.layers[-1]['b']
        self.cache[f'z{len(self.layers)}'] = z
        
        if self.task == 'binary':
            a = self.sigmoid(z)
        else:
            a = self.softmax(z)
        
        self.cache[f'a{len(self.layers)}'] = a
        return a
    
    def compute_loss(self, y_pred, y_true):
        """Compute loss based on task"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Ensure y_true has same shape as y_pred
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
        
        if self.task == 'binary':
            loss = -np.mean(y_true * np.log(y_pred) + 
                           (1 - y_true) * np.log(1 - y_pred))
        else:
            loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss
    
    def backward(self, y_true, y_pred):
        """Backpropagation using chain rule"""
        m = y_true.shape[0]
        gradients = {}
        
        # Ensure y_true has same shape as y_pred for proper broadcasting
        if len(y_true.shape) == 1 and len(y_pred.shape) == 2:
            y_true = y_true.reshape(-1, 1)
        
        # Output layer gradient
        if self.task == 'binary':
            dz = y_pred - y_true
        else:
            dz = y_pred - y_true
        
        # Backpropagate through layers
        for i in range(len(self.layers) - 1, -1, -1):
            a_prev = self.cache[f'a{i}']
            dw = np.dot(a_prev.T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            
            gradients[f'dw{i}'] = dw
            gradients[f'db{i}'] = db
            
            if i > 0:
                # Hidden layer gradient
                da_prev = np.dot(dz, self.layers[i]['w'].T)
                dz = da_prev * self.relu_derivative(self.cache[f'z{i}'])
            else:
                break
        
        return gradients
    
    def update_weights(self, gradients):
        """Update weights using gradient descent"""
        for i in range(len(self.layers)):
            self.layers[i]['w'] -= self.learning_rate * gradients[f'dw{i}']
            self.layers[i]['b'] -= self.learning_rate * gradients[f'db{i}']
    
    def fit(self, X, y):
        """Train the MLP"""
        self.initialize_weights(X.shape)
        
        # Ensure proper shape for labels
        if self.task == 'binary':
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
        else:
            # Handle labels encoding for multiclass
            if len(y.shape) == 1:
                y_one_hot = np.zeros((y.shape[0], self.num_classes))
                y_one_hot[np.arange(y.shape[0]), y.astype(int)] = 1
                y = y_one_hot
        
        for epoch in range(self.epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y_pred, y)
            self.loss_history.append(loss)
            
            # Backward pass
            gradients = self.backward(y, y_pred)
            
            # Update weights
            self.update_weights(gradients)
            
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs} - Loss: {loss:.6f}")
    
    def predict(self, X):
        """Make predictions"""
        y_pred = self.forward(X)
        if self.task == 'binary':
            return (y_pred >= 0.5).astype(int).flatten()
        else:
            return np.argmax(y_pred, axis=1)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.forward(X)
