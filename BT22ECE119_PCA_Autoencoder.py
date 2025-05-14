import numpy as np
import pandas as pd
from random import random
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from scipy.io import loadmat

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

class ANN:
    def __init__(self, num_inputs=3, num_hidden=[1, 5], num_outputs=2):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        
        layers = [self.num_inputs] + self.num_hidden + [num_outputs]
        
         # Initialize random weights
        self.weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])  # Current layer x Next layer matrix
            self.weights.append(w)

       # Save activations
        self.activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            self.activations.append(a)

        # Save derivatives
        self.derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            self.derivatives.append(d)

    def forward_propagate(self, inputs):
        activations = inputs
        self.activations[0] = inputs  # Input layer

        for i, w in enumerate(self.weights):
            net_inputs = np.dot(activations, w)
            
            # Use ReLU for hidden layers and Softmax for the last layer
            if i == len(self.weights) - 1:
                activations = self.softmax(net_inputs)
            else:
                activations = self.ReLU(net_inputs)
            
            self.activations[i + 1] = activations
        activations.reshape(1,-1)

        return activations

    def back_propagate(self, error, verbose=False):
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i + 1]
            
            if i == len(self.derivatives) - 1:
                delta = error  # Softmax + Cross-Entropy gives simplified gradient
            else:
                delta = error * self.ReLU_derivative(activations)
            
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i]
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
            
            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
            error = np.dot(delta, self.weights[i].T)
            
            if verbose:
                print("Derivatives for W{}: {}".format(i, self.derivatives[i]))

        return error

    def gradient_descent(self, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] += self.derivatives[i] * learning_rate

    def train(self, inputs, targets, val_inputs, val_targets, epochs, learning_rate=0.01):
        train_errors, val_errors = [], []
        
        for epoch in range(epochs):
            sum_error = 0
            for input, target in zip(inputs, targets):
                output = self.forward_propagate(input)
                error = target - output
                self.back_propagate(error)
                self.gradient_descent(learning_rate)
                sum_error += self.mse(target, output)
                
            train_errors.append(sum_error / len(inputs))
            val_errors.append(self.evaluate(val_inputs, val_targets))
            print(f"Epoch {epoch+1}: Training Error={train_errors[-1]}, Validation Error={val_errors[-1]}")
        plt.figure(figsize=(8,3))
        plt.plot(train_errors, label='Training Error')
        plt.plot(val_errors, label='Validation Error')
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.legend()
        plt.title("Training vs Validation Error")
        plt.grid()
        plt.show()

    def evaluate(self, inputs, targets):
        total_error = 0
        for input, target in zip(inputs, targets):
            output = self.forward_propagate(input)
            total_error += self.mse(target, output)
        return total_error / len(inputs)

    def test(self, inputs):
        ret = []
        for i in range(inputs.shape[0]):
            temp = self.forward_propagate(inputs[i, :])
            ret.append(temp)
        return np.array(ret)

    def ReLU(self, x):
        return np.maximum(0, x)

    def ReLU_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def mse(self, target, output):
        return np.average((target - output) ** 2)

class Autoencoder(Model):
    def __init__(self, input_dim=100, hidden_dims=[100, 8, 100]):
        super(Autoencoder, self).__init__()
        self.encoder_layers = [
            Dense(hidden_dims[0], activation='relu'),
            Dense(hidden_dims[1], activation='relu')
        ]
        self.decoder_layers = [
            Dense(hidden_dims[2], activation='relu'),
            Dense(input_dim, activation='linear')  # Linear activation for reconstruction
        ]

    def call(self, x):
        encoded = x
        for layer in self.encoder_layers:
            encoded = layer(encoded)
        
        decoded = encoded
        for layer in self.decoder_layers:
            decoded = layer(decoded)

        return decoded

    def encode(self, x):
        encoded = x
        for layer in self.encoder_layers:
            encoded = layer(encoded)
        return np.array(encoded)

def confusion_matrix(prediction, Y_test):
    TP, TN, FP, FN = 0, 0, 0, 0
    
    for i in range(prediction.shape[0]):
        if prediction[i][0] == 1 and Y_test[i][0] == 1:
            TP += 1
        elif prediction[i][0] == 0 and Y_test[i][0] == 0:
            TN += 1
        elif prediction[i][0] == 1 and Y_test[i][0] == 0:
            FP += 1
        elif prediction[i][0] == 0 and Y_test[i][0] == 1:
            FN += 1
    
    return TP, TN, FP, FN

def load_dataset(file_path):
    if file_path.endswith('.mat'):
        data = loadmat(file_path)
        X = data['data'] 
        y = data['labels'].flatten()
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        X = df.iloc[:, :-1].values  
        y = df.iloc[:, -1].values  
    elif file_path.endswith('.xlsx'):
        dataset = pd.read_excel(file_path)
        X = dataset.iloc[:, 0:-1].values 
        y = dataset.iloc[:, -1].values  
        feature_names = dataset.columns[:-1].tolist()
    else:
        raise ValueError("File not supported")

    return X,y

def our_train_test_split(X,y, train_ratio=0.8, val_ratio=0.2):
    X, y = shuffle(X, y)

    train_len = int(train_ratio * len(X))
    val_len = int(val_ratio * train_len)
    
    X_train, X_val, X_test = X[:train_len-val_len], X[train_len-val_len:train_len], X[train_len:]
    y_train, y_val, y_test = y[:train_len-val_len], y[train_len-val_len:train_len], y[train_len:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

data_path='Ovarian_cancer.xlsx'

if __name__=='__main__':
    plt.ion()

    #Load Data
    X,Y=load_dataset(data_path)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #PCA
    cov_matrix = np.cov(X_scaled, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    n_components = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"Number of components selected: {n_components}")

    W = eigenvectors[:, :n_components]
    X_PCA = X_scaled @ W

    X_train, X_val, X_test, Y_train, Y_val, Y_test= our_train_test_split(X_PCA,Y)
    Y_train = np.hstack((Y_train.reshape(-1, 1), 1 - Y_train.reshape(-1, 1)))
    Y_val = np.hstack((Y_val.reshape(-1, 1), 1 - Y_val.reshape(-1, 1)))
    Y_test = np.hstack((Y_test.reshape(-1, 1), 1 - Y_test.reshape(-1, 1)))

    our_ann=ANN(X_train.shape[1], [32,64], 2)
    our_ann.train(X_train,Y_train,X_val,Y_val,500,0.001)

    z=our_ann.test(X_test)
    prediction = (z == np.max(z, axis=1, keepdims=True)).astype(int)
    Final_MSE=np.mean((prediction-Y_test)**2)
    print("Mean Square Error for test set: ",Final_MSE)

    TP, TN, FP, FN = confusion_matrix(prediction, Y_test)
    print("Confusion Matrix")
    print("TP: ",TP)
    print("TN: ",TN)
    print("FP: ",FP)
    print("FN: ",FN)

    Accuracy=(TP+TN)/(TP+FP+FN+TN)
    Sensitivity=(TP)/(TP+FN)
    Specificity=(TN)/(TN+FP)
    Precission=(TP)/(TP+FP)

    print("Accuracy: ", Accuracy*100, "%")
    print("Sensitivity: ", Sensitivity)
    print("Specificity: ", Specificity)
    print("Precission: ", Precission)

    fpr, tpr, thresholds = roc_curve(Y_test[:,0], prediction[:,0])
    roc_auc = auc(fpr, tpr)


    plt.figure(figsize=(4, 3))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve for PCA (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Random classifier
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

    #Autoencoder
    Num_dimension_PCA=8
    autoencoder = Autoencoder(input_dim=X_scaled.shape[1])
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
    history = autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, verbose=1)

    X_encoded = autoencoder.encode(X_scaled)
    X1_train, X1_val, X1_test, Y1_train, Y1_val, Y1_test = our_train_test_split(X_encoded,Y)
    Y1_train = np.hstack((Y1_train.reshape(-1, 1), 1 - Y1_train.reshape(-1, 1)))
    Y1_val = np.hstack((Y1_val.reshape(-1, 1), 1 - Y1_val.reshape(-1, 1)))
    Y1_test = np.hstack((Y1_test.reshape(-1, 1), 1 - Y1_test.reshape(-1, 1)))

    our_ann=ANN(X1_train.shape[1], [16,16], 2)
    our_ann.train(X1_train,Y1_train,X1_val,Y1_val,300,0.001)

    z1=our_ann.test(X1_test)
    prediction1 = (z1 == np.max(z1, axis=1, keepdims=True)).astype(int)
    Final_MSE=np.mean((prediction1-Y1_test)**2)
    print("Mean Square Error for test set: ",Final_MSE)

    TP1, TN1, FP1, FN1 = confusion_matrix(prediction1, Y1_test)
    print("Confusion Matrix")
    print("TP: ",TP1)
    print("TN: ",TN1)
    print("FP: ",FP1)
    print("FN: ",FN1)

    fpr1, tpr1, thresholds1 = roc_curve(Y1_test[:,0], prediction1[:,0])
    roc_auc1 = auc(fpr1, tpr1)

    plt.figure(figsize=(4, 3))
    plt.plot(fpr1, tpr1, color='blue', label=f'ROC curve for Autoencoder(AUC = {roc_auc1:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Random classifier
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid()

    plt.ioff()  # Disable interactive mode
    plt.show() 

