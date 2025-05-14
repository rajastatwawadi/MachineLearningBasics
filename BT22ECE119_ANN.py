#imports
import numpy as np
import pandas as pd
from random import random
import time
from scipy.io import loadmat
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

#Train test split
def train_test_split(file_path, train_ratio=0.8, val_ratio=0.2):
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

    X, y = shuffle(X, y)
    
    train_len = int(train_ratio * len(X))
    val_len = int(val_ratio * train_len)
    
    X_train, X_val, X_test = X[:train_len-val_len], X[train_len-val_len:train_len], X[train_len:]
    y_train, y_val, y_test = y[:train_len-val_len], y[train_len-val_len:train_len], y[train_len:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names

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


#ANN class
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
        f1=plt.figure(figsize=(8,6))
        plt.plot(train_errors, label='Training Error')
        plt.plot(val_errors, label='Validation Error')
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.legend()
        plt.title("Training vs Validation Error")
        plt.grid()
        plt.show(block=False)

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

if __name__ == '__main__':

    #Load data
    data_path = 'Ovarian_cancer.xlsx'
    X_train, X_val, X_test, Y_train, Y_val, Y_test, Features = train_test_split(data_path)
    Y_train = np.hstack((Y_train.reshape(-1, 1), 1 - Y_train.reshape(-1, 1)))
    Y_val = np.hstack((Y_val.reshape(-1, 1), 1 - Y_val.reshape(-1, 1)))
    Y_test = np.hstack((Y_test.reshape(-1, 1), 1 - Y_test.reshape(-1, 1)))

    print(X_train.shape)
    print(Y_train.shape)

    print(X_val.shape)
    print(Y_val.shape)

    print(X_test.shape)
    print(Y_test.shape)
    print(len(Features))

    #declare and train ANN
    our_ann=ANN(100, [50,4], 2)
    start1=time.time()
    our_ann.train(X_train,Y_train,X_val,Y_val,300,0.001)
    stop1=time.time()
    print("Time required: ", stop1-start1)

    #Predict result
    z=our_ann.test(X_test)
    prediction = (z == np.max(z, axis=1, keepdims=True)).astype(int)
    Final_MSE=np.mean((prediction-Y_test)**2)   
    print("\nMean Square Error for test set: ",Final_MSE)

    #Confusion matrix
    TP, TN, FP, FN = confusion_matrix(prediction, Y_test)
    print("\nConfusion Matrix")
    print("TP: ",TP)
    print("TN: ",TN)
    print("FP: ",FP)
    print("FN: ",FN)

    #Evaluate parameters
    Accuracy=(TP+TN)/(TP+FP+FN+TN)
    Sensitivity=(TP)/(TP+FN)
    Specificity=(TN)/(TN+FP)
    Precission=(TP)/(TP+FP)

    print("\nAccuracy: ", Accuracy*100, "%")
    print("Sensitivity: ", Sensitivity)
    print("Specificity: ", Specificity)
    print("Precission: ", Precission)


    #Plot ROC curve 
    fpr, tpr, thresholds = roc_curve(Y_test[:,1], z[:,1])
    roc_auc = auc(fpr, tpr)  # Compute AUROC

    f2=plt.figure(figsize=(4, 3))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Random classifier
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show(block=False)

    plt.ioff()
    plt.show()





