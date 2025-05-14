import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from scipy.io import loadmat


datapath='Accidents_dataset.xlsx'

def train_test_split(file_path,train_ratio=0.8):
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
        X = dataset.iloc[:, 1:-1].values #all except last colomn data
        Y = dataset.iloc[:, -1].values  #last colomn as label
        feature_names = dataset.columns[:-1].tolist()
    else:
        raise ValueError("File not supported")

    X,Y=shuffle(X,Y)

    train_len = int(train_ratio * len(X))
    X_tr, X_ts = X[:train_len], X[train_len:]
    Y_tr, Y_ts = Y[:train_len], Y[train_len:]

    return X_tr, X_ts, Y_tr, Y_ts, feature_names


if __name__=='__main__':

    X_train,X_test,Y_train,Y_test,colomn_names=train_test_split(datapath)
    print("Colomn names : ", colomn_names)

    #Select required data
    X_train_selected=[X_train[:,4],X_train[:,5],X_train[:,8],X_train[:,9]]
    X_test_selected=[X_test[:,4],X_test[:,5],X_test[:,8],X_test[:,9]]

    X_train_selected=(np.array(X_train_selected)).T
    X_train_selected = np.column_stack([np.ones((40, 1)), X_train_selected])

    X_test_selected=(np.array(X_test_selected)).T
    X_test_selected = np.column_stack([np.ones((11, 1)), X_test_selected])
    Theta=X_train_selected

    print(X_train_selected.shape)
    print(X_test_selected.shape)

    #Pseudo inverse + prediction
    ThetaTrans_Theta_inv=np.linalg.inv(np.matmul(Theta.T,Theta))
    Weight_matrix=np.matmul(np.matmul(ThetaTrans_Theta_inv,Theta.T),Y_train)
    print("Weight matrix shape - ", Weight_matrix.shape)

    Y_predict=np.matmul(X_test_selected,Weight_matrix)
    print("Y_predict : ", Y_predict)
    print("Y_test : ", Y_test)

    fig1=plt.figure()
    plt.plot(Y_test,Y_test,label='Actual values')
    plt.scatter(Y_test,Y_predict,color='orange',label='Predicted values')
    plt.xlabel("Y test")
    plt.ylabel("Y predicted")
    plt.title("Y test Vs Y predicted")
    plt.legend()

    print("Weight matrix - ", Weight_matrix)

    mse=np.mean((Y_predict-Y_test)**2)
    print("Mean squared error for test data using pseudo - ",mse)

    #Gradient Descent
    M=len(X_train_selected)
    No_iterations=30
    step_size=0.001

    sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, learning_rate='optimal', eta0=0.01)
    sgd_reg.fit(X_train_selected, Y_train)  

    Weight_mat_gd = sgd_reg.coef_ 
    Y_pred=np.matmul(X_test_selected,Weight_mat_gd)

    mse2=np.mean((Y_pred-Y_test)**2)
    print("MSE for gradient descent - ", mse2)

    fig2=plt.figure()
    plt.plot(Y_test,Y_test,label='Actual values')
    plt.scatter(Y_test,Y_pred,color='orange',label='Predicted values')
    plt.xlabel("Y test")    
    plt.ylabel("Y predicted")
    plt.title("Y test Vs Y predicted")
    plt.legend()

