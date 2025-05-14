import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.utils import shuffle
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

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

data_path = 'Fisheriris_dataset.xlsx'

if __name__=='__main__':
    plt.ion()
    #Load Data
    X_train, X_val, X_test, Y_train, Y_val, Y_test, Features = train_test_split(data_path,train_ratio=0.8,val_ratio=0.0)

    for i in range(len(Y_train)):
        if Y_train[i]=='virginica':
            Y_train[i]=0
        elif Y_train[i]=='versicolor':
            Y_train[i]=1
        else:
            Y_train[i]=2
        
    for i in range(len(Y_test)):
        if Y_test[i]=='virginica':
            Y_test[i]=0
        elif Y_test[i]=='versicolor':
            Y_test[i]=1
        else:
            Y_test[i]=2
        
    for i in range(len(Y_val)):
        if Y_val[i]=='virginica':
            Y_val[i]=0
        elif Y_val[i]=='versicolor':
            Y_val[i]=1
        else:
            Y_val[i]=2

    Y_train = Y_train.astype(int)
    Y_test = Y_test.astype(int)
    Y_val = Y_val.astype(int)

    Y_train_bin = label_binarize(Y_train, classes=[0, 1, 2])
    Y_test_bin = label_binarize(Y_test, classes=[0, 1, 2])
    #Y_val_bin = label_binarize(Y_val, classes=[0,1,2])
    n_classes = Y_train_bin.shape[1]

    #Linear Kernal
    start1=time.time()

    param_grid1 = {
        'linear': {'C': [0.1, 1, 10, 100]}
    }
    svm1=SVC(kernel='linear',probability=True)
    print("\nPerforming Grid Search for Linear kernel...")

    grid_search1 = GridSearchCV(svm1, param_grid1['linear'], cv=5, scoring='accuracy', n_jobs=-1, return_train_score=True)
    grid_search1.fit(X_train, Y_train)
    stop1=time.time() 
    
    best_model1 = grid_search1.best_estimator_
    result1 = grid_search1.cv_results_
    C_values1 = [params['C'] for params in result1['params']] #variable names
    train_accuracy1 = result1['mean_train_score']
    test_accuracy1 = result1['mean_test_score']
    test_error1 = 1 - np.array(test_accuracy1)  # Error = 1 - accuracy

    f1=plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(C_values1, test_accuracy1, 'o-', label='Test Accuracy')
    plt.plot(C_values1, train_accuracy1, 's--', label='Train Accuracy')
    plt.xlabel('C Value')
    plt.ylabel('Accuracy')
    plt.xscale('log')
    plt.title('Accuracy vs C Linear kernel)')
    plt.legend()
        
    # Plot Error vs C
    plt.subplot(1, 2, 2)
    plt.plot(C_values1, test_error1, 'o-', color='red', label='Test Error')
    plt.xlabel('C Value')
    plt.ylabel('Error')
    plt.xscale('log')
    plt.title('Error vs C Linear kernel)')
    plt.legend()  
    plt.show(block=False)
  


    print("\nEvaluating Linear kernel...")

    #Predictions
    y_train_pred1 = best_model1.predict(X_train)
    y_test_pred1 = best_model1.predict(X_test)

    #Accuracy
    train_acc1 = best_model1.score(X_train, Y_train)
    test_acc1 = best_model1.score(X_test, Y_test)

    CM1 = confusion_matrix(Y_test, y_test_pred1)
    sensitivity1 = np.diag(CM1) / np.sum(CM1, axis=1)  # Recall
    specificity1 = np.diag(CM1) / np.sum(CM1, axis=0)  # True Negative Rate

    ovr_svm1 = OneVsRestClassifier(best_model1)
    ovr_svm1.fit(X_train, Y_train_bin)
    y_score1 = ovr_svm1.decision_function(X_test)

    fpr1, tpr1, roc_auc1 = {}, {}, {}

    f2=plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr1[i], tpr1[i], _ = roc_curve(Y_test_bin[:, i], y_score1[:, i])
        roc_auc1[i] = auc(fpr1[i], tpr1[i])
        plt.plot(fpr1[i], tpr1[i], label=f'Class {i} (AUC = {roc_auc1[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label="Random (AUC = 0.5)")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Linear Kernel')
    plt.legend(loc='lower right')
    plt.show(block=False)

    print("\nResults for Linear kernel:")
    print("  Time Required: ",stop1-start1)
    print("  Train Accuracy: ",100*train_acc1)
    print("  Test Accuracy: ",100*test_acc1)
    print("  Sensitivity (per class): ", sensitivity1)
    print("  Specificity (per class): ", specificity1)
    print("  AUROC (per class): ",roc_auc1)

    #Poly Kernel
    start2=time.time()
    param_grid2 = {
        'poly': {'C': [0.1, 1, 10, 100], 'degree': [2, 3, 4], 'gamma': ['scale', 'auto']}
    }
    svm2=SVC(kernel='poly',probability=True)
    print("\nPerforming Grid Search for Poly kernel...")

    grid_search2 = GridSearchCV(svm2, param_grid2['poly'], cv=5, scoring='accuracy', n_jobs=-1, return_train_score=True)
    grid_search2.fit(X_train, Y_train) 
    stop2=time.time()

    best_model2 = grid_search2.best_estimator_
    result2 = grid_search2.cv_results_

    C_values2 = [params['C'] for params in result2['params']] #variable names
    train_accuracy2 = result2['mean_train_score']
    test_accuracy2 = result2['mean_test_score']
    test_error2 = 1 - np.array(test_accuracy2)  # Error = 1 - accuracy

    f3=plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(C_values2, test_accuracy2, 'o-', label='Test Accuracy')
    plt.plot(C_values2, train_accuracy2, 's--', label='Train Accuracy')
    plt.xlabel('C Value')
    plt.ylabel('Accuracy')
    plt.xscale('log')
    plt.title('Accuracy vs C (Poly kernel)')
    plt.legend()
        
    # Plot Error vs C
    plt.subplot(1, 2, 2)
    plt.plot(C_values2, test_error2, 'o-', color='red', label='Test Error')
    plt.xlabel('C Value')
    plt.ylabel('Error')
    plt.xscale('log')
    plt.title('Error vs C (Poly kernel)')
    plt.legend()  
    plt.show(block=False)




    print("\nEvaluating Poly kernel...")

    #Predictions
    y_train_pred2 = best_model2.predict(X_train)
    y_test_pred2 = best_model2.predict(X_test)

    #Accuracy
    train_acc2 = best_model2.score(X_train, Y_train)
    test_acc2 = best_model2.score(X_test, Y_test)

    CM2 = confusion_matrix(Y_test, y_test_pred2)
    sensitivity2 = np.diag(CM2) / np.sum(CM2, axis=1)  # Recall
    specificity2 = np.diag(CM2) / np.sum(CM2, axis=0)  # True Negative Rate

    ovr_svm2 = OneVsRestClassifier(best_model2)
    ovr_svm2.fit(X_train, Y_train_bin)
    y_score2 = ovr_svm2.decision_function(X_test)

    fpr2, tpr2, roc_auc2 = {}, {}, {}

    f4=plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr2[i], tpr2[i], _ = roc_curve(Y_test_bin[:, i], y_score2[:, i])
        roc_auc2[i] = auc(fpr2[i], tpr2[i])
        plt.plot(fpr2[i], tpr2[i], label=f'Class {i} (AUC = {roc_auc2[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label="Random (AUC = 0.5)")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Poly Kernel')
    plt.legend(loc='lower right')
    plt.show(block=False)

    print("\nResults for Poly kernel:")
    print("  Time Required: ",stop2-start2)
    print("  Train Accuracy: ",100*train_acc2)
    print("  Test Accuracy: ",100*test_acc2)
    print("  Sensitivity (per class): ", sensitivity2)
    print("  Specificity (per class): ", specificity2)
    print("  AUROC (per class): ",roc_auc2)

    
    #RBF kernel
    start3=time.time()
    param_grid3 = {
        'rbf': {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 'scale', 'auto']}
    }
    svm3=SVC(kernel='rbf',probability=True)
    print("\nPerforming Grid Search for RBF kernel...")

    grid_search3 = GridSearchCV(svm3, param_grid3['rbf'], cv=5, scoring='accuracy', n_jobs=-1, return_train_score=True)
    grid_search3.fit(X_train, Y_train) 
    stop3=time.time()

    best_model3 = grid_search3.best_estimator_
    result3 = grid_search3.cv_results_

    C_values3 = [params['C'] for params in result3['params']] #variable names
    train_accuracy3 = result3['mean_train_score']
    test_accuracy3 = result3['mean_test_score']
    test_error3 = 1 - np.array(test_accuracy3)  # Error = 1 - accuracy

    f5=plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(C_values3, test_accuracy3, 'o-', label='Test Accuracy')
    plt.plot(C_values3, train_accuracy3, 's--', label='Train Accuracy')
    plt.xlabel('C Value')
    plt.ylabel('Accuracy')
    plt.xscale('log')
    plt.title('Accuracy vs C (RBF kernel)')
    plt.legend()
        
    # Plot Error vs C
    plt.subplot(1, 2, 2)
    plt.plot(C_values3, test_error3, 'o-', color='red', label='Test Error')
    plt.xlabel('C Value')
    plt.ylabel('Error')
    plt.xscale('log')
    plt.title('Error vs C (RBF kernel)')
    plt.legend()  
    plt.show(block=False)




    print("\nEvaluating RBF kernel...")

    #Predictions
    y_train_pred3 = best_model3.predict(X_train)
    y_test_pred3 = best_model3.predict(X_test)

    #Accuracy
    train_acc3 = best_model3.score(X_train, Y_train)
    test_acc3 = best_model3.score(X_test, Y_test)

    CM3 = confusion_matrix(Y_test, y_test_pred3)
    sensitivity3 = np.diag(CM3) / np.sum(CM3, axis=1)  # Recall
    specificity3 = np.diag(CM3) / np.sum(CM3, axis=0)  # True Negative Rate

    ovr_svm3 = OneVsRestClassifier(best_model3)
    ovr_svm3.fit(X_train, Y_train_bin)
    y_score3 = ovr_svm3.decision_function(X_test)

    fpr3, tpr3, roc_auc3 = {}, {}, {}

    f6=plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr3[i], tpr3[i], _ = roc_curve(Y_test_bin[:, i], y_score3[:, i])
        roc_auc3[i] = auc(fpr3[i], tpr3[i])
        plt.plot(fpr3[i], tpr3[i], label=f'Class {i} (AUC = {roc_auc3[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label="Random (AUC = 0.5)")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Poly Kernel')
    plt.legend(loc='lower right')
    plt.show(block=False)

    print("\nResults for RBF kernel:")
    print("  Time Required: ",stop3-start3)
    print("  Train Accuracy: ",100*train_acc3)
    print("  Test Accuracy: ",100*test_acc3)
    print("  Sensitivity (per class): ", sensitivity3)
    print("  Specificity (per class): ", specificity3)
    print("  AUROC (per class): ",roc_auc3)

    plt.ioff()  # Disable interactive mode
    plt.show() 

        
  

    
    