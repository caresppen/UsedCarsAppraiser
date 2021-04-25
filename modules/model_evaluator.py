from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def eval_cls(y_test, predictions):
    '''
    Function:
    Evaluates a classification model through its main metrics
    '''
    print("### MEASURES OF CLASSIFICATION MODEL ###")
    print("----------------------------------------\n")
    
    print("Accuracy score = {0:.4%}\n".format(accuracy_score(y_test, predictions)))
    print("Avg Precision score = {0:.4%}\n".format(precision_score(y_test, predictions, average='weighted')))
    print("Recall score = {0:.4%}\n".format(recall_score(y_test, predictions, average='weighted')))
    print("F1 score = {0:.4%}\n".format(f1_score(y_test, predictions, average='weighted')))
    
    cm = confusion_matrix(y_test, predictions)
    print("Confusion matrix:\n{}\n".format(cm))
    
    fig, ax = plt.subplots(figsize=(10,10)) 
    sns.heatmap(cm, annot=True, linewidths=.5, ax=ax);
    

def eval_reg(y_test, predictions):
    '''
    Function:
    Evaluates a regression model through its main metrics
    '''
    print("### MEASURES OF REGRESSION MODEL ###")
    print("------------------------------------\n")
    
    print("Mean Squared Error = {0:.4f}\n".format(mean_squared_error(y_test, predictions)))
    print("Mean Absolute Error (MAE) = {0:.4f}\n".format(mean_absolute_error(y_test, predictions)))
    print("Explained Variance Score = {0:.4%}\n".format(explained_variance_score(y_test, predictions)))
