from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, explained_variance_score as evs
from sklearn.metrics import log_loss, mean_squared_log_error as msle
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
    try:
        print("LogLoss = {0:.4f}\n".format(log_loss(y_test, predictions)))
    except:
        print("LogLoss cannot be applied to string.\n")
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
    
    print("RMSE = {0:.4f}\n".format(mse(y_test, predictions, squared=False))) # Root Mean Squared Error
    print("MSE = {0:.4f}\n".format(mse(y_test, predictions, squared=True))) # Mean Squared Error
    
    if len(predictions[predictions<0])>0:
        print("MSLE not possible to be applied. Predicitons contain negative values.\n")
    else:
        print("MSLE = {0:.4f}\n".format(msle(y_test, predictions))) # Mean Squared Log Error
    
    print("MAE = {0:.4f}\n".format(mae(y_test, predictions))) # Mean Absolute Error
    print("EVS = {0:.4%}\n".format(evs(y_test, predictions))) # Explained Variance Score
