from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, explained_variance_score as evs
from sklearn.metrics import log_loss, mean_squared_log_error as msle, r2_score
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import pandas as pd
import numpy as np

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
    
    print("R2 = {0:.4f}\n".format(r2_score(y_test, predictions))) # R2
    print("RMSE = {0:.4f}\n".format(mse(y_test, predictions, squared=False))) # Root Mean Squared Error
    print("MSE = {0:.4f}\n".format(mse(y_test, predictions, squared=True))) # Mean Squared Error
    
    if len(predictions[predictions<0])>0:
        print("MSLE not possible to be applied. Predicitons contain negative values.\n")
    else:
        print("MSLE = {0:.4f}\n".format(msle(y_test, predictions))) # Mean Squared Log Error
    
    print("MAE = {0:.4f}\n".format(mae(y_test, predictions))) # Mean Absolute Error
    print("EVS = {0:.4%}\n".format(evs(y_test, predictions))) # Explained Variance Score

    
def run_cv_reg(features, target, models, evaluator='r2'):
    '''
    Function that runs the cross-validation (CV) for the named algorithm
    
    Parameters:
    * models = List of Tuples: (name, model). Algorithms to be applied.
    * evaluator = metric used to evaluate the model (predefined: r2_score).
    '''
    # Set seed to obtain the same random numbers
    seed = 42

    # Evaluate each model
    names = []
    results = []
    mins = []
    quartiles_1 = []
    medians = []
    means = []
    stds = []
    quartiles_3 = []
    maxs = []
    times = []
    scoring = evaluator
    
    # Executing the function for every model in the list: models
    for name, model in models:
        # set start time
        print(f'Executing {name}...')
        start_time = time()

        kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
        cv_results = model_selection.cross_val_score(estimator=model,
                                                     X=features,
                                                     y=target,
                                                     cv=kfold,
                                                     scoring=scoring)
        # appending stats to lists
        names.append(name)
        results.append(cv_results)
        mins.append(cv_results.min())
        quartiles_1.append(np.percentile(cv_results, 25)) # Q1
        medians.append(np.median(cv_results)) # Q2 = median
        means.append(cv_results.mean())
        stds.append(cv_results.std())
        quartiles_3.append(np.percentile(cv_results, 75)) # Q3
        maxs.append(cv_results.max())

        # set end time: execution time
        exec_time = time() - start_time

        # Appending to the main list
        times.append(exec_time)
        print(f'CV finished for {name}')
        
    # Creating a DataFrame to see the performance of each model:
    df_models = pd.DataFrame({'model': names,
                              'min_r2_score': mins,
                              '1st_quantile': quartiles_1,
                              'median_r2_score': medians,
                              'mean_r2_score': means,
                              'std_r2_score': stds,
                              '3rd_quantile': quartiles_3,
                              'max_r2_score': maxs,
                              'exec_time_sec': times})
    # Rounding to 4 decimals
    round_cols = dict(zip(df_models.columns, len(df_models.columns)*[4]))
    df_models = df_models.round(round_cols)
    
    return (df_models, results)