import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def low_corr_matrix(df):
    '''
    Function:
    Plot the lower half of a correlation matrix in a nice format 
    
    Parameters:
    df = input DataFrame for the plot
    '''
    # Calculate pairwise-correlation
    matrix = df.corr()

    # Create a mask
    mask = np.triu(np.ones_like(matrix, dtype=bool))

    # Create a custom divergin palette
    cmap = sns.diverging_palette(250, 15, s=75, l=40,
                                n=9, center="light", as_cmap=True)

    plt.figure(figsize=(16, 12))
    sns.heatmap(matrix, mask=mask, center=0, annot=True,
                fmt='.2f', square=True, cmap=cmap)

    plt.show()
    
    return matrix