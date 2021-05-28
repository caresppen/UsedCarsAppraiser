import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

def low_corr_matrix(df):
    '''
    Function:
    Plot the lower half of a correlation matrix in a nice format 
    
    Parameters:
    * df = input DataFrame for the plot
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
    
    return matrix


def chang_hug_map(X, hex_colors, FONT_SIZE=12, BINS=30):
    '''
    Function that applies Chang & Hug map of preprocessing data to a normal distribution:
    REF: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_map_data_to_normal.html#sphx-glr-auto-examples-preprocessing-plot-map-data-to-normal-py
    
    Parameters:
    * X = features
    * hex_colors = hexadecimal colors to be used for each feature
    * FONT_SIZE = size of font on plots
    * BINS = number of bins on histogram plots
    '''
    # setting preprocessing methods: PowerTransformer (Box-Cox, Yeo-Johnson); QuantileTransformer
    scaler = MinMaxScaler(feature_range=(1, 2))
    boxcox = PowerTransformer(method='box-cox')
    bc = Pipeline(steps=[('s', scaler), ('bc', boxcox)])

    yj = PowerTransformer(method='yeo-johnson')

    rng = np.random.RandomState(304)
    qt = QuantileTransformer(n_quantiles=500, output_distribution='normal',
                             random_state=rng)

    # adding distributions of columns
    distributions = []
    for i in range(0, len(X.columns)):
        name = X.columns[i]
        array = X[X.columns[i]].to_numpy().reshape(-1,1)
        distributions.append((name, array))

    colors = hex_colors

    # generating the plot
    fig, axes = plt.subplots(nrows=8, ncols=15, figsize=(35, 20)) # cols = num of preprocessing methods + original
    axes = axes.flatten()
    axes_idxs = [(0, 15, 30, 45), (1, 16, 31, 46), (2, 17, 32, 47), (3, 18, 33, 48), (4, 19, 34, 49), (5, 20, 35, 50),    # first set
                 (6, 21, 36, 51), (7, 22, 37, 52), (8, 23, 38, 53), (9, 24, 39, 54), (10, 25, 40, 55), (11, 26, 41, 56),
                 (12, 27, 42, 57), (13, 28, 43, 58), (14, 29, 44, 59),
                 (60, 75, 90, 105), (61, 76, 91, 106), (62, 77, 92, 107), (63, 78, 93, 108), (64, 79, 94, 109), (65, 80, 95, 110),    # second set
                 (66, 81, 96, 111), (67, 82, 97, 112), (68, 83, 98, 113), (69, 84, 99, 114), (70, 85, 100, 115), (71, 86, 101, 116),
                 (72, 87, 102, 117), (73, 88, 103, 118), (74, 89, 104, 119)]

    axes_list = [(axes[i], axes[j], axes[k], axes[l])
                 for (i, j, k, l) in axes_idxs]

    for distribution, color, axes in zip(distributions, colors, axes_list):
        name, X_col = distribution
        X_train, X_test = train_test_split(X_col, test_size=0.2, random_state=rng)

        # perform power and quantile transforms
        X_trans_bc = bc.fit(X_train).transform(X_test)
        lmbda_bc = round(bc.named_steps['bc'].lambdas_[0], 2)
        X_trans_yj = yj.fit(X_train).transform(X_test)
        lmbda_yj = round(yj.lambdas_[0], 2)
        X_trans_qt = qt.fit(X_train).transform(X_test)

        ax_original, ax_bc, ax_yj, ax_qt = axes

        ax_original.hist(X_train, color=color, bins=BINS)
        ax_original.set_title(name, fontsize=FONT_SIZE)
        ax_original.tick_params(axis='both', which='major', labelsize=FONT_SIZE)

        for ax, X_trans, meth_name, lmbda in zip(
                                                (ax_bc, ax_yj, ax_qt),
                                                (X_trans_bc, X_trans_yj, X_trans_qt),
                                                ('Box-Cox', 'Yeo-Johnson', 'Quartile transform'),
                                                (lmbda_bc, lmbda_yj, None)
                                                ):
            ax.hist(X_trans, color=color, bins=BINS)
            title = f'After {meth_name}'
            if lmbda is not None:
                title += f'\n$\lambda$ = {lmbda}'
            ax.set_title(title, fontsize=FONT_SIZE)
            ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
            ax.set_xlim([-3.5, 3.5])

    # Setting last plot as empty
    ax_original, ax_bc, ax_yj, ax_qt = axes_list[-1]
    ax_original.axis('off')
    ax_bc.axis('off')
    ax_yj.axis('off')
    ax_qt.axis('off')

    # Export and last adjustments
    plt.tight_layout()
    plt.savefig('fig/09_col_trf.png')
    plt.show()
    

def autolabel(rects, ax):
    '''
    Function that aligns text labels at center-up in of each bar in a barplot.
    
    Parameter:
    * rects = barplot as an argument
    '''
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0,3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom")
    
def plot_model_eval(df, results):
    '''
    Create an evaluation plot. Contains a lineplot, barplot & boxplot that describe the performance of each model.
    
    Parameters:
    * df = DataFrame to be explored
    * results = resulting iterated values from the evaluated metric (i.e. R2 score)
    '''
    plt.style.use('tableau-colorblind10')
    
    fig = plt.figure(figsize=(15, 10))

    ax_it = plt.subplot2grid(shape=(2,2), loc=(0, 0), colspan=2)
    ax_mean = plt.subplot2grid(shape=(2,2), loc=(1, 0), colspan=1)
    ax_box = plt.subplot2grid(shape=(2,2), loc=(1, 1), colspan=1)
    axes = [ax_it, ax_mean, ax_box]

    # Add K-Fold plot
    sns.lineplot(data=results, dashes=True, ax=ax_it)

    ax_it.set_title('Regression models cv performance per K-Fold')
    ax_it.set_xlabel('K-Fold iteration')
    ax_it.set_ylabel('$R^2$ score')

    ax_it.set_xticks(np.arange(0, 10))
    ax_it.set_xticklabels(tuple([str(i) for i in range(1,11)]))
    ax_it.set_xlim([0, 9])

    ax_it.legend(df.model.to_list())

    # Add mean barplot
    rect1 = ax_mean.bar(df['model'], df['mean_r2_score'])
    ax_mean.set_ylim([0, 1])

    ax_mean.set_title('Mean $R^2$ score of Regression Models')
    ax_mean.set_xlabel('Model')
    ax_mean.set_ylabel('$R^2$ score')
    
    autolabel(rect1, ax=ax_mean)

    # Add boxplot
    ax_box.boxplot(results)

    ax_box.set_title('Regression Algorithms: Distribution Comparison')
    ax_box.set_xlabel('Model')
    ax_box.set_ylabel('$R^2$ score')
    
    ax_box.set_xticklabels(df.model)

    
def plot_search_results(grid, search):
    """
    Parameters: 
        * grid = A trained GridSearchCV object.
        * search = {'RandomSearchCV', 'GridSearchCV'} method used on the search grid.
    """
    ## Results from grid search
    results = grid.cv_results_
    means_test = results['mean_test_score']
    stds_test = results['std_test_score']
    means_train = results['mean_train_score']
    stds_train = results['std_train_score']

    ## Getting indexes of values per hyper-parameter
    masks=[]
    masks_names= list(grid.best_params_.keys())
    for p_k, p_v in grid.best_params_.items():
        masks.append(list(results['param_'+p_k].data==p_v))
    
    if search == 'RandomSearchCV':
        params=grid.param_distributions
    elif search == 'GridSearchCV':
        params=grid.param_grid
    else:
        print("Use a 'search' method contained in this list: ['RandomSearchCV', 'GridSearchCV']")

    ## Ploting results
    fig, ax = plt.subplots(1,len(params),sharex='none', sharey='all',figsize=(20,5))
    fig.suptitle('Score per parameter')
    fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')
    pram_preformace_in_best = {}
    for i, p in enumerate(masks_names):
        m = np.stack(masks[:i] + masks[i+1:])
        pram_preformace_in_best
        best_parms_mask = m.all(axis=0)
        best_index = np.where(best_parms_mask)[0]
        x = np.array(params[p])
        y_1 = np.array(means_test[best_index])
        e_1 = np.array(stds_test[best_index])
        y_2 = np.array(means_train[best_index])
        e_2 = np.array(stds_train[best_index])
        
        x = [np.inf if i==None else i for i in x]

        ax[i].errorbar(x, y_1, e_1, linestyle='--', marker='o', label='test')
        ax[i].errorbar(x, y_2, e_2, linestyle='-', marker='^',label='train' )
        ax[i].set_xlabel(p.upper())

    plt.legend(bbox_to_anchor = (-6.2, 0.6))