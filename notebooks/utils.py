
def pickle_to(item, filepath, verbose = True):
    '''Input :
    object to be pickled,
    path of where to pickle
    Output :
    pickled object in said path
    '''
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(item,f)
    if verbose:
        print(f'Sucessfully saved to {filepath}')
    return

def test():
    print('hi')
    
    
def pickle_from(filepath, verbose = True):
    '''Input : filepath to object to be unpickled
    Output: unpickled object'''
    import pickle
    with open(filepath, 'rb') as f:
        item = pickle.load(f)
    if verbose:
        print(f'Loaded file from {filepath}')
    return item


def plot_cmat(model,X_in,y_in):
    '''
    Plots the normalized confusion matrix
    Input: X values and y values
    Output: Normalized confusion matrix for the given X and y
    '''
    import seaborn as sns
    y_pred =model.predict(X_in)
    c_mat = confusion_matrix(y_in,y_pred)
    c_normalized = c_mat.astype('float')/c_mat.sum(axis=1)[:,np.newaxis]
#     print("Normalized Confusion Matrix\n ")
#     print(pd.crosstab(np.array(y_in), y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True))
    plt.figure(figsize=(5, 5))
    ax = plt.subplot()
    sns.heatmap(c_normalized, annot = True, ax = ax, cbar = False)
    ax.set_xlabel('Predicted', fontsize = 16)
    ax.set_ylabel('True', rotation = 0, fontsize = 16)
    ax.yaxis.set_label_coords(-.15, 0.9)
    ax.set_title('Normalized Confusion Matrix', fontsize = 20)
    ax.xaxis.set_ticklabels(['Fake','Real'],fontsize = 12)
    ax.yaxis.set_ticklabels(['Fake','Real'], rotation = 0, fontsize = 12)
    plt.show();
    

def get_accuracy(model,X_in,y_in):
    '''Get the accuracy of performance of a model
    Input: model to be applied, X values and y values
    Output: Accuracy of model in that data
    '''
    import numpy as np
    accuracy = np.mean(cross_val_score(model, X_in, y_in,cv=10))
    return accuracy
                
    
def ignore_warnings():
    import os, warnings
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "default"
    return
