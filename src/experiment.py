from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import numpy

# from src.ploting import plot_observations, plot_precision_recall_curves
from ploting import plot_observations, plot_precision_recall_curves


# add function paratemeters that will be needed in the pipeline
def run_experiment(scaler, classifier, split, data, labels, plots):

    # TODO make a check for the path here
    
    X = numpy.load('dataset/{0}'.format(data))
    y = numpy.load('dataset/{0}'.format(labels))

    plot_observations(X, plots['plots_number'])

    
    # X, y = shuffle(X, y, random_state=1)

    steps = [('scaler', scaler), ('classifier',classifier)]
    pipeline = Pipeline(steps)
    print(pipeline.steps)
    # print("Number of neighbors:", pipeline.named_steps['knn'].n_neighbors)
    # print("Number of neighbors:", pipeline.named_steps.knn.n_neighbors)
    if('split_method' not in split):
        return 1
    if(split['split_method'] != 'train_test_split' and split['split_method'] != 'kfolds'):
        return 1
    if(split['split_method'] == 'train_test_split'):
        try:
            int(split['test_size'])
        except ValueError:
            return 1
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=split['test_size'], shuffle=True, random_state=1)

    if(split['split_method'] == 'kfolds'):
        try:
            int(split['kfolds_folds_number'])
        except ValueError:
            return 1
        
        stratified_kfold= StratifiedKFold(n_splits=split['kfolds_folds_number'], shuffle=True, random_state=1)
        for train_index, test_index in stratified_kfold.split(X,y):            
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

    # print(X_train, X_test, y_train, y_test)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    score = pipeline.score(X_test, y_test)
    print('Accuracy: {0}'.format(score))

    mean_f1 = f1_score(y_test, y_pred, average='micro')
    print('F1-Score (micro): {0}'.format(mean_f1))

    mean_f1 = f1_score(y_test, y_pred, average='macro')
    print('F1-Score (macro): {0}'.format(mean_f1))

    mean_f1 = f1_score(y_test, y_pred, average='weighted')
    print('F1-Score (weighted): {0}'.format(mean_f1))

    mean_f1 = f1_score(y_test, y_pred, average=None)
    print('F1-Score (None): {0}'.format(mean_f1))

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    auc_score = auc(recall, precision)
    print('Prcision-Recall AUC: {0}'.format(auc_score))

    plot_precision_recall_curves(recall, precision)

    return score, mean_f1, auc_score
    # TODO should return some values
