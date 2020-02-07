import argparse
import os
import numpy as np
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier  

clf_names = ['SGDClassifier', 
                'GaussianNB', 
                'RandomForestClassifier',
                'MLPClassifier',
                'AdaBoostClassifier']
clfs = [SGDClassifier(loss='hinge'),
        GaussianNB(),
        RandomForestClassifier(max_depth=5, n_estimators=10),
        MLPClassifier(alpha=0.05),
        AdaBoostClassifier()]


def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    diagonal_sum = C.trace()
    sum_of_all_elements = C.sum()
    return 1.0 * diagonal_sum / sum_of_all_elements 


def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    r = []
    for label in range(len(C)):
        row = C[label, :].sum()
        if row != 0:
            r.append(C[label, label] / row)
        else:
            r.append(0.0)
    return r


def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    p = []
    for label in range(len(C)):
      col = C[:, label].sum()
      if col != 0:
        p.append(C[label, label] / col)
      else:
        p.append(0.0)

    return p

def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:
       i: int, the index of the supposed best classifier
    '''
    print('TODO Section 3.1')
    c_matrices = []
    acc_list = []
    for clf in clfs:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        c_temp = confusion_matrix(y_test, y_pred)
        c_matrices.append(c_temp)
        acc_list.append(accuracy(c_temp))

    best_acc = acc_list[0]
    iBest = 0
    for i in range(len(acc_list)):
        if acc_list[i] > best_acc:
            best_acc = acc_list[i]
            iBest = i

    with open(f"{output_dir}/a1_3.1.txt".format(output_dir), "w") as outf:
        # For each classifier, compute results and write the following output:
        for clf_name, clf, c_matrix in zip(clf_names, clfs, c_matrices):
            # outf.write(f'Results for {classifier_name}:\n')  # Classifier name
            outf.write(f'Results for {clf_name}:\n')  # Classifier name
            # outf.write(f'\tAccuracy: {accuracy:.4f}\n')
            outf.write(f'\tAccuracy: {accuracy(c_matrix):.4f}\n')
            # outf.write(f'\tRecall: {[round(item, 4) for item in recall]}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in recall(c_matrix)]}\n')
            # outf.write(f'\tPrecision: {[round(item, 4) for item in precision]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in precision(c_matrix)]}\n')
            # outf.write(f'\tConfusion Matrix: \n{conf_matrix}\n\n')
            outf.write(f'\tConfusion Matrix: \n{c_matrix}\n\n')

        pass

    return iBest


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    print('TODO Section 3.2')
    sizes = [1000, 5000, 10000, 15000, 20000]
    c_matrices = []
    acc_list = []
    X_tr, x, y_tr, x = train_test_split(X_train, y_train, train_size=20000)
    X_1k = X_tr[:1000]
    y_1k = y_tr[:1000]
    for s in sizes:
        #X_train_new = np.random.choice(X_train, size=s, replace=False)
        X_tr_temp =X_tr[:s]
        y_tr_temp = y_tr[:s]
        clfs[iBest].fit(X_tr_temp, y_tr_temp)
        y_pr = clfs[iBest].predict(X_test)
        c_temp = confusion_matrix(y_test, y_pr)
        c_matrices.append(c_temp)
        acc_list.append(accuracy(c_temp))

    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        # For each number of training examples, compute results and write
        for num_train, acc in zip(sizes, acc_list):
            # the following output:
            # outf.write(f'{num_train}: {accuracy:.4f}\n'))
            outf.write(f'{num_train}: {acc:.4f}\n')
        pass

    return (X_1k, y_1k)


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    print('TODO Section 3.3')
    #p_values = []

    
    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        # Prepare the variables with corresponding names, then uncomment
        # this, so it writes them to outf.
        
        # for each number of features k_feat, write the p-values for
        # that number of features:
        for k_feat in {5,50}:
            selector = SelectKBest(f_classif, k_feat)
            X_new = selector.fit_transform(X_train, y_train)
            p_values = selector.pvalues_
            outf.write(f'{k_feat} p-values: {[round(pval, 4) for pval in p_values]}\n')

        selector = SelectKBest(f_classif, 5)
        # outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')
        X_new = selector.fit_transform(X_1k, y_1k)
        selected_cols_1k = selector.get_support(indices=True)
        print("selected_cols_1k : ", selected_cols_1k)
        X_test_new = selector.transform(X_test)

        clfs[i].fit(X_new, y_1k)
        y_pr = clfs[i].predict(X_test_new)
        c = confusion_matrix(y_test, y_pr)
        accuracy_1k = accuracy(c)
        outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')
        selected_cols_1k = selector.get_support(indices=True)
        print("selected_cols_1k : ", selected_cols_1k)

        # outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')
        X_new = selector.fit_transform(X_train, y_train)
        X_test_new = selector.transform(X_test)
        clfs[i].fit(X_new, y_train)
        y_pr = clfs[i].predict(X_test_new)
        c = confusion_matrix(y_test, y_pr)
        accuracy_full = accuracy(c)
        outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')
        selected_cols_full = selector.get_support(indices=True)
        print("selected_cols_full : ", selected_cols_full)

        feature_intersection = list(set(selected_cols_1k).intersection(selected_cols_full))
        outf.write(f'Chosen feature intersection: {feature_intersection}\n')
        outf.write(f'Top-5 at higher: {selected_cols_full}\n')
        pass


def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    print('TODO Section 3.4')
    
    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        # Prepare kfold_accuracies, then uncomment this, so it writes them to outf.
        # for each fold:
        #     outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')
        # outf.write(f'p-values: {[round(pval, 4) for pval in p_values]}\n')
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()
    output_dir = args.output_dir
    
    # TODO: load data and split into train and test.
    data = np.load(args.input)
    lst = data.files
    for item in lst:
        data_list = data[item]
    X = data_list[:,:-1]
    y = data_list[:,-1]
    #for comment in data_list:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    zero = 0
    one = 0
    two = 0
    three =0
    for i in y:
      if i == 0:
        zero += 1
      if i == 1:
        one += 1
      if i == 2:
        two += 1
      if i == 3:
        three += 1
    print(zero)
    print(one)
    print(two)
    print(three)
    print("BESSSSSTTTT")
    #best_clf = class31(output_dir, X_train, X_test, y_train, y_test)
    best_clf = 4

    (X_1k, y_1k) = class32(output_dir, X_train, X_test, y_train, y_test, best_clf)
    # TODO : complete each classification experiment, in sequence.
    class33(output_dir, X_train, X_test, y_train, y_test, best_clf, X_1k, y_1k)