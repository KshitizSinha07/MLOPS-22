"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause
from sklearn import datasets, svm, metrics
from sklearn import tree 
import pdb
import statistics
from utils import (
    preprocess_digits,
    train_dev_test_split,
    h_param_tuning,
    data_viz,
    pred_image_viz,
    get_all_h_param_comb,
    tune_and_save,
    test_classifier_bias,
    test_classifier_all_class,
)



from joblib import dump, load

train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
assert train_frac + dev_frac + test_frac == 1.0

# 1. set the ranges of hyper parameters
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

params = {}
params["gamma"] = gamma_list
params["C"] = c_list

h_param_comb_SVM = get_all_h_param_comb(params)
h_param_comb_DT=[{'criterion':"gini"},{'criterion':"entropy"},{'criterion':"log_loss"}]

# PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)
# housekeeping
del digits
file1 = open("mREADME.md", "a")
svm_model_metric=[]
DT_model_metric=[]
print("run", "svm", "decision_tree")
file1.write("run"+ " svm"+  " decision_tree"+"\n")
for n_runs in range(5):
    
    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
    data, label, train_frac, dev_frac)   

    # PART: Define the model
    # Create a classifier: a support vector classifier
    clf = svm.SVC()
    # define the evaluation metric
    metric = metrics.accuracy_score
    SVM_best_model, SVM_best_metric, SVM_best_h_params = h_param_tuning(
        h_param_comb_SVM, clf, x_train, y_train, x_dev, y_dev, metric
    )
    predicted = SVM_best_model.predict(x_test)
    SVM_test_metric = metric(y_pred=predicted, y_true=y_test)

    clf_DT=tree.DecisionTreeClassifier()
    DT_best_model, DT_best_metric, DT_best_h_params = h_param_tuning(
        h_param_comb_DT, clf_DT, x_train, y_train, x_dev, y_dev, metric
    )
    DTpredicted = DT_best_model.predict(x_test)
    DT_test_metric = metric(y_pred=DTpredicted, y_true=y_test)

    svm_model_metric.append(SVM_test_metric)
    DT_model_metric.append(DT_test_metric)
    print(n_runs, SVM_test_metric,DT_test_metric)
    file1.writelines(str(n_runs)+' '+ str(SVM_test_metric)+' '+str(DT_test_metric)+"\n")


SVM_metric_mean=statistics.mean(svm_model_metric)
DT_metric_mean=statistics.mean(DT_model_metric)
SVM_metric_stdev=statistics.stdev(svm_model_metric)
DT_metric_stdev=statistics.stdev(DT_model_metric)

print("mean", SVM_metric_mean,DT_metric_mean)
print("std", SVM_metric_stdev,DT_metric_stdev)
file1.write("mean"+' '+str( SVM_metric_mean)+' '+str(DT_metric_mean)+"\n")
file1.write("std"+ ' '+str(SVM_metric_stdev)+' '+str(DT_metric_stdev)+"\n")
file1.close()
"""
# 2. load the best_model
best_model = load(actual_model_path)

# PART: Get test set predictions
# Predict the value of the digit on the test subset
predicted = best_model.predict(x_test)

pred_image_viz(x_test, predicted)

# 4. report the test set accurancy with that best model.
# PART: Compute evaluation metrics
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)
"""