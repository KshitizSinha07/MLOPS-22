import sys, os
import numpy as np
from joblib import load

sys.path.append(".")

from mlops.utils import get_all_h_param_comb, tune_and_save, train_dev_test_split
from sklearn import svm, metrics

# test case to check if all the combinations of the hyper parameters are indeed getting created
def test_get_h_param_comb():
    gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

    params = {}
    params["gamma"] = gamma_list
    params["C"] = c_list
    h_param_comb = get_all_h_param_comb(params)

    assert len(h_param_comb) == len(gamma_list) * len(c_list)

def helper_h_params():
    # small number of h params
    gamma_list = [0.01, 0.005]
    c_list = [0.1, 0.2]

    params = {}
    params["gamma"] = gamma_list
    params["C"] = c_list
    h_param_comb = get_all_h_param_comb(params)
    return h_param_comb

def helper_create_bin_data(n=100, d=7):
    x_train_0 = np.random.randn(n, d)
    x_train_1 = 1.5 + np.random.randn(n, d)
    x_train = np.vstack((x_train_0, x_train_1))
    y_train = np.zeros(2 * n)
    y_train[n:] = 1

    return x_train, y_train

def test_tune_and_save():    
    h_param_comb = helper_h_params()
    x_train, y_train = helper_create_bin_data(n=100, d=7)
    x_dev, y_dev = x_train, y_train

    clf = svm.SVC()
    metric = metrics.accuracy_score
    
    model_path = "test_run_model_path.joblib"
    actual_model_path = tune_and_save(clf, x_train, y_train, x_dev, y_dev, metric, h_param_comb, model_path)

    assert actual_model_path == model_path
    assert os.path.exists(actual_model_path)
    assert type(load(actual_model_path)) == type(clf)


def test_not_biased():    
    h_param_comb = helper_h_params()
    x_train, y_train = helper_create_bin_data(n=100, d=7)
    x_dev, y_dev = x_train, y_train
    x_test, y_test = x_train, y_train

    clf = svm.SVC()
    metric = metrics.accuracy_score
    
    model_path = "test_run_model_path.joblib"
    actual_model_path = tune_and_save(clf, x_train, y_train, x_dev, y_dev, metric, h_param_comb, model_path)
    best_model = load(actual_model_path)

    predicted = best_model.predict(x_test)

    assert len(set(predicted))!=1


def test_predicts_all():    
    h_param_comb = helper_h_params()
    x_train, y_train = helper_create_bin_data(n=100, d=7)
    x_dev, y_dev = x_train, y_train
    x_test, y_test = x_train, y_train

    clf = svm.SVC()
    metric = metrics.accuracy_score
    
    model_path = "test_run_model_path.joblib"
    actual_model_path = tune_and_save(clf, x_train, y_train, x_dev, y_dev, metric, h_param_comb, model_path)
    best_model = load(actual_model_path)

    predicted = best_model.predict(x_test)

    assert set(predicted) == set(y_test)


def test_seed_split_same():
    random_seed_val1=10
    random_seed_val2=10
    data, label = helper_create_bin_data(n=1000, d=7)

    train_frac, dev_frac, test_frac= 0.5,0.2,0.3

    X_train1, y_train1, X_dev1, y_dev1, X_test1, y_test1=train_dev_test_split(data, label, train_frac, dev_frac,random_seed_val1)
    X_train2, y_train2, X_dev2, y_dev2, X_test2, y_test2=train_dev_test_split(data, label, train_frac, dev_frac,random_seed_val2)


    assert (X_train1==X_train2).all() 
    assert (X_dev1==X_dev2).all() 
    assert (X_test1==X_test2).all()

def test_seed_split_different():
    random_seed_val1=10
    random_seed_val2=50
    data, label = helper_create_bin_data(n=1000, d=7)

    train_frac, dev_frac, test_frac= 0.5,0.2,0.3

    X_train1, y_train1, X_dev1, y_dev1, X_test1, y_test1=train_dev_test_split(data, label, train_frac, dev_frac,random_seed_val1)
    X_train2, y_train2, X_dev2, y_dev2, X_test2, y_test2=train_dev_test_split(data, label, train_frac, dev_frac,random_seed_val2)


    assert (X_train1==X_train2).all() 
    assert (X_dev1==X_dev2).all() 
    assert (X_test1==X_test2).all()


