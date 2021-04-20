import pandas as pd
import numpy as np
import sys
import os 
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime
import time
import json
import shap
import random
import math

import shap
from scipy.stats import pearsonr, spearmanr
from scipy.stats import ttest_rel

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn  import metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report

def compute_ci(array, legend, c=0.95):
    sorted_scores = np.array(array)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print("95% Confidence interval for {}: {:0.4f}[{:0.4f} - {:0.4}]".format(
        legend, sorted_scores.mean(), confidence_lower, confidence_upper))
    return sorted_scores.mean(), confidence_lower, confidence_upper
  
def DoLogisticRegression(X_train, y_train, X_test, y_test):
   clf = LogisticRegression(C=1, 
                            random_state=1, 
                            solver='lbfgs', 
                            multi_class='multinomial', 
                            max_iter=5000, 
                            penalty='l2', 
                            verbose=1, 
                            class_weight={0:2, 1:1})
                clf.fit(X_train, y_train)

    y_predict = clf.predict(X_train)
    prob = clf.predict_proba(X_train)
    prob_true = list(prob[:,1])
    label = y_train.copy()
    label[y_train!=1]=0
    label[y_train==1]=1
    label = list(label)
    roc = roc_curve(label, prob_true, pos_label=1)
    auc = metrics.auc(roc[0], roc[1])
    print('TRAIN_SCORE@class_%d: %.4f'%(i, clf.score(X_train, y_train)))
    print('TRAIN_AUC@class_%d: %.4f'%(i, auc))

    prob_true = list(clf.oob_decision_function_[:,i])
    roc = roc_curve(label, prob_true, pos_label=1)
    auc = metrics.auc(roc[0], roc[1])
    youden_index = 1 - roc[0] + roc[1] - 1
    idx = youden_index.argmax()
    assert(youden_index[idx]==youden_index.max())
    predict = (prob_true >= roc[2][idx]).astype(np.int)
    print('OOB max Youden index: %.4f'%(youden_index[idx]))
    print('OOB sensitivity: %.4f'%(roc[1][idx]))
    print('OOB specificity: %.4f'%(1 - roc[0][idx]))
    print('OOB acc: %.4f'%(1-(predict == y_train).sum()/len(y_train)))
    print('OOB auc: %.4f'%(auc))

    prob = clf.predict_proba(X_test)
    prob_true = np.array(list(prob[:,1]))
    label = y_test.copy()
    label[y_test!=1]=0
    label[y_test==1]=1
    roc = metrics.roc_curve(label, prob_true, pos_label=1)   
    auc = metrics.auc(roc[0], roc[1])
    youden_index = 1 - roc[0] + roc[1] - 1
    idx = youden_index.argmax()
    assert(youden_index[idx]==youden_index.max())
    predict = (prob_true >= roc[2][idx]).astype(np.int)
    print('max Youden index: %.4f'%(youden_index[idx]))
    print('sensitivity: %.4f'%(roc[1][idx]))
    print('specificity: %.4f'%(1 - roc[0][idx]))
    print('acc: %.4f'%((predict == yy_test_cv).sum()/len(yy_test_cv)))
    print('TEST_AUC@class_%d: %.4f'%(i,auc))
    
    return auc
  
  def DoSVC(X_train, y_train, X_test, y_test):
   clf = SVC(C=0.01, 
             kernel='rbf',
             degree=3,
             gamma='scale',
             probability=True,
             random_state=1,
             decision_function_shape='ovo',
             max_iter=-1, 
             verbose=1, 
             tol=0.0001,
             class_weight={0:2, 1:1})
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_train)
    prob = clf.predict_proba(X_train)
    prob_true = list(prob[:,1])
    label = y_train.copy()
    label[y_train!=1]=0
    label[y_train==1]=1
    label = list(label)
    roc = roc_curve(label, prob_true, pos_label=1)
    auc = metrics.auc(roc[0], roc[1])
    print('TRAIN_SCORE@class_%d: %.4f'%(i, clf.score(X_train, y_train)))
    print('TRAIN_AUC@class_%d: %.4f'%(i, auc))

    prob_true = list(clf.oob_decision_function_[:,i])
    roc = roc_curve(label, prob_true, pos_label=1)
    auc = metrics.auc(roc[0], roc[1])
    youden_index = 1 - roc[0] + roc[1] - 1
    idx = youden_index.argmax()
    assert(youden_index[idx]==youden_index.max())
    predict = (prob_true >= roc[2][idx]).astype(np.int)
    print('OOB max Youden index: %.4f'%(youden_index[idx]))
    print('OOB sensitivity: %.4f'%(roc[1][idx]))
    print('OOB specificity: %.4f'%(1 - roc[0][idx]))
    print('OOB acc: %.4f'%(1-(predict == y_train).sum()/len(y_train)))
    print('OOB auc: %.4f'%(auc))

    prob = clf.predict_proba(X_test)
    prob_true = np.array(list(prob[:,1]))
    label = y_test.copy()
    label[y_test!=1]=0
    label[y_test==1]=1
    roc = metrics.roc_curve(label, prob_true, pos_label=1)   
    auc = metrics.auc(roc[0], roc[1])
    youden_index = 1 - roc[0] + roc[1] - 1
    idx = youden_index.argmax()
    assert(youden_index[idx]==youden_index.max())
    predict = (prob_true >= roc[2][idx]).astype(np.int)
    print('max Youden index: %.4f'%(youden_index[idx]))
    print('sensitivity: %.4f'%(roc[1][idx]))
    print('specificity: %.4f'%(1 - roc[0][idx]))
    print('acc: %.4f'%((predict == yy_test_cv).sum()/len(yy_test_cv)))
    print('TEST_AUC@class_%d: %.4f'%(i,auc))
    
    return auc
  
  def DoRandomForest(X_train, y_train, X_test, y_test):
   clf = RandomForestClassifier(criterion='entropy',
                                bootstrap=True,
                                random_state=1,
                                oob_score=True,
                                n_estimators=100,
                                max_features=1,
                                max_depth=6,
                                min_samples_split=3,
                                min_samples_leaf=1,
                                class_weight={0:2,1:1})
                clf.fit(X_train, y_train)

    y_predict = clf.predict(X_train)
    prob = clf.predict_proba(X_train)
    prob_true = list(prob[:,1])
    label = y_train.copy()
    label[y_train!=1]=0
    label[y_train==1]=1
    label = list(label)
    roc = roc_curve(label, prob_true, pos_label=1)
    auc = metrics.auc(roc[0], roc[1])
    print('TRAIN_SCORE@class_%d: %.4f'%(i, clf.score(X_train, y_train)))
    print('TRAIN_AUC@class_%d: %.4f'%(i, auc))

    prob_true = list(clf.oob_decision_function_[:,i])
    roc = roc_curve(label, prob_true, pos_label=1)
    auc = metrics.auc(roc[0], roc[1])
    youden_index = 1 - roc[0] + roc[1] - 1
    idx = youden_index.argmax()
    assert(youden_index[idx]==youden_index.max())
    predict = (prob_true >= roc[2][idx]).astype(np.int)
    print('OOB max Youden index: %.4f'%(youden_index[idx]))
    print('OOB sensitivity: %.4f'%(roc[1][idx]))
    print('OOB specificity: %.4f'%(1 - roc[0][idx]))
    print('OOB acc: %.4f'%(1-(predict == y_train).sum()/len(y_train)))
    print('OOB auc: %.4f'%(auc))

    prob = clf.predict_proba(X_test)
    prob_true = np.array(list(prob[:,1]))
    label = y_test.copy()
    label[y_test!=1]=0
    label[y_test==1]=1
    roc = metrics.roc_curve(label, prob_true, pos_label=1)   
    auc = metrics.auc(roc[0], roc[1])
    youden_index = 1 - roc[0] + roc[1] - 1
    idx = youden_index.argmax()
    assert(youden_index[idx]==youden_index.max())
    predict = (prob_true >= roc[2][idx]).astype(np.int)
    print('max Youden index: %.4f'%(youden_index[idx]))
    print('sensitivity: %.4f'%(roc[1][idx]))
    print('specificity: %.4f'%(1 - roc[0][idx]))
    print('acc: %.4f'%((predict == yy_test_cv).sum()/len(yy_test_cv)))
    print('TEST_AUC@class_%d: %.4f'%(i,auc))
    
    return auc
