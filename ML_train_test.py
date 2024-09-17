import os
import pickle
import json
import sklearn
import random
import argparse
import numpy as np
import prettytable as pt
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.stats as stats
# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# 数据
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# 模型
# 线性模型
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
# 非线性模型
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
# 集成模型
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
# 评价
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import utils

parser = argparse.ArgumentParser()
parser.add_argument("-dn", "--data_name", default='c10', type=str)
args = parser.parse_args()

from datasets import load_dataset, Dataset, load_from_disk


""" tradional machine learning on di prediction """
abs_dir = "/data_share/hsz_project"
data_name = args.data_name
data_di_path = os.path.join(abs_dir, "data/di/di_{}/samples_{}.json".format(data_name, data_name))
model_tfidf_path = os.path.join(abs_dir, "model_trained/model_tfidf")
model_tml_path = os.path.join(abs_dir, "model_trained/model_di_{}/TML".format(data_name))
results_tml_path = os.path.join(abs_dir, "results/results_di_{}/TML".format(data_name))

if not os.path.exists(model_tfidf_path): os.makedirs(model_tfidf_path)
if not os.path.exists(model_tml_path): os.makedirs(model_tml_path)
if not os.path.exists(results_tml_path): os.makedirs(results_tml_path)


def load_dataset():
    train_dataset_path = os.path.join(os.path.dirname(data_di_path), "train_dataset(score)")
    valid_dataset_path = os.path.join(os.path.dirname(data_di_path), "valid_dataset(score)")
    test_dataset_path  = os.path.join(os.path.dirname(data_di_path), "test_dataset(score)")
    train_dataset = load_from_disk(train_dataset_path)
    valid_dataset = load_from_disk(valid_dataset_path)
    test_dataset  = load_from_disk(test_dataset_path)
    return train_dataset, valid_dataset, test_dataset


def train_tf_idf():
    """ 训练集上训练TF-IDF """
    train_dataset, valid_dataset, test_dataset = load_dataset()
    titles = train_dataset['titles']
    abstracts = train_dataset['abstracts']
    corpus = list()
    for title, abstract in zip(titles, abstracts):
        corpus.append(title + abstract)

    print("训练TF-IDF")
    CV_model = CountVectorizer(min_df=10)
    TT_model = TfidfTransformer()
    CV_model.fit(corpus)
    TT_model.fit(CV_model.transform(corpus))
    utils.save_pickle(CV_model, os.path.join(model_tfidf_path, "CV_model.pkl"))
    utils.save_pickle(TT_model, os.path.join(model_tfidf_path, "TT_model.pkl"))
    return CV_model, TT_model


def load_tf_idf(train_dataset):
    """ 生成TF-IDF """
    titles = train_dataset['titles']
    abstracts = train_dataset['abstracts']
    labels = train_dataset['labels']
    corpus = list()
    for title, abstract in zip(titles, abstracts):
        corpus.append(title + abstract)

    # 读取模型
    CV_model_path = os.path.join(model_tfidf_path, "CV_model.pkl")
    TT_model_path = os.path.join(model_tfidf_path, "TT_model.pkl")
    if os.path.exists(CV_model_path) and os.path.exists(TT_model_path):
        CV_model = utils.read_pickle(CV_model_path)
        TT_model = utils.read_pickle(TT_model_path)
    else:
        CV_model, TT_model = train_tf_idf()

    # 生成TF-IDF
    X = TT_model.transform(CV_model.transform(corpus))
    return X.toarray(), labels


def train_tml():
    train_dataset, valid_dataset, test_dataset = load_dataset()
    X_train, y_train = load_tf_idf(train_dataset)
    X_valid, y_valid = load_tf_idf(valid_dataset)
    X_test,  y_test  = load_tf_idf(test_dataset)

    def eval_func(model, tmp_model_path, tmp_results_path):
        y_pred_train = model.predict(X_train)
        y_pred_valid = model.predict(X_valid)
        y_pred_test  = model.predict(X_test)

        def report_eval_func(y_train, y_pred_train):
            conf_matrix_train = confusion_matrix(y_train, y_pred_train)
            cls_report_train = classification_report(y_train, y_pred_train, output_dict=True)
            return conf_matrix_train, cls_report_train

        conf_matrix_train, cls_report_train = report_eval_func(y_train, y_pred_train)
        conf_matrix_valid, cls_report_valid = report_eval_func(y_valid, y_pred_valid)
        conf_matrix_test,  cls_report_test  = report_eval_func(y_test,  y_pred_test)

        results = {"conf_matrix_train": conf_matrix_train, "cls_report_train": cls_report_train,
                   "conf_matrix_valid": conf_matrix_valid, "cls_report_valid": cls_report_valid, 
                   "conf_matrix_test":  conf_matrix_test,  "cls_report_test": cls_report_test}

        utils.save_pickle(model, tmp_model_path)
        utils.save_pickle(results, tmp_results_path)

    print("----训练LogisticRegression (模型输入: {}, 模型输出: {})----".format(X_train.shape, len(y_train)))
    tmp_model_path = os.path.join(model_tml_path, "LogisticRegression_Model.pkl")
    tmp_results_path = os.path.join(results_tml_path, "LogisticRegression_Results.pkl")
    if not os.path.exists(tmp_model_path) or not tmp_results_path:
        model = LogisticRegression().fit(X_train, y_train)
        eval_func(model, tmp_model_path, tmp_results_path)

    print("----训练KNeighborsClassifier (模型输入: {}, 模型输出: {})----".format(X_train.shape, len(y_train)))
    tmp_model_path = os.path.join(model_tml_path, "KNeighborsClassifier_Model.pkl")
    tmp_results_path = os.path.join(results_tml_path, "KNeighborsClassifier_Results.pkl")
    if not os.path.exists(tmp_model_path) or not tmp_results_path:
        model = KNeighborsClassifier().fit(X_train, y_train)
        eval_func(model, tmp_model_path, tmp_results_path)

    print("----训练GaussianNB (模型输入: {}, 模型输出: {})----".format(X_train.shape, len(y_train)))
    tmp_model_path = os.path.join(model_tml_path, "GaussianNB_Model.pkl")
    tmp_results_path = os.path.join(results_tml_path, "GaussianNB_Results.pkl")
    if not os.path.exists(tmp_model_path) or not tmp_results_path:
        model = GaussianNB().fit(X_train, y_train)
        eval_func(model, tmp_model_path, tmp_results_path)

    # print("----训练SVC (模型输入: {}, 模型输出: {})----".format(X_train.shape, len(y_train)))
    # tmp_model_path = os.path.join(model_tml_path, "SVC_Model.pkl")
    # tmp_results_path = os.path.join(results_tml_path, "SVC_Results.pkl")
    # if not os.path.exists(tmp_model_path) or not tmp_results_path:
    #     model = SVC().fit(X_train, y_train)
    #     eval_func(model, tmp_model_path, tmp_results_path)

    print("----训练RandomForestClassifier (模型输入: {}, 模型输出: {})----".format(X_train.shape, len(y_train)))
    tmp_model_path = os.path.join(model_tml_path, "RandomForestClassifier_Model.pkl")
    tmp_results_path = os.path.join(results_tml_path, "RandomForestClassifier_Results.pkl")
    if not os.path.exists(tmp_model_path) or not tmp_results_path:
        model = RandomForestClassifier().fit(X_train, y_train)
        eval_func(model, tmp_model_path, tmp_results_path)

    print("----训练XGBClassifier (模型输入: {}, 模型输出: {})----".format(X_train.shape, len(y_train)))
    tmp_model_path = os.path.join(model_tml_path, "XGBClassifier_Model.pkl")
    tmp_results_path = os.path.join(results_tml_path, "XGBClassifier_Results.pkl")
    if not os.path.exists(tmp_model_path) or not tmp_results_path:
        model = XGBClassifier().fit(X_train, y_train)
        eval_func(model, tmp_model_path, tmp_results_path)

    print("----训练AdaBoostClassifier (模型输入: {}, 模型输出: {})----".format(X_train.shape, len(y_train)))
    tmp_model_path = os.path.join(model_tml_path, "AdaBoostClassifier_Model.pkl")
    tmp_results_path = os.path.join(results_tml_path, "AdaBoostClassifier_Results.pkl")
    if not os.path.exists(tmp_model_path) or not tmp_results_path:
        model = AdaBoostClassifier().fit(X_train, y_train)
        eval_func(model, tmp_model_path, tmp_results_path)


def print_results():

    def show_results(results):
        matrix_train = results["conf_matrix_train"]
        report_on_train  = results["cls_report_train"]
        matrix_valid = results["conf_matrix_valid"]
        report_on_valid  = results["cls_report_valid"]
        matrix_test = results["conf_matrix_test"]
        report_on_test = results["cls_report_test"]
        # print("训练集上评价:")
        # print(matrix_train)
        # print(report_on_train)
        # print("验证集上评价:")
        # print(matrix_valid)
        # print(report_on_valid)
        print("测试集上评价:")
        print(matrix_test)
        print(report_on_test)

    print("----读取LogisticRegression结果----")
    results = utils.read_pickle(os.path.join(results_tml_path, "LogisticRegression_Results.pkl"))
    show_results(results)

    print("----读取KNeighborsClassifier结果----")
    results = utils.read_pickle(os.path.join(results_tml_path, "KNeighborsClassifier_Results.pkl"))
    show_results(results)

    print("----读取GaussianNB结果----")
    results = utils.read_pickle(os.path.join(results_tml_path, "GaussianNB_Results.pkl"))
    show_results(results)
    
    print("----读取RandomForestClassifier结果----")
    results = utils.read_pickle(os.path.join(results_tml_path, "RandomForestClassifier_Results.pkl"))
    show_results(results)

    print("----读取XGBClassifier结果----")
    results = utils.read_pickle(os.path.join(results_tml_path, "XGBClassifier_Results.pkl"))
    show_results(results)

    print("----读取AdaBoostClassifier结果----")
    results = utils.read_pickle(os.path.join(results_tml_path, "AdaBoostClassifier_Results.pkl"))
    show_results(results)

# train_tml()
print_results()
