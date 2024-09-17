import os
import pickle
import json
import sklearn
import random
import argparse
import pandas as pd
import openpyxl
import numpy as np
import prettytable as pt
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.stats as stats
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import utils

parser = argparse.ArgumentParser()
parser.add_argument("-g",  "--gpu", default='4', type=str)
parser.add_argument("-mn", "--model_name", default='scibert', type=str)
parser.add_argument("-dn", "--data_name", default='c10', type=str)
parser.add_argument("-cf", "--contrib_flag", default=0, type=int)
parser.add_argument("-mf", "--mesh_flag", default=0, type=int)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import RobertaTokenizer, RobertaModel
from transformers import BartTokenizer, BartModel
from transformers import  TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from torch.utils.data.dataloader import DataLoader
from torch import nn

import evaluate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


abs_dir = "/data_share/hsz_project"
data_name    = args.data_name
contrib_flag = args.contrib_flag
mesh_flag    = args.mesh_flag
num_labels = 3

data_di_path        = os.path.join(abs_dir, "data/di/di_{}/samples_{}_balance.json".format(data_name, data_name))
data_contrib_path   = os.path.join(abs_dir, "data/di/di_{}/samples_{}_contrib_parsed.json".format(data_name, data_name))
data_mesh_path      = os.path.join(abs_dir, "data/di/di_{}/samples_{}_mesh.json".format(data_name, data_name))
data_score_path     = os.path.join(abs_dir, "data/di/di_{}/samples_{}_score.json".format(data_name, data_name))
data_mesh_type_path = os.path.join(abs_dir, "data/di/mesh_find_specific_level.pkl")
metric = evaluate.load("./evaluate-main/metrics/accuracy")

if args.model_name == "bert":
    model_base_path   = os.path.join(abs_dir, "model_based/BERT/bert-base-uncased")
    model_save_path   = os.path.join(abs_dir, "model_trained/model_di_{}/BERT_{}_{}/bert-base-uncased".format(data_name, str(contrib_flag), str(mesh_flag)))
    results_save_path = os.path.join(abs_dir, "results/results_di_{}/BERT_{}_{}/bert-base-uncased".format(data_name, str(contrib_flag), str(mesh_flag)))

if args.model_name == "bertlarge":
    model_base_path   = os.path.join(abs_dir, "model_based/BERT/bert-large-uncased")
    model_save_path   = os.path.join(abs_dir, "model_trained/model_di_{}/BERTLARGE_{}_{}/bert-large-uncased".format(data_name, str(contrib_flag), str(mesh_flag)))
    results_save_path = os.path.join(abs_dir, "results/results_di_{}/BERTLARGE_{}_{}/bert-large-uncased".format(data_name, str(contrib_flag), str(mesh_flag)))

if args.model_name == "scibert":
    model_base_path   = os.path.join(abs_dir, "model_based/SCIBERT/scibert-scivocab-uncased")
    model_save_path   = os.path.join(abs_dir, "model_trained/model_di_{}/SCIBERT_{}_{}/scibert-scivocab-uncased".format(data_name, str(contrib_flag), str(mesh_flag)))
    results_save_path = os.path.join(abs_dir, "results/results_di_{}/SCIBERT_{}_{}/scibert-scivocab-uncased".format(data_name, str(contrib_flag), str(mesh_flag)))

if args.model_name == "roberta":
    model_base_path   = os.path.join(abs_dir, "model_based/ROBERTA/roberta-base")
    model_save_path   = os.path.join(abs_dir, "model_trained/model_di_{}/ROBERTA_{}_{}/roberta-base".format(data_name, str(contrib_flag), str(mesh_flag)))
    results_save_path = os.path.join(abs_dir, "results/results_di_{}/ROBERTA_{}_{}/roberta-base".format(data_name, str(contrib_flag), str(mesh_flag)))

if args.model_name == "robertalarge":
    model_base_path   = os.path.join(abs_dir, "model_based/ROBERTA/roberta-large")
    model_save_path   = os.path.join(abs_dir, "model_trained/model_di_{}/ROBERTALARGE_{}_{}/roberta-large".format(data_name, str(contrib_flag), str(mesh_flag)))
    results_save_path = os.path.join(abs_dir, "results/results_di_{}/ROBERTALARGE_{}_{}/roberta-large".format(data_name, str(contrib_flag), str(mesh_flag)))

if args.model_name == "bart":
    model_base_path   = os.path.join(abs_dir, "model_based/BART/bart-base")
    model_save_path   = os.path.join(abs_dir, "model_trained/model_di_{}/BART_{}_{}/bart-base".format(data_name, str(contrib_flag), str(mesh_flag)))
    results_save_path = os.path.join(abs_dir, "results/results_di_{}/BART_{}_{}/bart-base".format(data_name, str(contrib_flag), str(mesh_flag)))

if not os.path.exists(model_save_path):   os.makedirs(model_save_path)
if not os.path.exists(results_save_path): os.makedirs(results_save_path)

print("---GPU情况:---")
print(os.environ['CUDA_VISIBLE_DEVICES'])
print("---路径依赖---")
print("数据路径: {}".format(data_di_path))
print("预训练模型路径: {}".format(model_base_path))
print("模型储存路径: {}".format(model_save_path))
print("模型结果路径: {}".format(results_save_path))
print("---任务种类---")
print("BERT + AutoModelForSequenceClassification + Disruptive index classification")
print("标签类别数目: {}".format(num_labels))
print("---输入信息---")
print("采用引用数据类型: {}".format(data_name))
print("是否输入贡献种类: {}".format(contrib_flag))
print("是否输入主题种类: {}".format(mesh_flag))
print("\n")


def get_special_tokens_MeSH():
    """ MeSH词种类"""
    mesh_find_specific_level = utils.read_pickle(data_mesh_type_path)
    additional_special_tokens_MeSH = list()
    for i in mesh_find_specific_level:
        additional_special_tokens_MeSH += list(mesh_find_specific_level[i])
    additional_special_tokens_MeSH = set(additional_special_tokens_MeSH)

    additional_special_tokens_MeSH_format = list()
    for mesh in additional_special_tokens_MeSH:
        if mesh == '':
            mesh_format = '[***]'
        else:
            mesh_format = '[' + mesh + ']'
        additional_special_tokens_MeSH_format.append(mesh_format)
    return additional_special_tokens_MeSH_format


def get_special_tokens_Contrib():
    # 关于贡献种类的tokens
    additional_special_tokens_contribs = ['[Exploration_of_New_Problems]',
                                          '[Innovative_Results]',
                                          '[Innovation_in_Application]',
                                          '[Establishment_of_New_Theories]', 
                                          '[Development_of_New_Methods]']
    return additional_special_tokens_contribs

# 关于MeSH主题的tokens
additional_special_tokens_MeSH = get_special_tokens_MeSH()
additional_special_tokens_contribs = get_special_tokens_Contrib()

# tokenizer
if args.model_name in ['bert', 'scibert', 'bertlarge']:
    tokenizer = AutoTokenizer.from_pretrained(model_base_path)
if args.model_name in ['roberta', 'robertalarge']:
    tokenizer = RobertaTokenizer.from_pretrained(model_base_path)
if args.model_name in ['bart']:
    tokenizer = BartTokenizer.from_pretrained(model_base_path)

# add special tokens
special_tokens_dict = {'additional_special_tokens': additional_special_tokens_contribs + additional_special_tokens_MeSH}
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
# data_collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def tokenize(batch):
    titles = batch['titles']
    abstracts = batch['abstracts']
    if contrib_flag:
        contribs = batch['contribs']
    else:
        contribs = [''] * len(titles)
    if mesh_flag:
        meshs = batch['MeSH']
    else:
        meshs = [''] * len(titles)
    texts = [i + j + m + n for i, j, m, n in zip(meshs, contribs, titles, abstracts)]
    return tokenizer(texts, max_length=512, truncation=True)


def load_samples():
    """ 本地路径读取数据 """
    samples = utils.read_json(data_di_path)
    dataset  = {'pids': list(), 'titles': list(), 'abstracts': list(), 'labels': list()}
    for pid in samples:
        title, abstract, label = samples[pid]['title'], samples[pid]['abstract'], samples[pid]['label']
        dataset['pids'].append(pid)
        dataset['titles'].append(title)
        dataset['abstracts'].append(abstract)
        dataset['labels'].append(int(label))
    dataset = Dataset.from_dict(dataset)
    return dataset


def load_samples_Contrib(train_dataset, valid_dataset, test_dataset):
    """ 载入贡献种类数据 """
    samples_c10_contrib_parsed = utils.read_json(data_contrib_path)

    def add_contribs(train_dataset):
        train_contribs = list()
        for pid in train_dataset['pids']:
            if pid in samples_c10_contrib_parsed:
                contribs = samples_c10_contrib_parsed[pid]['contribs']
                contribs = " ".join(["[{}]".format("_".join(i.split(" "))) for i in contribs])
            else:
                contribs = ""
            train_contribs.append(contribs)
        train_dataset = train_dataset.add_column('contribs', train_contribs)
        return train_dataset
    
    train_dataset = add_contribs(train_dataset)
    valid_dataset = add_contribs(valid_dataset)
    test_dataset  = add_contribs(test_dataset)
    return train_dataset, valid_dataset, test_dataset


def load_samples_MeSH(train_dataset, valid_dataset, test_dataset):
    """ 载入主题种类数据 """
    samples_c10_mesh = utils.read_json(data_mesh_path)

    def add_MeSH(train_dataset):
        train_MeSHs = list()
        for pid in train_dataset['pids']:
            meshs_format = list()
            if pid in samples_c10_mesh:
                meshs = samples_c10_mesh[pid]
                meshs = sorted(set(meshs))
                for mesh in meshs:
                    if mesh == '':
                        meshs_format.append('[***]')
                    else:
                        meshs_format.append('[' + mesh + ']')
            else:
                meshs_format.append('[***]')
            train_MeSHs.append(' '.join(meshs_format).strip())
        train_dataset = train_dataset.add_column('MeSH', train_MeSHs)
        return train_dataset

    train_dataset = add_MeSH(train_dataset)
    valid_dataset = add_MeSH(valid_dataset)
    test_dataset  = add_MeSH(test_dataset)
    return train_dataset, valid_dataset, test_dataset


def load_dataset():

    def split_data(dataset):
        print("按照 8:1:1 划分训练集, 验证集, 测试集")
        # 训练集
        split_dataset  = dataset.train_test_split(test_size=0.2, shuffle=True)
        train_dataset = split_dataset['train']
        # 验证集 & 测试集
        split_dataset2 = split_dataset['test'].train_test_split(test_size=0.5, shuffle=True)
        test_dataset  = split_dataset2['train']
        valid_dataset = split_dataset2['test']

        train_dataset.save_to_disk(os.path.join(os.path.dirname(data_di_path), "train_dataset(score)"))
        valid_dataset.save_to_disk(os.path.join(os.path.dirname(data_di_path), "valid_dataset(score)"))
        test_dataset.save_to_disk(os.path.join(os.path.dirname(data_di_path), "test_dataset(score)"))
        return train_dataset, valid_dataset, test_dataset

    train_dataset_path = os.path.join(os.path.dirname(data_di_path), "train_dataset(score)")
    valid_dataset_path = os.path.join(os.path.dirname(data_di_path), "valid_dataset(score)")
    test_dataset_path  = os.path.join(os.path.dirname(data_di_path), "test_dataset(score)")

    if not os.path.exists(train_dataset_path) or not os.path.exists(valid_dataset_path) or not os.path.exists(test_dataset_path) :
        train_dataset, valid_dataset, test_dataset = split_data(load_samples())
    else:
        train_dataset = load_from_disk(train_dataset_path)
        valid_dataset = load_from_disk(valid_dataset_path)
        test_dataset  = load_from_disk(test_dataset_path)

    if contrib_flag:
        train_dataset, valid_dataset, test_dataset = load_samples_Contrib(train_dataset, valid_dataset, test_dataset)
    if mesh_flag:
        train_dataset, valid_dataset, test_dataset = load_samples_MeSH(train_dataset, valid_dataset, test_dataset)

    train_dataset = train_dataset.map(tokenize, batched=True)
    valid_dataset = valid_dataset.map(tokenize, batched=True)
    test_dataset  = test_dataset.map(tokenize,  batched=True)

    def show_label_num(train_dataset):
        """  训练集中每个类别的数目 """
        label2num = dict()
        for  label in train_dataset['labels']:
            if label not in label2num:
                label2num[label] = 1
            else:
                label2num[label] += 1
        return label2num

    label2num_train = show_label_num(train_dataset)
    label2num_valid = show_label_num(valid_dataset)
    label2num_test  = show_label_num(test_dataset)

    tb = pt.PrettyTable()
    tb.field_names = [""] + list(np.arange(len(label2num_train)))
    tb.add_row(["训练集"] + [label2num_train[i] for i in range(len(label2num_train))])
    tb.add_row(["验证集"] + [label2num_valid[i] for i in range(len(label2num_valid))])
    tb.add_row(["测试集"] + [label2num_test[i]  for i in range(len(label2num_test))])

    print(tb)
    return train_dataset, valid_dataset, test_dataset


def train_model(batch_size=32):

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    print("----载入模型----")
    model = AutoModelForSequenceClassification.from_pretrained(model_base_path, num_labels=num_labels)
    model.resize_token_embeddings(len(tokenizer))

    print("----载入数据----")
    train_dataset, valid_dataset, test_dataset = load_dataset()
    bert_input_features = [feature for feature in train_dataset.features if feature not in ['input_ids', 'token_type_ids', 'attention_mask', 'labels']]
    train_dataset = train_dataset.remove_columns(bert_input_features)
    valid_dataset = valid_dataset.remove_columns(bert_input_features)

    training_args = TrainingArguments(
        output_dir=model_save_path,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=2,
        weight_decay=0.01,
        load_best_model_at_end=True,
        disable_tqdm=False,
        )

    trainer = Trainer(model=model, 
        args=training_args, 
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        )
    trainer.train()


def load_trained_model(peft_path=''):
    if not peft_path:
        checkpoints_steps = dict()
        for checkpoints in os.listdir(model_save_path):
            if checkpoints.startswith('checkpoint'):
                beg_str, end_number = checkpoints.split("-")
                checkpoints_steps[int(end_number)] = checkpoints
        final_ck = checkpoints_steps[np.max(list(checkpoints_steps.keys()))]
        peft_path = os.path.join(model_save_path, final_ck)

    print("模型读取路径: {}".format(peft_path))
    model = AutoModelForSequenceClassification.from_pretrained(peft_path)
    model.cuda()
    model.eval()
    if args.model_name in ['bert', 'scibert', 'bertlarge']:
        tokenizer = AutoTokenizer.from_pretrained(model_base_path)
    if args.model_name in ['roberta', 'robertalarge']:
        tokenizer = RobertaTokenizer.from_pretrained(model_base_path)
    if args.model_name in ['bart']:
        tokenizer = BartTokenizer.from_pretrained(model_base_path)
    return model, tokenizer


def evaluate_model(eval_set, batch_size=8, peft_path=''):
    train_dataset, valid_dataset, test_dataset = load_dataset()
    bert_input_features = [feature for feature in train_dataset.features if feature not in ['input_ids', 'token_type_ids', 'attention_mask', 'labels']]

    model, tokenizer = load_trained_model(peft_path)

    # 评价结果存储路径
    if not peft_path:
        checkpoints_steps = dict()
        for checkpoints in os.listdir(model_save_path):
            if checkpoints.startswith('checkpoint'):
                beg_str, end_number = checkpoints.split("-")
                checkpoints_steps[int(end_number)] = checkpoints
        final_ck = checkpoints_steps[np.max(list(checkpoints_steps.keys()))]
        peft_path = os.path.join(model_save_path, final_ck)
    eval_save_path = os.path.join(results_save_path, os.path.basename(peft_path))
    os.makedirs(eval_save_path, exist_ok=True)

    if eval_set == "train_set":
         dataloader = DataLoader(train_dataset.remove_columns(bert_input_features), batch_size=batch_size, shuffle=False, collate_fn=data_collator)
         eval_save_path = os.path.join(eval_save_path,  "eval_on_train.pkl")
         dataset =  train_dataset
    elif eval_set == "valid_set":
         dataloader = DataLoader(valid_dataset.remove_columns(bert_input_features), batch_size=batch_size, shuffle=False, collate_fn=data_collator)
         eval_save_path = os.path.join(eval_save_path,  "eval_on_valid.pkl")
         dataset =  valid_dataset
    elif eval_set == "test_set":
         dataloader = DataLoader(test_dataset.remove_columns(bert_input_features), batch_size=batch_size, shuffle=False, collate_fn=data_collator)
         eval_save_path = os.path.join(eval_save_path,  "eval_on_test.pkl")
         dataset =  test_dataset
    else:
        print("评价数据集错误")
        return -1

    pred_labels = list()
    true_labels = list()
    pred_scores = list()
    for batch in tqdm(dataloader):
        labels = batch['labels']
        if args.model_name in ['bert', 'scibert']:
            input_ids = batch['input_ids'].cuda()
            token_type_ids = batch['token_type_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            output = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        else:
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            output = model(input_ids=input_ids, attention_mask=attention_mask)

        # 预测概率转换成得分
        output_softmax = nn.functional.softmax(output.logits, dim=-1).detach().cpu().numpy()
        scores = compute_score(output_softmax)
        # 分类标签
        labels = labels.detach().cpu().numpy()
        logits = output['logits'].detach().cpu().numpy()
        true_labels += list(labels)
        pred_labels += list(np.argmax(logits, axis=-1))
        pred_scores += scores

    evaluate_result = dict()
    for i, pmid in enumerate(dataset['pids']):
        pred_label = pred_labels[i]
        true_label = true_labels[i]
        pred_score = pred_scores[i]
        evaluate_result[pmid] = dict()
        evaluate_result[pmid]["pred_label"] = pred_label
        evaluate_result[pmid]["true_label"] = true_label
        evaluate_result[pmid]["pred_score"] = int(pred_score)

    utils.save_pickle(evaluate_result, eval_save_path)


def compute_score(output_softmax):
    """ 根据分类的logits, 经过softmax, 计算0-100的得分""" 
    batch_size, num_cls = output_softmax.shape
    
    # boundary = np.arange(0, 100+1e-1, int(100/num_cls))
    # upper_boundary = boundary[1:]
    # lower_boundary = boundary[:-1]
    
    lower_boundary = np.array([0, 50, 50])
    upper_boundary = np.array([50, 0, 100])

    score_in_cls = lower_boundary + output_softmax * (upper_boundary - lower_boundary)
    score_avg = np.sum(output_softmax * score_in_cls, axis=-1)
    return list(score_avg)


def spearman_corr(x, y):
    # 计算Spearman相关系数
    corr_coef, p_value = stats.spearmanr(x, y)
    print("Spearman correlation: {:.4f}".format( corr_coef))
    print("P value: {:.4f}".format(p_value))
    return corr_coef, p_value


def print_results(peft_path=''):
    """  分类精度 """
    if not peft_path:
        checkpoints_steps = dict()
        for checkpoints in os.listdir(model_save_path):
            if checkpoints.startswith('checkpoint'):
                beg_str, end_number = checkpoints.split("-")
                checkpoints_steps[int(end_number)] = checkpoints
        final_ck = checkpoints_steps[np.max(list(checkpoints_steps.keys()))]
        peft_path = os.path.join(model_save_path, final_ck)

    checkpoint_name = os.path.basename(peft_path)
    eval_save_path  = os.path.join(results_save_path, checkpoint_name)

    eval_on_train = utils.read_pickle(os.path.join(eval_save_path,  "eval_on_train.pkl"))
    eval_on_valid = utils.read_pickle(os.path.join(eval_save_path,  "eval_on_valid.pkl"))
    eval_on_test  = utils.read_pickle(os.path.join(eval_save_path,  "eval_on_test.pkl"))

    # samples_score = utils.read_json(data_score_path)
    
    def read_eval_results(eval_on_train):
        y_pred_on_train = [eval_on_train[pid]['pred_label'] for pid in eval_on_train]
        y_true_on_train = [eval_on_train[pid]['true_label'] for pid in eval_on_train]
        y_predscore_on_train = [eval_on_train[pid]['pred_score'] for pid in eval_on_train]
        y_c10xdi_on_train = list()
        y_nlcsxdi_on_train = list()
        for pmid in eval_on_train:
            # cc = samples_score[pmid]['cc']
            # di = samples_score[pmid]['di']
            y_c10xdi_on_train.append(0)
        if args.data_name == "c10":
            y_actualscore_on_train = y_c10xdi_on_train
        return y_pred_on_train, y_true_on_train, y_predscore_on_train, y_actualscore_on_train

    y_pred_on_train, y_true_on_train, y_predscore_on_train, y_actualscore_on_train = read_eval_results(eval_on_train)
    y_pred_on_valid, y_true_on_valid, y_predscore_on_valid, y_actualscore_on_valid = read_eval_results(eval_on_valid)
    y_pred_on_test,  y_true_on_test,  y_predscore_on_test,  y_actualscore_on_test  = read_eval_results(eval_on_test)
    # 混淆矩阵
    matrix_train = confusion_matrix(y_true_on_train, y_pred_on_train)
    matrix_valid = confusion_matrix(y_true_on_valid, y_pred_on_valid)
    matrix_test  = confusion_matrix(y_true_on_test,  y_pred_on_test)
    # 分类精度
    report_on_train = classification_report(y_true_on_train, y_pred_on_train, digits=4)
    report_on_valid = classification_report(y_true_on_valid, y_pred_on_valid, digits=4)
    report_on_test  = classification_report(y_true_on_test,  y_pred_on_test,  digits=4)

    print("训练集上评价:")
    print(matrix_train)
    print(report_on_train)
    corr_coef_train, p_value_train = spearman_corr(y_predscore_on_train, y_actualscore_on_train)
    print("验证集上评价:")
    print(matrix_valid)
    print(report_on_valid)
    corr_coef_valid, p_value_valid = spearman_corr(y_predscore_on_valid, y_actualscore_on_valid)
    print("测试集上评价:")
    print(matrix_test)
    print(report_on_test)
    corr_coef_test, p_value_test = spearman_corr(y_predscore_on_test, y_actualscore_on_test)
    
    eval_results = dict()
    eval_results["train"] = dict()
    eval_results["valid"] = dict()
    eval_results["test"]  = dict()

    eval_results["train"]['matrix_train'] = matrix_train
    eval_results["train"]['report_on_train'] = report_on_train
    eval_results["train"]['corr_coef_train'] = corr_coef_train

    eval_results["valid"]['matrix_valid'] = matrix_valid
    eval_results["valid"]['report_on_valid'] = report_on_valid
    eval_results["valid"]['corr_coef_valid'] = corr_coef_valid

    eval_results["test"]['matrix_test'] = matrix_test
    eval_results["test"]['report_on_test'] = report_on_test
    eval_results["test"]['corr_coef_test'] = corr_coef_test

    utils.save_pickle(eval_results, os.path.join(eval_save_path, 'eval_results.pkl'))


def compare_results_among_models(peft_path=''):
    """ 将模型的结果统一比较 """
    if not peft_path:
        checkpoints_steps = dict()
        for checkpoints in os.listdir(model_save_path):
            if checkpoints.startswith('checkpoint'):
                beg_str, end_number = checkpoints.split("-")
                checkpoints_steps[int(end_number)] = checkpoints
        final_ck = checkpoints_steps[np.max(list(checkpoints_steps.keys()))]
        peft_path = os.path.join(model_save_path, final_ck)

    for i in range(0, 2):
        for j in range(0, 2):
            results_save_path = os.path.join(abs_dir, "results/results_di_{}_{}_{}/SCIBERT/scibert-scivocab-uncased".format(data_name, str(i), str(j)))
            checkpoint_name = os.path.basename(peft_path)
            eval_save_path  = os.path.join(results_save_path, checkpoint_name)
            eval_results = utils.read_pickle(os.path.join(eval_save_path, 'eval_results.pkl'))

            print("\ncontrib : {}, mesh: {}".format(i, j))
            report_on_test = eval_results["test"]['report_on_test']
            corr_coef_test = eval_results["test"]['corr_coef_test']
            print(report_on_test)
            print('Spearman correlation : {:.4f}'.format(corr_coef_test))


def evaluate_model_on_2024(peft_path=''):
    """ 在2024年的期刊论文上评价 """
    model, tokenizer = load_trained_model(peft_path)

    def test_func(data_2024):
        batch_size = 4
        journals = list()
        titles = list()
        pred_scores = list()
        pred_labels = list()
        texts = list()
        for i, (journal, title, abstract) in enumerate(data_2024):
            text = title + abstract
            texts.append(text)
            journals.append(journal)
            titles.append(title)
            if len(texts) >= batch_size or i == len(data_2024)-1:
                batch = tokenizer(texts, max_length=512, truncation=True, return_tensors='pt', padding=True)

                if args.model_name in ['bert', 'scibert']:
                    input_ids = batch['input_ids'].cuda()
                    token_type_ids = batch['token_type_ids'].cuda()
                    attention_mask = batch['attention_mask'].cuda()
                    output = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                else:
                    input_ids = batch['input_ids'].cuda()
                    attention_mask = batch['attention_mask'].cuda()
                    output = model(input_ids=input_ids, attention_mask=attention_mask)
                
                output_softmax = nn.functional.softmax(output.logits, dim=-1).detach().cpu().numpy()
                scores = compute_score(output_softmax)
                labels = np.argmax(output_softmax, axis=-1)

                for score, label in zip(scores, labels):
                    pred_scores.append(score)
                    pred_labels.append(label)
                texts = list()

        data = {"Index": list(), "Journal": list(), "Title": list(), "Score": list(), "Category": list()}
        tb = pt.PrettyTable()
        tb.field_names = ["编号", "期刊", "标题", "原创性得分", "论文类别"]
        idx = 0
        for journal, title, pred_score, pred_label in zip(journals, titles, pred_scores, pred_labels):
            idx += 1
            tb.add_row([idx, journal, title[:20], "{:.2f}".format(pred_score), pred_label])
            data['Index'].append(idx)
            data['Journal'].append(journal)
            data['Title'].append(title)
            data['Score'].append(pred_score)
            data['Category'].append(pred_label)
        df = pd.DataFrame(data)
        df.to_excel("/data_share/hsz_project/data/{}.xlsx".format(journal))
        print(tb)

    # test_func(utils.BMJ_2024)
    # test_func(utils.JCM_2024)
    # test_func(utils.AJE_2024)
    # test_func(utils.MSM_2024)
    test_func(utils.Structure_INPUT)
    print("\n")
    # test_func(utils.Science_2024)
    # test_func(utils.Nature_2024)
    # test_func(utils.Lancet_2024)
    # test_func(utils.JAMA_2024)
    # test_func(utils.CELL_2024)


def evaluate_model_on_gpt_finetuing(peft_path=''):
    """ 为gpt训练数据提供预测得分 """
    neg_trainset = utils.read_json(os.path.join(abs_dir, "data/di/gpt_finetuning/neg_trainset.json"))
    pos_trainset = utils.read_json(os.path.join(abs_dir, "data/di/gpt_finetuning/pos_trainset.json"))

    dataset = {'pids': list(), 'titles': list(), 'abstracts': list()}
    for trainset in [neg_trainset, pos_trainset]:
        for focal_paper_info in trainset:
            pmid_core = focal_paper_info['pmid']
            title_core = focal_paper_info['title_core']
            abstract_core = focal_paper_info['abstract_core']
            dataset['pids'].append(pmid_core)
            dataset['titles'].append(title_core)
            dataset['abstracts'].append(abstract_core)
    dataset = Dataset.from_dict(dataset)

    # 载入模型
    model, tokenizer = load_trained_model(peft_path)

    # 预测得分
    batch_size = 16
    pmid2predscore = dict()
    pmids = list()
    texts = list()
    total_pids = dataset['pids']
    total_titles = dataset['titles']
    total_abstracts = dataset['abstracts']
    for i in tqdm(range(len(dataset))):
        pmid = total_pids[i]
        title = total_titles[i]
        abstract = total_abstracts[i]
        text = title + abstract
        pmids.append(pmid)
        texts.append(text)
        if len(texts) >= batch_size or i == len(dataset)-1:
            batch = tokenizer(texts, max_length=512, truncation=True, return_tensors='pt', padding=True)
            input_ids = batch['input_ids'].cuda()
            token_type_ids = batch['token_type_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()

            output = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            output_softmax = nn.functional.softmax(output.logits, dim=-1).detach().cpu().numpy()
            scores = compute_score(output_softmax)
            for pmid, score in zip(pmids, scores):
                pmid2predscore[pmid] = int(score)
            pmids = list()
            texts = list()
    utils.save_json(pmid2predscore, os.path.join(abs_dir, "data/di/gpt_finetuning/pmid2predscore.json"))


if __name__ == "__main__":
    # scirbert 32 bert 32 roberta 32; bertlarge 8 robertalarge 8
    # batch_size = 8
    # train_model(batch_size)
    # evaluate_model("train_set", 8)
    # evaluate_model("valid_set", 8)
    # evaluate_model("test_set", 8)
    print_results()
    # compare_results(peft_path)
    # evaluate_model_on_2024()
    # evaluate_model_on_gpt_finetuing()
    pass