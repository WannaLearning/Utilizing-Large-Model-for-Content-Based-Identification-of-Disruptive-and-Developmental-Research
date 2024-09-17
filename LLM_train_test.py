import os
import pickle
import json
import sklearn
import random
import argparse
import prettytable as pt
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.stats as stats
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", default='2', type=str)
parser.add_argument("-ms", "--model_size", default=7, type=str)
parser.add_argument("-mn", "--model_name", default='mistral', type=str)
parser.add_argument("-dn", "--data_name", default='c10', type=str)
parser.add_argument("-sf", "--similars_flag", default=0, type=int)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import transformers
from torch import nn
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import  TrainingArguments, Trainer, GenerationConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel

from datasets import load_dataset, Dataset, load_from_disk
from transformers import DataCollatorWithPadding, DataCollatorForSeq2Seq
from torch.utils.data.dataloader import DataLoader

import evaluate

import utils

abs_dir = "/home/irlab/project_hsz"
data_name = args.data_name
model_size = args.model_size
similars_flag = args.similars_flag
num_labels = 3
num_select_similars = 5

# 实验数据
data_di_path = os.path.join(abs_dir, "data/di/di_{}/samples_{}_balance.json".format(data_name, data_name))
# 医学领域诺奖论文
data_nobel_path = os.path.join(abs_dir, "data/di/nobel_medical")
# 2024年journal上的出版论文
data_2024_path = os.path.join(abs_dir, "data/di/journal_2024")

data_structure_abstract_path = os.path.join(abs_dir, "data/di/di_{}/samples_structure_abstract_{}.json".format(data_name, data_name))
data_score_path = os.path.join(abs_dir, "data/di/di_{}/samples_{}_score.json".format(data_name, data_name))
metric = evaluate.load("./evaluate-main/metrics/accuracy")

if args.model_name == "llama":
    model_base_path = "/data_share/model_hub/llama/Llama-2-{}b-chat-hf".format(model_size)
    model_save_path = os.path.join(abs_dir, "model_trained/model_di_{}/LLAMA(CLS)_{}/Llama-2-{}b-chat-di".format(data_name, str(similars_flag), model_size))
    results_save_path = os.path.join(abs_dir, "results/results_di_{}/LLAMA(CLS)_{}/Llama-2-{}b-chat-di".format(data_name, str(similars_flag), model_size))

if args.model_name == "mistral":
    model_base_path = "/data_share/model_hub/Mixtral/Mistral-7B-v0.1"
    model_save_path = os.path.join(abs_dir, "model_trained/model_di_{}/Mistral(CLS)_{}/Mistral-7B-v0.1".format(data_name, str(similars_flag)))
    results_save_path = os.path.join(abs_dir, "results/results_di_{}/Mistral(CLS)_{}/Mistral-7B-v0.1".format(data_name, str(similars_flag)))

if args.model_name == "biomistral":
    model_base_path = "/data_share/model_hub/Mixtral/biomistral"
    model_save_path = os.path.join(abs_dir, "model_trained/model_di_{}/Mistral(CLS)_{}/biomistral".format(data_name, str(similars_flag)))
    results_save_path = os.path.join(abs_dir, "results/results_di_{}/Mistral(CLS)_{}/biomistral".format(data_name, str(similars_flag)))

if not os.path.exists(model_save_path): os.makedirs(model_save_path)
if not os.path.exists(results_save_path): os.makedirs(results_save_path)


tokenizer = AutoTokenizer.from_pretrained(model_base_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def llama_prompt(title_core, abstract_core, title_similars, abstract_similars):

    # truncate the abstract
    max_abstract_words = 640
    abstract_core = " ".join(abstract_core.split(" ")[: max_abstract_words]).strip()
    abstract_similars = [" ".join(abstract_sim.split(" ")[: max_abstract_words]).strip() for abstract_sim in abstract_similars]

    prompt_begin = (#"Below is an instruction that describes a task. With a response that appropriately completes the request.\n\n"
                    "### Instruction:\n"
                    "As a professional researcher in biomedicine field, you are tasked to evaluate the originality and novelty of a given core paper "
                    "by comparing the text of the core paper with that of its similar papers.\n"
                    "### Input:\n"
                    )

    prompt_core = ("The text of the given core paper:\n"
                  f"{title_core}\n"
                  f"{abstract_core}\n\n")

    prompt_similars = ""
    num_similars = min(len(title_similars), num_select_similars)  # 选用的相似论文数目
    if similars_flag:
        for i in range(num_similars):
            j = i + 1
            title_similar_i = title_similars[i]
            abstract_similar_i = abstract_similars[i]
            prompt_similar_i = (f"The text of the {j}th similar paper:\n"
                                f"{title_similar_i}\n"
                                f"{abstract_similar_i}\n\n")
            prompt_similars += prompt_similar_i

    prompt_end = "### Response:"

    if similars_flag != 0:
        prompt = prompt_begin + prompt_core + prompt_similars + prompt_end
    else:
        prompt = title_core + "\n" + abstract_core
    return prompt.strip()


def tokenize(batch):
    title_core_list = batch['titles']
    abstract_core_list = batch["abstracts"]
    title_similars_list = batch["title_similars"]
    abstract_similars_list = batch["abstract_similars"]

    prompts = list()
    for title_core, abstract_core, title_similars, abstract_similars in zip(title_core_list, abstract_core_list, title_similars_list, abstract_similars_list):
        title_similars = title_similars.split("[CAT]")
        abstract_similars = abstract_similars.split("[CAT]")
        prompt = llama_prompt(title_core, abstract_core, title_similars, abstract_similars)
        prompts.append(prompt)
    return tokenizer(prompts, max_length=4096, truncation=True)


def load_dataset():

    def load_samples(samples, train_pids):
        """ 本地路径读取数据 """
        if similars_flag == 2:
            print("读取结构化摘要数据")
            samples_structure_abstract = utils.read_json(data_structure_abstract_path)

        train_pids = set(train_pids)
        dataset  = {'pids': list(), 'titles': list(), 'abstracts': list(), 'labels': list(), 'title_similars': list(), 'abstract_similars': list()}
        for pid in samples:
            if pid not in train_pids:
                continue
            # 评价论文信息
            title, abstract, label = samples[pid]['title'], samples[pid]['abstract'], samples[pid]['label']
            # 相似文献信息
            if similars_flag == 0:
                title_similars_cat = ''
                abstract_similars_cat = ''
            # 标题 + 全部摘要
            if similars_flag == 1: 
                pid_similars = samples[pid]['pmid_similars']
                title_similars = samples[pid]['title_similars']
                abstract_similars = samples[pid]['abstract_similars']
                title_similars_cat = "[CAT]".join(title_similars)
                abstract_similars_cat = "[CAT]".join(abstract_similars)
            # 标题 + 部分结构化摘要
            if similars_flag == 2:
                pid_similars = samples[pid]['pmid_similars']
                title_similars = samples[pid]['title_similars']
                abstract_similars = list()
                for pid_sim in pid_similars:
                    if "CONCLUSIONS" in samples_structure_abstract[pid_sim]:
                        conclusions = samples_structure_abstract[pid_sim]["CONCLUSIONS"]
                        conclusions = sorted(conclusions, key=lambda x: x[-1])
                        conclusions = " ".join([sentence for sentence, idx in conclusions]).strip()
                    else:
                        conclusions = ''
                    abstract_similars.append(conclusions)
                title_similars_cat = "[CAT]".join(title_similars)
                abstract_similars_cat = "[CAT]".join(abstract_similars)

            dataset['pids'].append(pid)
            dataset['titles'].append(title)
            dataset['abstracts'].append(abstract)
            dataset['labels'].append(int(label))
            dataset['title_similars'].append(title_similars_cat)
            dataset['abstract_similars'].append(abstract_similars_cat)

        dataset = Dataset.from_dict(dataset)
        return dataset

    #  创建训练数据
    train_dataset_path = os.path.join(os.path.dirname(data_di_path), "train_dataset(score)")
    valid_dataset_path = os.path.join(os.path.dirname(data_di_path), "valid_dataset(score)")
    test_dataset_path  = os.path.join(os.path.dirname(data_di_path), "test_dataset(score)")

    train_dataset = load_from_disk(train_dataset_path)
    valid_dataset = load_from_disk(valid_dataset_path)
    test_dataset  = load_from_disk(test_dataset_path)

    train_pids = train_dataset['pids']
    valid_pids = valid_dataset['pids']
    test_pids  = test_dataset['pids']
    del train_dataset, valid_dataset, test_dataset

    samples = utils.read_json(data_di_path)
    train_dataset = load_samples(samples, train_pids)
    valid_dataset = load_samples(samples, valid_pids)
    test_dataset  = load_samples(samples, test_pids)

    train_dataset = train_dataset.map(tokenize, batched=True)
    valid_dataset = valid_dataset.map(tokenize, batched=True)
    test_dataset  = test_dataset.map(tokenize,  batched=True)

    # 输入的Token数目分布
    tb = pt.PrettyTable()
    tb.title = "训练集中词数目统计"
    tb.field_names = ["均值", "标准差", "中位数", "25分位数", "75分位数", "最小数", "最大数"]
    train_input_ids = train_dataset['input_ids']
    texts_token_num = np.array([len(train_input_ids[i]) for i in range(len(train_dataset))])

    mean_token_num =  np.mean(texts_token_num)
    std_token_num = np.std(texts_token_num)
    median_token_num = np.median(texts_token_num)
    perc25_token_num = np.percentile(texts_token_num, 25)
    perc75_token_num = np.percentile(texts_token_num, 75)
    min_token_num = min(texts_token_num)
    max_token_num = max(texts_token_num)

    tb.add_row([int(mean_token_num), int(std_token_num), median_token_num, perc25_token_num, perc75_token_num, min_token_num, max_token_num])
    print(tb)
    return train_dataset, valid_dataset, test_dataset

# train_dataset, valid_dataset, test_dataset = load_dataset()
# print(tokenizer.decode(test_dataset['input_ids'][118]))

def load_model_with_qlora():
    """ https://huggingface.co/docs/peft/quicktour """
    # 模型量化配置
    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.bfloat16
                                    )
    
    model = AutoModelForSequenceClassification.from_pretrained(model_base_path, num_labels=num_labels, quantization_config=bnb_config)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False 
    # LoRA配置
    config = LoraConfig(r=128,
                        lora_alpha = 64,
                        target_modules=["q_proj", 'k_proj', "v_proj"], 
                        lora_dropout=0.05, 
                        bias="none", 
                        task_type="SEQ_CLS") # task_type = "SEQ_CLS", "CAUSAL_LM"  # 注意这里要修改
    # 添加LoRA 
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, config)

    # 模型结构和lora微调参数
    # print(model)
    model.print_trainable_parameters()
    return  model

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


class SaveScoreCallback(TrainerCallback):  
    """ 模型库bug, 未能正确存储分类头"""
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def on_save(self, 
                args: TrainingArguments, 
                state: TrainerState,
                control: TrainerControl,
                **kwargs ):
        fname = f"{model_save_path}/checkpoint-{state.global_step}/score.original_module.pt"
        torch.save(self.model.model.score.original_module.state_dict(), fname)


def train_model(batch_size):

    print("----创建qlora模型----")
    model = load_model_with_qlora()

    print("----载入数据----")
    train_dataset, valid_dataset, test_dataset = load_dataset()
    train_dataset = train_dataset.remove_columns(['pids', 'titles', 'abstracts', 'title_similars', 'abstract_similars'])
    eval_dataset = valid_dataset.remove_columns(['pids', 'titles', 'abstracts', 'title_similars', 'abstract_similars'])

    training_args = TrainingArguments(
                output_dir=model_save_path,
                fp16=True,
                evaluation_strategy = "epoch",
                save_strategy = "epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=1,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=1,
                weight_decay=0.01,
                load_best_model_at_end=True,
                disable_tqdm=False,
    )
    trainer = Trainer(model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    compute_metrics=compute_metrics,
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    trainer.add_callback(SaveScoreCallback(model)) 
    trainer.train()


def load_trained_model(peft_path=''):
    print("----载入路径----")
    print(peft_path)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.bfloat16
                                    )
    model = AutoModelForSequenceClassification.from_pretrained(model_base_path, num_labels=num_labels, quantization_config=bnb_config)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False 
    # 载入lora权重
    model = PeftModel.from_pretrained(model, peft_path, torch_dtype=torch.float16, )
    # # 载入分类头权重
    score_weights = torch.load(os.path.join(peft_path, "score.original_module.pt"), map_location='cpu')
    model.score.original_module.load_state_dict(score_weights)
    # model.cuda() # 影响qlora模型报错
    model.eval()
    return model


def evaluate_model(eval_set, peft_path=''):

    if not peft_path:
        checkpoints_steps = dict()
        for checkpoints in os.listdir(model_save_path):
            if checkpoints.startswith('checkpoint'):
                beg_str, end_number = checkpoints.split("-")
                checkpoints_steps[int(end_number)] = checkpoints
        final_ck = checkpoints_steps[np.max(list(checkpoints_steps.keys()))]
        peft_path = os.path.join(model_save_path, final_ck)

    model = load_trained_model(peft_path)

    train_dataset, valid_dataset, test_dataset = load_dataset()
    llama_input_features = [feature for feature in train_dataset.features if feature not in ['input_ids', 'attention_mask', 'labels']]

    checkpoint_name = os.path.basename(peft_path)
    eval_save_path = os.path.join(results_save_path, checkpoint_name)
    os.makedirs(eval_save_path, exist_ok=True)

    batch_size = 16
    if eval_set == "train_set":
         dataloader = DataLoader(train_dataset.remove_columns(llama_input_features), batch_size=batch_size, shuffle=False, collate_fn=data_collator)
         eval_save_path = os.path.join(eval_save_path, "eval_on_train.pkl")
         dataset = train_dataset
    elif eval_set == "valid_set":
         dataloader = DataLoader(valid_dataset.remove_columns(llama_input_features), batch_size=batch_size, shuffle=False, collate_fn=data_collator)
         eval_save_path = os.path.join(eval_save_path, "eval_on_valid.pkl")
         dataset = valid_dataset
    elif eval_set == "test_set":
         dataloader = DataLoader(test_dataset.remove_columns(llama_input_features), batch_size=batch_size, shuffle=False, collate_fn=data_collator)
         eval_save_path = os.path.join(eval_save_path, "eval_on_test.pkl")
         dataset = test_dataset
    else:
        print("评价数据集错误")
        return -1

    pred_labels = list()
    true_labels = list()
    pred_scores = list()
    for batch in tqdm(dataloader):
        labels = batch['labels']
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

    # 自动根据类别划分区间
    # boundary = np.arange(0, 100+1e-1, int(100/num_cls))
    # upper_boundary = boundary[1:]
    # lower_boundary = boundary[:-1]

    # 0类是Development papers; 1类是General papers; 2类是Disruptive papers
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
    """  打印分类精度 """
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

    # Key: pid, Value: nlcs, cc, di, ref_num, label_nlcs_x_di, label_cc_x_di, label_nlcs_x_di_ci, label_cc_x_di_ci
    samples_score = utils.read_json(data_score_path)

    def read_eval_results(eval_on_train):
        y_pred_on_train = [eval_on_train[pid]['pred_label'] for pid in eval_on_train]
        y_true_on_train = [eval_on_train[pid]['true_label'] for pid in eval_on_train]
        y_predscore_on_train = [eval_on_train[pid]['pred_score'] for pid in eval_on_train]
        y_c10xdi_on_train = list()
        y_nlcsxdi_on_train = list()
        for pmid in eval_on_train:
            cc = samples_score[pmid]['cc']
            di = samples_score[pmid]['di']
            y_c10xdi_on_train.append(cc * di)
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


def evaluate_model_on_2024(peft_path=''):
    """ 在2024年的期刊论文上评价 """
    if not peft_path:
        checkpoints_steps = dict()
        for checkpoints in os.listdir(model_save_path):
            if checkpoints.startswith('checkpoint'):
                beg_str, end_number = checkpoints.split("-")
                checkpoints_steps[int(end_number)] = checkpoints
        final_ck = checkpoints_steps[np.max(list(checkpoints_steps.keys()))]
        peft_path = os.path.join(model_save_path, final_ck)

    model = load_trained_model(peft_path)

    def test_func(data_2024):
        batch_size = 12
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
                batch = tokenizer(texts, max_length=640, truncation=True, return_tensors='pt', padding=True)
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

        tb = pt.PrettyTable()
        tb.field_names = ["编号", "期刊", "标题", "原创性得分", "论文类别"]
        idx = 0
        for journal, title, pred_score, pred_label in zip(journals, titles, pred_scores, pred_labels):
            idx += 1
            tb.add_row([idx, journal, title[:15], "{:.2f}".format(pred_score), pred_label])
        print(tb)

        journal_results = dict()
        journal_results["titles"] = titles
        journal_results["pred_labels"] = pred_labels
        journal_results["pred_scores"] = pred_scores
        utils.save_pickle(journal_results, os.path.join(data_2024_path, "{}.pkl".format(journal)))

    test_func(utils.BMJ_2024)
    test_func(utils.JCM_2024)
    test_func(utils.AJE_2024)
    test_func(utils.MSM_2024)
    # test_func(utils.Structure_INPUT)
    print("\n")
    # test_func(utils.Science_2024)
    # test_func(utils.Nature_2024)
    test_func(utils.JAMA_2024)
    test_func(utils.Lancet_2024)
    test_func(utils.CELL_2024)


def evaluate_model_on_nobel(peft_path=''):
    """ 在2024年的期刊论文上评价 """
    if not peft_path:
        checkpoints_steps = dict()
        for checkpoints in os.listdir(model_save_path):
            if checkpoints.startswith('checkpoint'):
                beg_str, end_number = checkpoints.split("-")
                checkpoints_steps[int(end_number)] = checkpoints
        final_ck = checkpoints_steps[np.max(list(checkpoints_steps.keys()))]
        peft_path = os.path.join(model_save_path, final_ck)
    model = load_trained_model(peft_path)

    def test_func(Nobel_papers, save_name):
        batch_size = 12
        pmid_nobels = list()
        titles = list()
        pred_scores = list()
        pred_labels = list()
        texts = list()
        for i, (pmid_nobel, title, abstract) in enumerate(Nobel_papers):
            text = title + abstract
            texts.append(text)
            pmid_nobels.append(pmid_nobel)
            titles.append(title)
            if len(texts) >= batch_size or i == len(Nobel_papers)-1:
                batch = tokenizer(texts, max_length=640, truncation=True, return_tensors='pt', padding=True)
                input_ids = batch['input_ids'].cuda()
                attention_mask = batch['attention_mask'].cuda()

                output = model(input_ids=input_ids, attention_mask=attention_mask)
                output_softmax = nn.functional.softmax(output.logits, dim=-1).detach().cpu().numpy()
                scores = compute_score(output_softmax)
                labels = np.argmax(output_softmax, axis=-1)

                for score, label in zip(scores, labels):
                    pred_scores.append(score)
                    pred_labels.append(str(label))
                texts = list()

        results = dict()
        for pmid_nobel, pred_score, pred_label in zip(pmid_nobels, pred_scores, pred_labels):
            results[pmid_nobel] = {'pred_score': pred_score, 'pred_label': pred_label}
        utils.save_json(results, os.path.join(data_nobel_path, save_name))

    def eval_func(read_name, save_name):
        eval_exp = utils.read_json(os.path.join(data_nobel_path, read_name))
        Nobel_papers = list()
        for pmid_nobel in eval_exp:
            target_info = eval_exp[pmid_nobel]['target_info']
            title = target_info['title']
            abstract = target_info['abstract']
            if len(abstract) <= 20:
                continue
            Nobel_papers.append((pmid_nobel, title, abstract))
        print("含有摘要的论文数目 : {} / {}".format(len(Nobel_papers), len(eval_exp)))
        test_func(Nobel_papers, save_name)

    # 医学诺奖论文
    eval_func("eval_exp.json",   "eval_exp_pred_results.json")
    eval_func("eval_ctl.json",   "eval_ctl_pred_results.json")
    eval_func("eval_ctl_r.json", "eval_ctl_r_pred_results.json")


if __name__ == "__main__":
    # train_model(8)
    # evaluate_model("test_set")
    # evaluate_model("valid_set")
    # evaluate_model("train_set")
    # print_results()

    evaluate_model_on_2024()
    # evaluate_model_on_nobel()
    pass
