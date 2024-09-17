#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 19:27:25 2024

@author: aixuexi
"""
import re
import os
import json
import math
import time
import random
import datetime
import multiprocessing
import nltk
import copy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import prettytable as pt
import matplotlib
from matplotlib import rcParams
from tqdm import tqdm
from itertools import combinations
from clickhouse_driver import Client
from scipy import stats

from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

from pandas.plotting import table

import utils, parse, compute_di_score


# 数据库参数
params = utils.params


# 生成数据存放路径ss
score_cls_dir = "./CLS_score_samples"

label2labelname = {
    0: 'Developmental papers', 
    1: 'General papers', 
    2: 'Disruptive papers', 
}

label2labelname_ch = {
    0: '巩固', 
    1: '一般', 
    2: '颠覆', 
}


def get_CORE_title_abstract_ref(samples_t):
    """ 查询core论文的标题, 摘要, 参考文献 """
    client = Client(**params)
    pmids = list(samples_t.keys())
    total_size = len(pmids)
    batch_size = 10000
    loop_num = math.ceil(total_size / batch_size)
    for i in range(loop_num):
        # 以bacth分批查询
        start = i * batch_size
        end = min((i + 1) * batch_size, total_size)
        pmids_string = utils.search_pmid_string(pmids[start: end])
        sql_query = "SELECT pmid, title, abstract, references FROM pubmedmeta where pmid IN {}".format(pmids_string)
        results = client.execute(sql_query)
        # 检查有无摘要
        without_abstract = list()
        for pmid, title, abstract, references in results:
            if len(abstract) == 0:
                without_abstract.append(pmid)
                continue
            else:
                references = json.loads(references)
                pmids_of_ref = list()
                if len(references) != 0:
                    for ref in references:
                        if 'pmid' in ref:
                            if len(ref['pmid']) > 0:
                                pmids_of_ref.append(ref['pmid'])
                
                samples_t[pmid]['title'] = title
                samples_t[pmid]['abstract'] = abstract
                samples_t[pmid]['references'] = pmids_of_ref
        # 剔除无摘要的论文
        for pmid in without_abstract:
            del samples_t[pmid] 
    client.disconnect()
    return samples_t


def get_SIMILAR_title_abstract_ref_MP(samples_t, top_num=10):
    """ 多进程调用 get_ref_title_abstract 函数 """
    batch_size = 10000
    mp_num = 4
    total_keys = list(samples_t.keys())
    total_size = len(total_keys)
    batch_size_of_mp = math.ceil(total_size / mp_num)
    beg_idx_of_mp = 0
    end_idx_of_mp = 0
    # 创建进程池
    pool = multiprocessing.Pool(processes=mp_num)
    # 存放更新结果
    results = list()  
    for mp_i in range(mp_num):
        end_idx_of_mp = min(beg_idx_of_mp + batch_size_of_mp, total_size)
        keys_of_mp = total_keys[beg_idx_of_mp: end_idx_of_mp]
        samples_t_mp = {pmid: samples_t[pmid] for pmid in keys_of_mp}
        results.append(pool.apply_async(get_SIMILAR_title_abstract_ref, (samples_t_mp, batch_size, top_num)))
        beg_idx_of_mp = end_idx_of_mp
    pool.close()
    pool.join()
    # 集成结果
    for res in results:
        samples_t_mp = res.get()
        for pmid in samples_t_mp:
            samples_t[pmid] = samples_t_mp[pmid]
    return samples_t   
            

def get_SIMILAR_title_abstract_ref(samples_t, batch_size, top_num):
    """ 获取论文的参考文献的标题和摘要, 
        并确定top_num的相似文献 
    """
    client = Client(**params)
    pids_list = list()
    refs_list = list()
    for i, pmid_core in enumerate(samples_t):
        pids_list.append(pmid_core)
        refs_list += samples_t[pmid_core]['references']
        
        if len(refs_list) >= batch_size or i == len(samples_t)-1:
            # 查找论文的参考文献的标题和摘要
            refs_list = set(refs_list)
            pmids_string = utils.search_pmid_string(refs_list)
            results = client.execute("SELECT pmid, title, abstract FROM pubmedmeta where pmid IN {}".format(pmids_string))
            refs_info = dict()
            for pmid_ref, title_ref, abstract_ref in results:
                refs_info[pmid_ref] = {'title': title_ref, 'abstract': abstract_ref}
            
            # 检查共被引情况 - 相似文献是共被引次数多的论文 - 故影响DI中的j
            # 此外, 影响DI的是高被引论文 - 影响DI中的k
            refs_list = list(refs_list)
            pmids_string = utils.search_pmid_string(pids_list + refs_list)
            results = client.execute("SELECT citing_pmid, cited_pmid, year FROM pubmedcc where cited_pmid IN {}".format(pmids_string))
            citing_dict = dict() # Key: citing_pmid Value: a list of cited_pmids
            cited_dict = dict()  # Key: cited_pmid  Value: a list of citing_pmids
            for citing_pmid, cited_pmid, year in results:
                if citing_pmid not in citing_dict:
                    citing_dict[citing_pmid] = list()
                citing_dict[citing_pmid].append(cited_pmid)
                if cited_pmid not in cited_dict:
                    cited_dict[cited_pmid] = list()
                cited_dict[cited_pmid].append(citing_pmid)
            
            # 为每篇论文确定相似论文
            for pmid_core in pids_list:
                title_core = samples_t[pmid_core]['title']
                abstract_core = samples_t[pmid_core]['abstract']
                refs_core = samples_t[pmid_core]['references']
                tokens_core = set(nltk.word_tokenize(title_core.lower()))
                
                refs_cocite_score = {pmid_ref: 0 for pmid_ref in refs_core}
                refs_title_score = {pmid_ref: 0 for pmid_ref in refs_core}
                refs_abstract_score = {pmid_ref: 0 for pmid_ref in refs_core}
                
                # 确定pmid_core和其reference的共被引次数
                cc_core = 0
                if pmid_core in cited_dict:
                    for pmid_i in cited_dict[pmid_core]:   # pmid_i 引用过 pmid_core
                        cc_core += 1                       # pmid_core的引用次数
                        for pmid_j in citing_dict[pmid_i]: # 判断pmid_i 是否 引用过 pmid_core的参考文献
                            if pmid_j in refs_cocite_score:# pmid_core 和 pmid_j 共被引
                                refs_cocite_score[pmid_j] += 1
                
                # 根据标题和摘要确定相似论文        
                cc_refs = dict()
                for pmid_ref in refs_core:
                    cc_ref = 0
                    if pmid_ref in cited_dict:
                        cc_ref = len(cited_dict[pmid_ref])
                    cc_refs[pmid_ref] = cc_ref
                    
                    if pmid_ref in refs_info:
                        title_ref = refs_info[pmid_ref]['title']
                        abstract_ref = refs_info[pmid_ref]['abstract']
                        tokens_ref = set(nltk.word_tokenize(title_ref.lower()))
                        # 标题重叠词数目越多越相似
                        score_title = len(tokens_core.intersection(tokens_ref)) / np.sqrt(max(len(tokens_core), 1) * max(len(tokens_ref), 1))
                        refs_title_score[pmid_ref] = score_title
                        # 优先选取有摘要的论文
                        if len(abstract_ref) > 50:
                            score_abstract = 1
                        else:
                            score_abstract = -1
                        refs_abstract_score[pmid_ref] = score_abstract                            
                    else:
                        refs_title_score[pmid_ref] = 0
                        refs_abstract_score[pmid_ref] = -1
                            
                # 确定排名前10的参考文献 (有摘要 + 共被引 + 标题名字重叠)
                final_similar_score = dict()
                for pmid_ref in refs_core:
                    final_similar_score[pmid_ref] = 1000 * refs_abstract_score[pmid_ref] + 10 * refs_cocite_score[pmid_ref] + 1 * refs_title_score[pmid_ref]
                final_similar_score = [(pmid_ref, final_similar_score[pmid_ref]) for pmid_ref in final_similar_score]
                final_similar_score = sorted(final_similar_score, key=lambda x: x[-1], reverse=True)
                similars = [pmid_ref for pmid_ref, _ in final_similar_score[: top_num]]                
                
                samples_t[pmid_core]['cc_core'] = cc_core
                samples_t[pmid_core]['cc_refs'] = [cc_refs[pmid_ref] for pmid_ref in similars]
                # 相似论文pmid
                samples_t[pmid_core]['pmid_similars'] = similars
                # 相似论文的共被引次数
                samples_t[pmid_core]['cocite_similars'] = [refs_cocite_score[pmid_ref] for pmid_ref in similars]
                # 相似论文的标题
                title_similars = list()
                for pmid_ref in similars:
                    if pmid_ref in refs_info:
                        title_similars.append(refs_info[pmid_ref]['title'])
                    else:
                        title_similars.append('')
                samples_t[pmid_core]['title_similars'] = title_similars
                # 相似论文的摘要
                abstract_similars = list()
                for pmid_ref in similars:
                    if pmid_ref in refs_info:
                        abstract_similars.append(refs_info[pmid_ref]['abstract'])
                    else:
                        abstract_similars.append('')                
                samples_t[pmid_core]['abstract_similars'] = abstract_similars
            # 清空缓存
            pids_list = list()
            refs_list = list()
            refs_info = list()
    client.disconnect()
    return samples_t
    
   
def epsilon_greedy_sample_algorithm(samples, sample_size, epsilon=0.5):
    """ 根据置信区间 epsilon-greedy 采样"""
    acutal_sample_size = min(len(samples), sample_size)
    greedy_size = int(acutal_sample_size * (1 - epsilon))
    random_size = acutal_sample_size - greedy_size
    # 贪婪样本 - 根据置信区间采样
    alternative_samples = sorted(samples, reverse=True, key=lambda x: x[-1])
    greedy_samples = alternative_samples[:greedy_size]
    # 随机样本 - 在区间内随机采样
    random_samples = random.sample(alternative_samples[greedy_size: ], random_size)
    return greedy_samples, random_samples


def construct_raw_dataset_for_classification():
    """ pubmed中根据颠覆性得分抽取论文
        (1) 过滤参考文献不足的论文 - 影响DI的计算 - 重要步骤
        (2) epsilon-greedy采样, 优先选择取间中间的样本 - 目前无效果
        (3) c10 * di==0中过滤c10==0的样本 - 无数据证实其是颠覆性或发展性
        (4) 获取参考文献的摘要 (参考文献作为相似文献, 标题的相似度)
    """
    client = Client(**params)
    
    for t in np.arange(2000, 2020):
        # t时间发表的论文 
        results = client.execute("SELECT pmid, pubdate FROM pubmedmeta where SUBSTRING(pubdate, 1, 4)=={}".format("\'" + str(t) + "\'"))
        pmids_t = list()
        for pmid, pubdate in results:
            if pmid != '':
                pmids_t.append(pmid)
                
        # 过滤参考文献数目不足的论文
        total_size = len(pmids_t)
        batch_size = 10000
        loop_num = math.ceil(total_size / batch_size)
        results = list()
        for i in range(loop_num):
            start = i * batch_size
            end = min((i + 1) * batch_size, total_size)
            results += client.execute("SELECT * FROM pubmeddilabel where pmid IN {}".format(utils.search_pmid_string(pmids_t[start: end])))
        pmids_t = dict()
        for pmid, c10, di, pi, pj, ref_num, label, lable_ci in results:
            # *** 影响DI的计算
            if ref_num >= 10:
                pmids_t[pmid] = dict()
                pmids_t[pmid]['cc'] = c10
                pmids_t[pmid]['di'] = di
                pmids_t[pmid]['pi'] = pi
                pmids_t[pmid]['pj'] = pj
                pmids_t[pmid]['ref_num'] = ref_num
                pmids_t[pmid]['label'] = label
                pmids_t[pmid]['lable_ci'] = lable_ci
                
        # 过滤引用数目不足的论文 - 因为无引用无法说明 c10 * di == 0
        label2pmid_c10 = dict()
        for pmid in pmids_t:
            cc = pmids_t[pmid]['cc']
            di = pmids_t[pmid]['di']
            pi = pmids_t[pmid]['pi']
            pj = pmids_t[pmid]['pj']
            ref_num = pmids_t[pmid]['ref_num']
            label = pmids_t[pmid]['label']
            lable_ci = pmids_t[pmid]['lable_ci'] 
            
            if label not in label2pmid_c10: label2pmid_c10[label] = list()
            
            # *** 需充分数据证实其是颠覆性或发展性
            if label in {0, 2}:
               # development papers, disruptive papers 是高质量论文
               # 有充足引用保证DI合理性
               if cc >= 15:
                  label2pmid_c10[label].append((pmid, lable_ci))  # pmid, 标签置信度
            else:
               # general papers
               pk = 1 - pi - pj
               if pk > 0.8:
                   # 说明development贡献和disruptive贡献均不足
                   if cc >= 1 and cc < 15:
                       # 说明影响力不足
                       label2pmid_c10[label].append((pmid, lable_ci))  # pmid, 标签置信度 
        
        num_t = 3000
        label_num = len(label2pmid_c10)
        sample_size = int(num_t / label_num) * 15  # 更多样本保证有摘要, 有相似文献
        
        # 根据c10 * di构建的集合 epsilon-greedy采样
        samples_t_c10 = dict()
        for label_cc_x_di in label2pmid_c10:
            greedy_samples, random_samples = epsilon_greedy_sample_algorithm(label2pmid_c10[label_cc_x_di], sample_size)
            for pmid, ci in greedy_samples + random_samples:
                samples_t_c10[pmid] = dict()
                samples_t_c10[pmid]['label'] = str(label_cc_x_di)
                samples_t_c10[pmid]['pubyear'] = str(t)
        
        # 获取标题和摘要
        samples_t_c10 = get_CORE_title_abstract_ref(samples_t_c10)
        # 获取相似文献 - 标题和摘要
        samples_t_c10 = get_SIMILAR_title_abstract_ref_MP(samples_t_c10)
        
        # 保证每类样本数
        label_num_c10_total  = {str(i): 0 for i in range(label_num)}
        label_num_c10_sample = {str(i): 0 for i in range(label_num)}
        # 保证相似文献数目
        samples_t_c10_pmids  = [(pid, len(samples_t_c10[pid]['pmid_similars'])) for pid in samples_t_c10]
        samples_t_c10_pmids  = sorted(samples_t_c10_pmids, key=lambda x: x[-1], reverse=True)
        # 
        samples_t_c10_filter = dict()
        for pmid, similars_num in samples_t_c10_pmids:
            label = samples_t_c10[pmid]['label']
            label_num_c10_total[label] += 1
            if label_num_c10_total[label] <= int(num_t / label_num):
                samples_t_c10_filter[pmid] = samples_t_c10[pmid]
                label_num_c10_sample[label] += 1                
        print("{}: {}".format(t, {i: "{}/{}".format(label_num_c10_sample[str(i)], label_num_c10_total[str(i)]) for i in range(label_num)}))
        utils.save_json(samples_t_c10_filter,  os.path.join(score_cls_dir, "samples_c10_{}.json".format(t)))
    
    client.disconnect()
    
    # 合并结果
    samples_c10 = dict()
    for t in np.arange(2000, 2020):
        samples_c10_t = utils.read_json(os.path.join(score_cls_dir, "samples_c10_{}.json".format(t)))
        for pmid in samples_c10_t:
            samples_c10[pmid] = samples_c10_t[pmid]
    utils.save_json(samples_c10, os.path.join(score_cls_dir, "samples_c10.json"))
       

#%%
def construct_abstract_format_balanced_dataset():
    """ 相同摘要 构建结构化版本 和 非结构化版本 - 2024-3-6
        发现类别中结构化摘要数目不平衡 - 影响结果稳定性(加入结构标签影响结果精度)
    """
    samples_c10 = utils.read_json(os.path.join(score_cls_dir, "samples_c10.json"))  
    # samples_c10 = utils.read_json(os.path.join(score_cls_dir, "samples_c10_balance.json"))
    
    STRUCTURES = ["BACKGROUND", "METHODS", "RESULTS", "CONCLUSIONS", "OBJECTIVE"]
    STRUCTURES = set([i.lower() for i in STRUCTURES])
    pattern = re.compile(r"(?=\b({})\b)".format("|".join(STRUCTURES)))
    
    # 统计每一类中结构化摘要数目 (发现类别中结构化摘要数目不平衡)
    label2WhetherStructure = dict()
    for pid in samples_c10:
        abstract = samples_c10[pid]['abstract'].lower()
        label = samples_c10[pid]['label']
    
        if label not in label2WhetherStructure:
            label2WhetherStructure[label] = {'YS': list(), 'NS': list()}
    
        found_terms = re.findall(pattern, abstract)
        found_terms = set(found_terms)
        if len(found_terms) >= 3:
            label2WhetherStructure[label]['YS'].append(pid)  # 是结构化摘要
        else:
            label2WhetherStructure[label]['NS'].append(pid)  # 非结构化摘要
    
    # 注重类别平衡 + 注重结构功能和非结构功能的摘要平衡
    tb = pt.PrettyTable()
    tb.field_names = ["种类", "结构化摘要", "非结构化摘要", "累计"]
    for label in label2WhetherStructure:
        label_yes_structure = len(label2WhetherStructure[label]['YS'])
        label_no_structure  = len(label2WhetherStructure[label]['NS'])
        tb.add_row([label2labelname[int(label)], label_yes_structure, label_no_structure, label_yes_structure + label_no_structure])    
    print(tb)
    
    
    # 下采样: (1) 避免结构化摘要和非结构化摘要的数目不平衡; 
    label2WhetherStructure_downsample = dict()
    YS_downsample_num = 3000
    NS_downsample_num = 13000
    for label in label2WhetherStructure:
        YS_total_num = len(label2WhetherStructure[label]['YS'])
        NS_total_num = len(label2WhetherStructure[label]['NS'])
        label2WhetherStructure_downsample[label] = dict()
        label2WhetherStructure_downsample[label]['YS'] = random.sample(label2WhetherStructure[label]['YS'], min(YS_total_num, YS_downsample_num))
        label2WhetherStructure_downsample[label]['NS'] = random.sample(label2WhetherStructure[label]['NS'], min(NS_total_num, NS_downsample_num))
        
    # 构建重复样本: (2) 迫使模型关注除结构标签以外的信息
    samples_c10_downsample = dict()
    for label in label2WhetherStructure_downsample:
        
        # 结构化摘要 转成 非结构化摘要, 同时保留结构化摘要和非结构化摘要
        for pid in label2WhetherStructure_downsample[label]['YS']:
            # 结构化摘要版本
            struture_abstract = samples_c10[pid]['abstract'].lower()
            samples_c10[pid]['abstract'] = struture_abstract
            
            # 构建非结构化摘要版本
            abstract_split = struture_abstract.split("\n")
            abstract_parsed = dict()
            try:
                idx_paragraph = 0
                for oneline in abstract_split:
                    if oneline:
                        if oneline in STRUCTURES:
                            structure = oneline
                            if structure not in abstract_parsed:
                                abstract_parsed[structure] = list()
                        else:
                            abstract_parsed[structure].append((oneline, idx_paragraph)) # 结构的内容, 段落的编号
                            idx_paragraph += 1
                abstract = list()
                for structure in abstract_parsed:
                    abstract += abstract_parsed[structure]
                abstract = sorted(abstract, key=lambda x: x[-1])
                abstract = " ".join([sentence for sentence, idx in abstract]).strip()
            except:
                # 部分论文第一段无小标题(后续部分存在标题), 但训练数据充足, 故舍弃它们
                continue
            
            # 添加***非结构化摘要版本***
            pid_info_copy = copy.copy(samples_c10[pid])
            pid_info_copy['abstract'] = abstract
            samples_c10_downsample["COPY" + pid] = pid_info_copy
            
            # 添加***结构化摘要版本***
            samples_c10_downsample[pid] = samples_c10[pid]
        
        # 非结构化摘要全部保存
        for pid in label2WhetherStructure_downsample[label]['NS']:
            samples_c10[pid]['abstract'] = samples_c10[pid]['abstract'].lower()
            samples_c10_downsample[pid] = samples_c10[pid]
    

    # 重新统计每一类中结构化摘要数目 (发现类别中结构化摘要数目已被平衡)
    label2WhetherStructure = dict()
    for pid in samples_c10_downsample:
        abstract = samples_c10_downsample[pid]['abstract'].lower()
        label = samples_c10_downsample[pid]['label']
    
        if label not in label2WhetherStructure:
            label2WhetherStructure[label] = {'YS': list(), 'NS': list()}
    
        found_terms = re.findall(pattern, abstract)
        found_terms = set(found_terms)
        if len(found_terms) >= 3:
            label2WhetherStructure[label]['YS'].append(pid)  # 是结构化摘要
        else:
            label2WhetherStructure[label]['NS'].append(pid)  # 非结构化摘要
    
    # 注重类别平衡 + 注重结构功能和非结构功能的摘要平衡
    tb = pt.PrettyTable()
    tb.field_names = ["种类", "结构化摘要", "非结构化摘要"]
    for label in label2WhetherStructure:
        label_yes_structure = len(label2WhetherStructure[label]['YS'])
        label_no_structure  = len(label2WhetherStructure[label]['NS'])
        tb.add_row([label2labelname[int(label)], label_yes_structure, label_no_structure])    
    print(tb)
    utils.save_json(samples_c10_downsample, os.path.join(score_cls_dir, "samples_c10_balance.json"))

