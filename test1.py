from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import pickle
import jieba
import os
import re
import string
from gensim.models.word2vec import Word2Vec # 使用Word2Vec包
from sklearn.linear_model import SGDClassifier

file_path1 = 'D:/QQ/两万条训练语料+3000条测试语料/两万条训练语料+3000条测试语料/训练集/pos.txt'
file_path2 = 'D:/QQ/两万条训练语料+3000条测试语料/两万条训练语料+3000条测试语料/训练集/neg.txt'
outFile_path = './out/word2vecTestResult.txt'
#训练分词
def train_fenci():
    list_words = []

    test_text = open(file_path1,'r',encoding='utf-8').readlines()
    for line in test_text:
        # 清洗数据
        text = ''.join(line.split())
        # 实现目标文本中对正则表达式中的模式字符串进行替换
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）～-]+", "", text)
        # 利用jieba包自动对处理后的文本进行分词
        text_list = jieba.cut(text, cut_all=False)
        # 得到所有分解后的词
        list_words.append(' '.join(text_list))

    test_text = open(file_path2,'r',encoding='utf-8').readlines()
    for line in test_text:
        # 清洗数据
        text = ''.join(line.split())
        # 实现目标文本中对正则表达式中的模式字符串进行替换
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）～-]+", "", text)
        # 利用jieba包自动对处理后的文本进行分词
        text_list = jieba.cut(text, cut_all=False)
        # 得到所有分解后的词
        list_words.append(' '.join(text_list))
    return list_words

#测试分词
def test_fenci():
    FindPath1 = 'D:/QQ/两万条训练语料+3000条测试语料/两万条训练语料+3000条测试语料/测试集/test.txt_utf8'
    neg_words = []

    file1 = open(FindPath1,'r',encoding='utf-8')
    lines = file1.readlines()
    for line in lines:
        temp = ''.join(line.split())
        #实现目标文本中对正则表达式中的模式字符串进行替换
        temp = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）～-]+", "", temp)
        #利用jieba包自动对处理后的文本进行分词
        temp_list = jieba.cut(temp,cut_all=False)
        #得到所有分解后的词
        neg_words.append(' '.join(temp_list))
    return neg_words

train_words = train_fenci()
f_train_words = open('train.words.txt','w',encoding='utf-8')
f_train_labels = open('train.labels.txt','w',encoding='utf-8')
for line in train_words:
    f_train_words.write(line+'\n')
f_train_words.close()
print(len(open(file_path1,'r',encoding='utf-8').readlines()))
for i in range(len(open(file_path1,'r',encoding='utf-8').readlines())):
    f_train_labels.write('POS\n')
for i in range(len(open(file_path2,'r',encoding='utf-8').readlines())):
    f_train_labels.write('NEG\n')
f_train_labels.close()

test_words = test_fenci()
f_test_words = open('eval.words.txt','w',encoding='utf-8')
f_test_labels = open('eval.labels.txt','w',encoding='utf-8')
for line in test_words:
    f_test_words.write(line+'\n')
f_test_words.close()
for i in range(1500):
    f_test_labels.write('POS\n')
for i in range(1500):
    f_test_labels.write('NEG\n')
f_test_labels.close()