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

file_path1 = './dataset/train/neg.txt'
file_path2 = './dataset/train/pos.txt'
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
        list_words.append(list(text_list))

    test_text = open(file_path2,'r',encoding='utf-8').readlines()
    for line in test_text:
        # 清洗数据
        text = ''.join(line.split())
        # 实现目标文本中对正则表达式中的模式字符串进行替换
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）～-]+", "", text)
        # 利用jieba包自动对处理后的文本进行分词
        text_list = jieba.cut(text, cut_all=False)
        # 得到所有分解后的词
        list_words.append(list(text_list))
    return list_words

#测试分词
def test_fenci():
    FindPath1 = './dataset/test/test.txt_utf8'
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
        neg_words.append(list(temp_list))
    return neg_words

train_words = train_fenci()
w2v = Word2Vec(vector_size=300, min_count=10) # 初始化高维向量空间，300个维度，一个词出现的次数小于10则丢弃
w2v.build_vocab(train_words) # build高维向量空间
w2v.train(train_words,total_examples=w2v.corpus_count,epochs=10) # 训练，获取到每个词的向量

def total_vec(words): # 获取整个句子的向量
    vec = np.zeros(300).reshape((1,300))
    for word in words:
        try:
            vec += w2v.wv[word].reshape((1,300))
        except KeyError:
            continue
    return vec
train_vec = np.concatenate([total_vec(words) for words in train_words]) # 计算每一句的向量，得到用于训练的数据集，用来训练高维向量模型

# SVM模型
model = SVC(kernel='rbf',verbose=True)
# SGDC模型(随机梯度下降)
#model = SGDClassifier(loss='log',penalty='l1')
model.fit(train_vec,['[neg]']*len(open(file_path1,'r',encoding='utf-8').readlines())+
['[pos]']*len(open(file_path2,'r',encoding='utf-8').readlines())) # 训练模型  

test_words = test_fenci()
sum_counter = 0
pos_right = 0
pos_wrong = 0
neg_right = 0
neg_wrong = 0

for line in test_words:
    words_vec = total_vec(line)
    result = model.predict(words_vec)
    if sum_counter<1500:
        if result[0]=='[pos]':
            pos_right += 1
        elif result[0]=='[neg]':
            pos_wrong += 1
    else:
        if result[0]=='[neg]':
            neg_right += 1
        elif result[0]=='[pos]':
            neg_wrong += 1
    sum_counter += 1

print('pos_right ',pos_right,'\tpos_wrong ',pos_wrong)
print('neg_right ',neg_right,'\tneg_wrong',neg_wrong)