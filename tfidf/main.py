from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
import pickle
import jieba
import os
import re
import string

file_path1 = './dataset/train/neg.txt'
file_path2 = './dataset/train/pos.txt'
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
        test_list = jieba.cut(text, cut_all=False)
        # 得到所有分解后的词
        list_words.append(' '.join(test_list))
       
    test_text = open(file_path2,'r',encoding='utf-8').readlines()
    for line in test_text:
        # 清洗数据
        text = ''.join(line.split())
        # 实现目标文本中对正则表达式中的模式字符串进行替换
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）～-]+", "", text)
        # 利用jieba包自动对处理后的文本进行分词
        test_list = jieba.cut(text, cut_all=False)
        # 得到所有分解后的词
        list_words.append(' '.join(test_list))
    return list_words

#测试分词
def test_fenci():
    FindPath1 = './dataset/test/hotel_neg.txt_utf8'
    neg_words = []

    lines = open(FindPath1,'r',encoding='utf-8').readlines()
    for line in lines:
        temp = ''.join(line.split())
        #实现目标文本中对正则表达式中的模式字符串进行替换
        temp = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）～-]+", "", temp)
        #利用jieba包自动对处理后的文本进行分词
        temp_list = jieba.cut(temp,cut_all=False)
        #得到所有分解后的词
        neg_words.append(' '.join(temp_list))
    return neg_words


if __name__=='__main__':
    tfidf_vect = TfidfVectorizer(analyzer='word',stop_words=['是','的','在','这里'])
    train_tfidf = tfidf_vect.fit_transform(train_fenci())
    test_tfidf = tfidf_vect.transform(test_fenci())
    #words = tfidf_vect.get_feature_names()
    #print(words)
    #print(train_tfidf)
    #print(len(words))
    #print(train_tfidf)
    #print(tfidf_vect.vocabulary_)

    lr = SGDClassifier(loss='log',penalty='l1')
    lr.fit(train_tfidf,['neg']*len(open(file_path1,'r',encoding='utf-8').readlines())+
['pos']*len(open(file_path2,'r',encoding='utf-8').readlines()))
    y_pred = lr.predict(test_tfidf)
    print(y_pred)
    
    #model = SVC(kernel='rbf',verbose=True)
    #model.fit(train_tfidf,['neg']*3000+['pos']*3000)
    #model.predict(test_tfidf)

    