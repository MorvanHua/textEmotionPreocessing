import jieba # 结巴分词
import numpy as np # numpy处理向量
import pandas as pd 

neg = pd.read_excel('D:/tools/temp_shixun/word2vec/neg2.xls',header = None) # 读取负面词典
pos = pd.read_excel('D:/tools/temp_shixun/word2vec/pos2.xls',header = None) # 读取正面词典

neg['words'] = neg[0].apply(lambda x: jieba.lcut(x)) # 负面词典分词
pos['words'] = pos[0].apply(lambda x: jieba.lcut(x)) # 正面词典分词
print(neg['words'])

x = np.concatenate((pos['words'],neg['words'])) # 将原句子丢弃掉，并合并正面和负面分词结果

from gensim.models.word2vec import Word2Vec # 使用Word2Vec包
w2v = Word2Vec(vector_size=300, min_count=10) # 初始化高维向量空间，300个维度，一个词出现的次数小于10则丢弃
w2v.build_vocab(x) # build高维向量空间

w2v.train(x,total_examples=w2v.corpus_count,epochs=10) # 训练，获取到每个词的向量

def total_vec(words): # 获取整个句子的向量
  vec = np.zeros(300).reshape((1,300))
  for word in words:
    try:
      vec += w2v.wv[word].reshape((1,300))
    except KeyError:
      continue
  return vec

train_vec = np.concatenate([total_vec(words) for words in x]) # 计算每一句的向量，得到用于训练的数据集，用来训练高维向量模型

import joblib
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

model = SVC(kernel='rbf',verbose=True)
model.fit(train_vec,['[pos]']*len(pos)+['[neg]']*len(neg)) # 训练SVM模型
joblib.dump(model,'D:/tools/temp_shixun/word2vec/svm_model.pkl') # 保存训练好的模型

# 对测试数据进行情感判断
def svm_predict():
  df = pd.read_csv('D:/tools/temp_shixun/word2vec/neg_test.csv') # 读取测试数据
  mode = joblib.load('D:/tools/temp_shixun/word2vec/svm_model.pkl') # 读取支持向量机模型
  for string in df['内容']:
    # 对评论分词
    words = jieba.lcut(str(string))
    words_vec = total_vec(words)
    result = mode.predict(words_vec) 
    # 实时返回积极或消极结果
    #print(string,result[0]) 

svm_predict()