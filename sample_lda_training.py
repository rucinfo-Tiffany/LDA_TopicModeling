# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 15:05:32 2017

@author: Tiffany

- 读取每篇文章
- 文本预处理
    - 分词
    - 去除停用词
    - 词性筛选
    - 词干化
- LDA训练
    - CountVectorizer统计词频，文本向量化
    - k=the number of topics 取值范围[5,100]，以步长为5递增
    - 计算相应的perplexity score，找到较优的k值
    - 每10次训练打印一次perplexity score
- 保留最优的LDA模型并打印
"""

import os
import nltk
import pandas as pd
import matplotlib.pyplot as plt

from time import time
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals import joblib

#参数定义
n_features = 2500
n_top_words = 40
n_topics = range(5, 105, 5) #话题数从5到100，根据perplexity决定
perplexityLst = [1.0]*len(n_topics)
originFilePath = 'xxx.xlsx' #从xlsx文件中读取dframe
textPre_FilePath = 'xxx.txt' #保存文本预处理后的文件，避免重复预处理
tf_ModelPath = os.path.join('model', 'tfVector2500.model') #保存词频模型
lda_ModelPath =  os.path.join('model', 'ldaModels.model') #保存训练的lda模型

#读取每个项目的文字描述
dframe = pd.read_excel(originFilePath)

#文本预处理
def textPrecessing(text):
    #分词
    wordLst = nltk.word_tokenize(text)
    #去除停用词
    filtered = [w for w in wordLst if w not in stopwords.words('english')]
    #仅保留名词或特定POS
    refiltered =nltk.pos_tag(filtered)
    filtered = [w for w, pos in refiltered if pos.startswith('NN')]
    #词干化
    ps = PorterStemmer()
    filtered = [ps.stem(w) for w in filtered]
    return " ".join(filtered)

#打印topic_top_word
#注意：由于sklearn LDA函数限制，此函数中输出的topic_word矩阵未normalize
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic #%d:" % topic_idx
        print " ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]])
    print
    with open(os.path.join('lda_result', 'res_topic_word.csv'), 'w') as f:
        f.write("Topic, Top Word\n")
        for topic_idx, topic in enumerate(model.components_):
            f.write(str(topic_idx)+',')
            topic_word_dist = [(feature_names[i], topic[i]) 
                    for i in topic.argsort()[:-n_top_words - 1:-1]]
            for word, score in topic_word_dist:
                f.write(word+'#'+str(score)+';')
            f.write('\n')

#==============================================================================
# #该区域仅运行首次，进行文本预处理
# docLst = []
# for desc in dframe['Project Description']:
#     docLst.append(textPrecessing(desc).encode('utf-8'))
# with open(textPre_FilePath, 'w') as f:
#     for line in docLst:
#         f.write(line+'\n')
#==============================================================================
#得到存储的docLst,节省预处理时间
docLst = []
with open(textPre_FilePath, 'r') as f:
    for line in f.readlines():
        if line != '':
            docLst.append(line.strip())

#构建词汇统计向量并保存，仅运行首次
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')
tf = tf_vectorizer.fit_transform(docLst)
joblib.dump(tf_vectorizer,tf_ModelPath )
#==============================================================================
# #得到存储的tf_vectorizer,节省预处理时间
# tf_vectorizer = joblib.load(tf_ModelPath)
# tf = tf_vectorizer.fit_transform(docLst)
#==============================================================================


#训练LDA并打印训练时间
lda_models = []
for idx, n_topic in enumerate(n_topics):
    lda = LatentDirichletAllocation(n_topics=n_topic,
                                    max_iter=8000,
                                    learning_method='batch',
                                    evaluate_every=200,
                                    perp_tol=0.01)
    t0 = time()
    lda.fit(tf)
    perplexityLst[idx] = lda.perplexity(tf)
    lda_models.append(lda)
    print "# of Topic: %d, " % n_topics[idx],
    print "done in %0.3fs, N_iter %d, " % ((time() - t0), lda.n_iter_),
    print "Perplexity Score %0.3f" % perplexityLst[idx]


#打印最佳模型
best_index = perplexityLst.index(min(perplexityLst)) 
best_n_topic = n_topics[best_index]
best_model = lda_models[best_index]
print "Best # of Topic: ", best_n_topic
print "Best Model: "

#保存每个n_topics下的LDA模型，以便后续查看使用
joblib.dump(lda_models, lda_ModelPath)#按照ddhhmm进行命名
    
#保存并输出topic_word矩阵
print "#########Topic-Word Distribution#########"
tf_vectorizer._validate_vocabulary()
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(best_model, tf_feature_names, n_top_words)

#保存perplexity结果并绘制perplexity的趋势
with open(os.path.join('lda_result', 'res_perplexity.txt'), 'w') as f:
    perplexityLst_str = '\n'.join([str(x) for x in perplexityLst])
    f.write(perplexityLst_str)
    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(n_topics, perplexityLst)
ax.set_xlabel("# of topics")
ax.set_ylabel("Approximate Perplexity")
plt.grid(True)
plt.savefig(os.path.join('lda_result', 'perplexityTrend.png'))
plt.show()

#保存doc_topic_distr 
doc_topic_dist = best_model.transform(tf)
with open(os.path.join('lda_result', 'res_doc_topic.csv'), 'w') as f:
    f.write('ProjectID,TopicDistribution\n')
    for idx, dist in enumerate(doc_topic_dist):
        #注意：由于sklearn LDA函数限制，此函数中输出的topic_word矩阵未normalize
        dist = [str(x) for x in dist]
        f.write(str(idx+1)+',')
        f.write(','.join(dist)+'\n')




