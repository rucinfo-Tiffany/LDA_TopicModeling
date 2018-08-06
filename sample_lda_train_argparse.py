# -*- coding: utf-8 -*-
from __future__ import print_function
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
import sys
import csv
import nltk
import argparse
import pandas as pd

from time import time
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals import joblib


# 文本预处理
def textPrecessing(text):
    # 分词
    wordLst = nltk.word_tokenize(text)
    # 去除停用词
    filtered = [w for w in wordLst if w not in stopwords.words('english')]
    # 仅保留名词或特定POS
    refiltered = nltk.pos_tag(filtered)
    filtered = [w for w, pos in refiltered if pos.startswith('NN')]
    # 词干化
    ps = PorterStemmer()
    filtered = [ps.stem(w) for w in filtered]
    return " ".join(filtered)


def print_top_words(model, feature_names, n_top_words):
    # 打印topic_top_word
    # 注意：由于sklearn LDA函数限制，此函数中输出的topic_word矩阵未normalize, 需手动
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
    with open(os.path.join('lda_result', 'res_topic_word.csv'), 'w') as f:
        f.write("Topic, Top Word\n")
        for topic_idx, topic in enumerate(model.components_):
            f.write(str(topic_idx) + ',')
            topic_word_dist = [(feature_names[i], topic[i]) for i in topic.argsort()[:-n_top_words - 1:-1]]
            for word, score in topic_word_dist:
                f.write(word + '#' + str(score) + ';')
            f.write('\n')


def text_preprocess(dframe, args):
    #该区域仅运行首次，进行文本预处理
    docLst = []
    for desc in dframe[args.key]:
        docLst.append(textPrecessing(desc).encode('utf-8'))
    with open(args.text_pre_path, 'w') as f:
        for line in docLst:
            f.write(line+'\n')
    return docLst


def load_text_pre(args):
    # 得到存储的docLst,节省预处理时间
    docLst = []
    with open(args.text_pre_path, 'r') as f:
        for line in f.readlines():
            if line != '':
                docLst.append(line.strip())
    return docLst


def get_count_vector(docLst, tf_ModelPath, args):
    # 构建词汇统计向量并保存，仅运行首次
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                    max_features=args.n_features,
                                    stop_words='english')
    vec = tf_vectorizer.fit_transform(docLst)
    joblib.dump(tf_vectorizer, tf_ModelPath)
    return vec, tf_vectorizer


def load_count_vector(docLst, tf_ModelPath):
    #得到存储的tf_vectorizer,节省预处理时间
    tf_vectorizer = joblib.load(tf_ModelPath)
    vec = tf_vectorizer.fit_transform(docLst)
    return vec


def train_lda(vec, args):
    # 训练LDA并打印训练时间
    lda = LatentDirichletAllocation(n_topics=args.n_topics,
                                    max_iter=args.max_iter,
                                    learning_method='batch',
                                    evaluate_every=200,
                                    perp_tol=args.perp_tol)
    t0 = time()
    lda.fit(vec)
    print("# of Topic: %d, " % args.n_topics)
    print("done in %0.3fs, N_iter %d, " % ((time() - t0), lda.n_iter_), end=' ')
    print("Perplexity Score %0.3f" % lda.perplexity(vec))

    # 保存每个n_topics下的LDA模型，以便后续查看使用
    joblib.dump(lda, args.lda_model_path)  # 按照ddhhmm进行命名
    return lda


if __name__ == "__main__":
    # parse args

    parser = argparse.ArgumentParser(description="LDA training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input_path', '-i', type=str, default="sample_input.xlsx",
                        help='the path of input text corpus')

    parser.add_argument('--do_text_pre', '-D', type=int)
    parser.add_argument('--do_tf', type=int)

    parser.add_argument('--key', '-k', type=str,
            help="the key to find the content line in your excel/csv file")

    parser.add_argument('--text_pre_path', '-t', type=str,
        help="the file path to store text after preprocess")

    parser.add_argument('--n_topics', '-n', type=int, default=70)

    parser.add_argument('--n_features', '-f', type=int, default=2500)

    parser.add_argument('--n_top_words', '-w', type=int, default=40)

    parser.add_argument('--max_iter', '-m', type=int, default=5000)

    parser.add_argument('--perp_tol', '-p', type=float, default=0.01)

    parser.add_argument('--lda_model_path', '-l', type=str, default="model/lda.model",
            help='the file path to dump lda model')

    parser.add_argument('--tf_vector_path', '-v', type=str, default="model/tf_vec.model",
            help="the file path to dump tf vector model")

    parser.add_argument('--doc_topic_path', '-D', type=str, default="model/doc_topic_dist.csv",
            help="the file path to dump doc topic distribution")
    args = parser.parse_args()

    # 文本预处理
    docLst = list()
    if args.do_text_pre:
        # 读取每个项目的文字描述
        if '.xls' in args.input_path:
            dframe = pd.read_excel(args.input_path)
        elif '.csv' in args.input_path:
            dframe = pd.read_csv(args.input_path)
        else:
            print("Unknown input file type")
            sys.exit(1)
        docLst = text_preprocess(dframe, args)
    else:
        docLst = load_text_pre(args)

    # 加载/生成count vector词频统计
    tf, tf_vectorizer = get_count_vector(docLst, args) 
    
    # 训练LDA模型
    lda_model = train_lda(tf, args)

    # 保存并输出topic_word矩阵
    print("#########Topic-Word Distribution#########")
    tf_vectorizer._validate_vocabulary()
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda_model, tf_feature_names, args.n_top_words)
    
    # 保存doc_topic_distr 
    doc_topic_dist = lda_model.transform(tf)
    with open(args.doc_topic_path,  'w') as f:
        writer = csv.write(f)
        f.writerow(['Index', 'TopicDistribution'])
        for idx, dist in enumerate(doc_topic_dist):
            # 注意：由于sklearn LDA函数限制，此函数中输出的topic_word矩阵未normalize
            f.writerow([idx+1] + dist)
