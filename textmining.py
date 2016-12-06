#encoding=utf-8
import csv
import pandas as pd
import jieba
import jieba.analyse
import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer 

#=====initial and set txt=====
jieba.set_dictionary('dict.txt.big.txt')
jieba.analyse.set_stop_words("stop_words.txt")
jieba.analyse.set_idf_path("idf.txt.big.txt");
df = pd.read_csv('testdata.csv',encoding='UTF-8')
rows=len(df.axes[0])# #100
Matrix = ["" for y in range(100)]#0-19

#=====process tags=====
for row in range(0,100):
	content = df.values[row][1]
	tags = jieba.analyse.extract_tags(content,topK=15)
	Matrix[row]=(" ".join(tags))

#=====process sklearn.feature_extraction=====
vectorizer=CountVectorizer() 
transformer=TfidfTransformer()
tfidf=transformer.fit_transform(vectorizer.fit_transform(Matrix))
word=vectorizer.get_feature_names()
weight=tfidf.toarray()

#=====process kmeans to n_clusters=====
X = np.array(weight)
kmeans = KMeans(n_clusters=7, random_state=0).fit(X)

#=====conclusion for article=====
for i in range(0,100): 
	print ("The",(i+1),"article belongs to ",kmeans.labels_[i],"-->tags",Matrix[i])

#=====conclusion for cluster=====
cluster = [[] for y in range(7)]#0-19
for i in range(0,100): 
	for j in range(0,7): 
		if kmeans.labels_[i] == j:
			cluster[j].append(Matrix[i])

for c in range(0,7):
	print ("The",(c+1),"cluster has\n",",\n".join(cluster[c]),"\n")


