#!/usr/bin/python3
import gensim
import jieba
import pymysql
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import numpy as np

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

#加载停用词词库
stopwords = stopwordslist('./stopwords.txt')  # 这里加载停用词的路径

# 打开数据库连接
db = pymysql.connect("192.168.2.99", "root", "1234", "python")
cursor = db.cursor()
sql = "SELECT * FROM `new_zhangyue` WHERE star=1 LIMIT 20000;"
commit = []
label = []
# 使用 execute()  方法执行 SQL 查询
try:
   # 执行 SQL 语句
    cursor.execute(sql)
   # 获取所有记录列表
    results = cursor.fetchall()
    for row in results:
        commit.append(row[4])
        label.append(0)
except:
   print ("Error: unable to fetch data")

sql2 = "SELECT * FROM `new_zhangyue` WHERE star=5 LIMIT 20000;"
try:
   # 执行 SQL 语句
    cursor.execute(sql2)
   # 获取所有记录列表
    results = cursor.fetchall()
    for row in results:
        commit.append(row[4])
        label.append(1)
except:
   print ("Error: unable to fetch data")
# 关闭数据库连接
db.close()
print('load data over')

#加载词向量
model = gensim.models.KeyedVectors.load_word2vec_format('./vectors.bin',binary=True)

# jieba.load_userdict('userdict.txt')
# 创建停用词list
# def stopwordslist(filepath):
#     stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
#     return stopwords


seg_commit = []
count = -1          #数组角标
for row in commit:
    count += 1
    i = jieba.cut(row)
    vector = np.zeros(200)
    num_of_word = 0
    for j in i:
        # if j in stopwords:
        #     continue
        try:
            a = model[j]
            for q in a:
                if np.isnan(q):
                    continue
        except:
            # print(j+"is not in vocabulary")
            continue

        vector = vector + a
        num_of_word += 1

    if (num_of_word == 0) is True:
        try:
            del label[count]
            count -= 1
        except:
            print("{}这句话  删除标签有误".format(row))
            continue
        continue
    vector = vector/num_of_word

    for q in vector:
        if np.isnan(q):
            print('1')

    seg_commit.append(vector)

#逻辑回归
lr_model = LogisticRegression(C=0.1, max_iter=100)
lr_model.fit(seg_commit, label)
scores = cross_val_score(lr_model, seg_commit, label, cv=10, scoring='roc_auc')
print("\n逻辑回归 10折交叉验证得分: \n", scores)
print("\n逻辑回归 10折交叉验证平均得分: \n", np.mean(scores))

#保存模型
joblib.dump(lr_model, "train_model.m")
print("保存模型完毕")

# SVM
# from sklearn.svm import SVC
# from sklearn.model_selection import cross_val_score
# model = SVC(C=4, kernel='rbf')
# svm_model = SVC(kernel='linear', probability=True)
# svm_model.fit(seg_commit, label)
# scores = cross_val_score(svm_model, seg_commit, label, cv=10, scoring='roc_auc')
# print("\nSVM 10折交叉验证得分: \n", scores)
# print("\nSVM 10折交叉验证平均得分: \n", np.mean(scores))


