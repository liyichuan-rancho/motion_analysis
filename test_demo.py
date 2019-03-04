import gensim
import jieba
from sklearn.externals import joblib
import numpy as np

# 加载词向量
model = gensim.models.KeyedVectors.load_word2vec_format('./vectors.bin', binary=True)

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

def seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    stopwords = stopwordslist('./stopwords.txt')  # 这里加载停用词的路径
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr

def seg_text_to_vector(sentence):
    splited_text = sentence.split(" ")
    vector = np.zeros(200)
    num_of_word = 0
    for word in splited_text:
        try:
            a = model[word]
            for q in a:
                if np.isnan(q):
                    continue
        except:
            # print(j+"is not in vocabulary")
            continue

        vector += a
        num_of_word += 1

    if (num_of_word == 0) is True:
        return np.zeros(200)
    else:
        vector = vector/num_of_word
        return vector



clf = joblib.load("train_model.m")
# text = "这个书写得实在是高啊，太好了！"
text = "写得太烂了，垃圾"
seg_text = seg_sentence(text)
vector_input = seg_text_to_vector(seg_text)
a = clf.predict(vector_input.reshape(1,-1))
print(a)
