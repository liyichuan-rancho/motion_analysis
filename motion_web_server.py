import json

import functools
import gensim
import numpy as np
from flask import Flask, make_response

# from api import QQMusic, BaiduMusic
# from knowledge_graph import KnowledgeGraph
# from database import Database
# from tools import cors, time_me
# from neo4j.api import QQMusic, BaiduMusic
# from neo4j.knowledge_graph import KnowledgeGraph
# from neo4j.database import Database
# from neo4j.tools import cors, time_me
from sklearn.externals import joblib

from test_demo import seg_sentence, seg_text_to_vector

app = Flask(__name__)

# 加载词向量
model = gensim.models.KeyedVectors.load_word2vec_format('./vectors.bin', binary=True)

def cors(func):
    # 跨域
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        response = make_response(func(*args, **kwargs))
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response

    return wrapper

@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/motionanalysis/<str>')
@cors
def query_mention(str):
    return to_json(predict(str))


# @app.route('/findInfo/<str>')
# @time_me()
# @cors
# def find_info(str):
#     return to_json(kg.find_info(str))


def to_json(o):
    return json.dumps(o, ensure_ascii=False)

def predict(sentence):
    # text = "这个书写得实在是高啊，太好了！"
    text = sentence
    seg_text = seg_sentence(text)
    vector_input = seg_text_to_vector(seg_text)
    if (vector_input == np.zeros(200)).all() == True:
        return sentence + "：无法判断"
    a = clf.predict(vector_input.reshape(1, -1))
    if a == [1]:
        print("{}:正面评论".format(sentence))
        return sentence + "：正面评论"
    elif a == [0]:
        print("{}:负面评论".format(sentence))
        return sentence + "：负面评论"
    else:
        print("{}:预测结果出错".format(sentence))
        return sentence + "：预测结果出错"

if __name__ == '__main__':
    # app.wsgi_app = ProxyFix(app.wsgi_app)
    clf = joblib.load("train_model.m")
    app.run(host='0.0.0.0', port=8099, threaded=True)