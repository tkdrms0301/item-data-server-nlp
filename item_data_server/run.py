# __init__.py
# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request
from flask_ngrok import run_with_ngrok

import tensorflow as tf
import tensorflow_addons as tfa
from transformers import *
import os
import numpy as np
import tqdm
import threading

from model import KoBertTokenizer
import database
import kb

tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
RATING_COLUMN = 'rating'

# id 칼럼
ID_COLUMN = 'id'

# 긍부정 문장을 포함하고 있는 칼럼
DATA_COLUMN = 'document'

# 긍정인지 부정인지를 (1=긍정,0=부정) 포함하고 있는 칼럼
LABLE_COLUMN = 'lable'

SEQ_LEN = 64
BATCH_SIZE = 32

def convert_data(data_df):
    global tokenizer
    
    SEQ_LEN = 64 #SEQ_LEN : 버트에 들어갈 인풋의 길이
    
    tokens, masks, segments, targets = [], [], [], []

    for i in tqdm(range(len(data_df))):

        # token : 문장을 토큰화함
        token = tokenizer.encode(data_df[DATA_COLUMN][i], truncation=True, padding='max_length', max_length=SEQ_LEN)
       
        # 마스크는 토큰화한 문장에서 패딩이 아닌 부분은 1, 패딩인 부분은 0으로 통일
        num_zeros = token.count(0)
        mask = [1]*(SEQ_LEN-num_zeros) + [0]*num_zeros
        
        # 문장의 전후관계를 구분해주는 세그먼트는 문장이 1개밖에 없으므로 모두 0
        segment = [0]*SEQ_LEN

        # 버트 인풋으로 들어가는 token, mask, segment를 tokens, segments에 각각 저장
        tokens.append(token)
        masks.append(mask)
        segments.append(segment)
        
        # 정답(긍정 : 1 부정 0)을 targets 변수에 저장해 줌
        targets.append(data_df[LABLE_COLUMN][i])

    # tokens, masks, segments, 정답 변수 targets를 numpy array로 지정    
    tokens = np.array(tokens)
    masks = np.array(masks)
    segments = np.array(segments)
    targets = np.array(targets)

    return [tokens, masks, segments], targets

# 위에 정의한 convert_data 함수를 불러오는 함수를 정의
def load_data(pandas_dataframe):
    data_df = pandas_dataframe
    data_df[DATA_COLUMN] = data_df[DATA_COLUMN].astype(str)
    data_df[LABLE_COLUMN] = data_df[LABLE_COLUMN].astype(int)

    data_x, data_y = convert_data(data_df)
    return data_x, data_y

model = TFBertModel.from_pretrained("monologg/kobert", from_pt=True)

# 토큰 인풋, 마스크 인풋, 세그먼트 인풋 정의
token_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_word_ids')
mask_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_masks')
segment_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_segment')

# 인풋이 [토큰, 마스크, 세그먼트]인 모델 정의
bert_outputs = model([token_inputs, mask_inputs, segment_inputs])

bert_outputs

bert_outputs = bert_outputs[1]

# Rectified Adam 옵티마이저 사용

# 총 batch size * 4 epoch = 2344 * 4
opt = tfa.optimizers.RectifiedAdam(learning_rate=5.0e-5, total_steps = 2344*2, warmup_proportion=0.1, min_lr=1e-5, epsilon=1e-08, clipnorm=1.0)

sentiment_drop = tf.keras.layers.Dropout(0.5)(bert_outputs)
sentiment_first = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(sentiment_drop)
sentiment_model = tf.keras.Model([token_inputs, mask_inputs, segment_inputs], sentiment_first)
sentiment_model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(), metrics = ['accuracy'])

MODEL_NAME = 'fine-tuned-kobert-base'
MODEL_SAVE_PATH = os.path.join("_model", MODEL_NAME) # change this to your preferred location

sentiment_model.load_weights(MODEL_SAVE_PATH)

def predict_convert_data(data_df):
    global tokenizer
    tokens, masks, segments = [], [], []
    
    for i in tqdm(range(len(data_df))):

        token = tokenizer.encode(data_df[DATA_COLUMN][i], max_length=SEQ_LEN, truncation=True, padding='max_length')
        num_zeros = token.count(0)
        mask = [1]*(SEQ_LEN-num_zeros) + [0]*num_zeros
        segment = [0]*SEQ_LEN

        tokens.append(token)
        segments.append(segment)
        masks.append(mask)

    tokens = np.array(tokens)
    masks = np.array(masks)
    segments = np.array(segments)
    return [tokens, masks, segments]

# 위에 정의한 convert_data 함수를 불러오는 함수를 정의
def predict_load_data(pandas_dataframe):
    data_df = pandas_dataframe
    data_df[DATA_COLUMN] = data_df[DATA_COLUMN].astype(str)
    data_x = predict_convert_data(data_df)
    return data_x

def sentence_convert_data(data):
    global tokenizer
    tokens, masks, segments = [], [], []
    token = tokenizer.encode(data, max_length=SEQ_LEN, truncation=True, padding='max_length')
    
    num_zeros = token.count(0) 
    mask = [1]*(SEQ_LEN-num_zeros) + [0]*num_zeros 
    segment = [0]*SEQ_LEN

    tokens.append(token)
    segments.append(segment)
    masks.append(mask)

    tokens = np.array(tokens)
    masks = np.array(masks)
    segments = np.array(segments)
    return [tokens, masks, segments]

def data_evaluation_predict(sentence):
    data_x = sentence_convert_data(sentence)
    predict = sentiment_model.predict(data_x, verbose=0)
    predict_value = np.ravel(predict)
    predict_answer = np.round(predict_value,0).item()
    return predict_answer

app = Flask(__name__)
run_with_ngrok(app)

headers={
    'Content-type':'application/json',
    'Accept':'application/json'
}
def keyword_extraction_func(product_id, post):
    key_word_result = kb.input_preprocesing(post)

    db_class = database.Database()

    select_query = "SELECT * from data where product_id = {};".format(product_id)
    rows = db_class.execute_all(select_query)

    if rows == None:
        query = ""
        for i in key_word_result:
            query = "insert into data(count, vocab, product_id) values({}, {}, {});".format(1, str("'"+ i + "'"), product_id)
            db_class.execute(query)
        db_class.commit()
    else:
        query = ""
        for i in key_word_result:
            flag = False
            for j in rows:
                if j['vocab'] == i:
                    query = "update data set count = {} where data_id = {}".format(j['count'] + 1, j['data_id'])
                    db_class.execute(query)
                    flag = True
                    break
            if not flag:
                query = "insert into data(count, vocab, product_id) values({}, {}, {});".format(1, str("'"+ i + "'"), product_id)
                db_class.execute(query)
            flag = False
        db_class.commit()

def sentiment_classification_func(product_id, review):
    sentiment_classification_result = int(data_evaluation_predict(review)) # 0 이면 부정, 1이면 긍정
    db_class = database.Database()
    select_query = "SELECT * FROM pos_and_neg where product_id={};".format(product_id)
    row = db_class.execute_one(select_query)
    query = ""
    
    if(row == None and sentiment_classification_result == 0):
        # 새로 추가 (부정)
        query = "insert into pos_and_neg(product_id, positive, negative) values({}, {}, {});".format(product_id, 0, 1)
        db_class.execute(query)
        db_class.commit()
    elif(row == None and sentiment_classification_result == 1) :
        # 새로 추가 (긍정)
        query = "insert into pos_and_neg(product_id, positive, negative) values({}, {}, {});".format(product_id, 1, 0)
        db_class.execute(query)
        db_class.commit()
    elif(row != None and sentiment_classification_result == 0) :
        # 업데이트 (부정)
        query = "update pos_and_neg set negative = {} where product_id = {};".format(row['negative'] + 1, product_id)
        db_class.execute(query)
        db_class.commit()
    elif(row != None and sentiment_classification_result == 1) :
        # 업데이트 (긍정)
        query = "update pos_and_neg set positive = {} where product_id = {};".format(row['positive'] + 1, product_id)
        db_class.execute(query)
        db_class.commit()

@app.route("/hello", methods=['GET'])
def hello():
    return jsonify("hello"),200


@app.route("/data-search", methods=['POST'])
def data_search():
    values = request.get_json()

    input = values['input'] # 제품 id 입력
    
    input_list = input.split('|')
    preprocessing_input = kb.list_to_list_keyword(input_list)

    return jsonify(preprocessing_input), 200
    

# keyword-extraction
@app.route("/keyword-extraction", methods=['POST'])
def keyword_extraction():
    values = request.get_json()
    product_id = values['productId'] # 제품 id 입력
    post = values['content'] # 게시글 입력

    if product_id == 0:
        return jsonify(), 200
    
    keyword_extraction_thread = threading.Thread(target=keyword_extraction_func, args=(product_id, post))
    keyword_extraction_thread.daemon = True
    keyword_extraction_thread.start()

    return jsonify(), 200


# sentiment-classification
@app.route("/sentiment-classification", methods=['POST'])
def sentiment_classification():
    values = request.get_json()
    productId = values['productId']
    review = values['review']

    sentiment_classification_thread = threading.Thread(target=sentiment_classification_func, args=(productId, review))
    sentiment_classification_thread.daemon = True
    sentiment_classification_thread.start()
    

    return jsonify(), 200

print("run flask server")
app.run()
