#!/usr/bin python3
# -*- encoding: utf-8 -*-
# @Author : cobra
# @File : preprocess.py
# @Time : 2020/9/20 23:06
import pandas as pd
import jieba
import keras
import json
from keras import backend as K

from sklearn.preprocessing import LabelEncoder
from model.text_cnn import simple_text_cnn
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences


def convert_json2excel(infile, outfile):
    """
    转换json格式为excel
    :param infile:
    :param outfile:
    :return:
    """
    f = open(infile, encoding='utf-8')
    df_out = pd.DataFrame()
    label_id_list = []
    label_list = []
    text_list = []
    keyword_list = []

    for line in f.readlines():
        line = line.strip()
        line = json.loads(line)
        label_id_list.append(line['label'])
        label_list.append(line['label_desc'])
        text_list.append(line['sentence'])
        keyword_list.append(line['keywords'])
    df_out['label_id'] = label_id_list
    df_out['label'] = label_list
    df_out['text'] = text_list
    df_out['keywords'] = keyword_list
    df_out.to_excel(outfile)

def cut_x(x):

    return list(jieba.cut(x)) # jieba.cut(x)返回值是一个generator，所以前面要加list


def excel_preprocess(infile):
    df = pd.read_excel(infile)
    function_cut = lambda x: list(jieba.cut(x))
    df['words'] = df['text'].apply(cut_x)
    return df

def train_test_split_dataset(text,label,test_size = 0.1):
    return train_test_split(text,label,test_size = 0.1)



def get_token_id(text_list):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_list)
    vocab = tokenizer.word_index
    return tokenizer

def pad_sequence(text,max_len =50,padding='post'):
    text_pad = pad_sequences(text,maxlen=50,padding=padding)
    return text_pad

def self_loss(y_true, y_pred):
    return K.sum((y_pred + y_true), axis=-1)


if __name__ == '__main__':
    # convert_json2excel('../data/train.json', '../data/train.xlsx')

    df = excel_preprocess('../data/train_demo.xlsx')
    tokenizer= get_token_id(list(df['words']))
    x_train, x_test, y_train, y_test = train_test_split(list(df['words']), list(df['label']), test_size=0.1)
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)
    x_train_pad = pad_sequence(x_train,max_len =50)
    x_test_pad = pad_sequence(x_test, max_len=50)
    encoder = LabelEncoder()
    y_train_id = encoder.fit_transform(y_train)
    model = simple_text_cnn((50,),len(tokenizer.word_index))
    model.summary()
    model.compile(loss=self_loss, optimizer='adam', metrics=['accuracy'])
    one_hot_labels = keras.utils.to_categorical(y_train_id, num_classes=15)  # 将标签转换为one-hot编码
    model.fit(x_train_pad, one_hot_labels, batch_size=128, epochs=10)


