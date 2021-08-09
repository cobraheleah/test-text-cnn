#!/usr/bin python3
# -*- encoding: utf-8 -*-
# @Author : cobra
# @File : text_cnn.py
# @Time : 2020/9/22 23:22
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Embedding, Conv1D, MaxPooling1D, concatenate

from keras.models import Model


def simple_text_cnn(input_shape, vocab_len):
    main_input = Input(shape=input_shape, dtype='float64')
    # 词嵌入（使用预训练的词向量）
    embedder = Embedding(vocab_len+ 1, 300, input_length=50, trainable=False)
    embed = embedder(main_input)
    # 词窗大小分别为3,4,5
    cnn1 = Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
    cnn1 = MaxPooling1D(pool_size=48)(cnn1)
    cnn2 = Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
    cnn2 = MaxPooling1D(pool_size=47)(cnn2)
    cnn3 = Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
    cnn3 = MaxPooling1D(pool_size=46)(cnn3)
    # 合并三个模型的输出向量
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
    flat = Flatten()(cnn)
    drop = Dropout(0.2)(flat)
    main_output = Dense(15, activation='softmax', name='dsfdsfs')(drop)
    model = Model(inputs=main_input, outputs=main_output)
    return model
