#!/usr/bin/env python
# -*- coding: utf-8 -*-
# from collections import namedtuple
# import numpy as np


trainfile = './data/train.csv'
testfile = './data/test.csv'

def loaddata(train):
    # for line in open(trainfile):
    # Model = namedtuple("Model", 'price sqft_living')
    if train == 'train':
        trainf = open(trainfile)
    else:
        trainf = open(testfile)
    titles = trainf.readline()
    title = titles.split(',')
    price_id = 0
    sqft_living_id = 0
    for i,name in enumerate(title):
        if name == 'price':
            price_id = i
        if name == 'sqft_living':
            sqft_living_id = i
    x_train = []
    for line in trainf:
        data = line.split(',')
        price = data[price_id]
        sqft_living = data[sqft_living_id]
        # x_ = Model(price, sqft_living)
        x_ = [price, sqft_living]
        x_train.append(x_)

    trainf.close()
    return x_train

if __name__ == "__main__":
    loaddata()

