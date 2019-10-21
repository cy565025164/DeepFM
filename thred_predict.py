#!/usr/bin/env python
# encoding: utf-8
import os
from sklearn.metrics import classification_report

thred = 0.3
testfile = "./result/DeepFM_result"
y_test, y_pred = [], []
n = 0
for line in open(testfile, 'r'):
    line = line.strip().split("\t")
    n += 1
    if n == 1:
        continue
    if len(line) != 3:
        continue
    pin, y, score = line[0], line[1], float(line[2])
    y_test.append(y)
    if score >= thred:
        y_pred.append("1")
    else:
        y_pred.append("0")
print(classification_report(y_test, y_pred, digits=3))