#!/usr/bin/env python
# encoding: utf-8

import sys,random

cps_99 = ["99-10", "99-20", "99-15"]
cps_59 = ["59-6", "59-10"]

n = 0
dct = dict()
for line in open("tes", 'r'):
    line = line.strip().split('\t')
    n += 1
    if n == 1:
        continue
    if len(line) != 4:
        print("error n:", n)
        break
    pin, money, score = line[0], float(line[1]), float(line[-1])
    if pin not in dct or dct[pin][1] < score:
        dct[pin] = [money, score]
    del pin, money, score
dct = sorted(dct.items(), key=lambda x:x[1][1], reverse=True)

num_99_15 = 0
m = 0
for k in dct:
    pin, money, score = k[0], k[1][0], k[1][1]
    cps = ""
    is_dx = "0"
    if money>31.3 and money<76.7:
        i = random.randint(0,1)
        cps = cps_59[i]
    else:
        i = random.randint(0,2)
        if i == 2:
            num_99_15 += 1
        if num_99_15 > 93000:
            i = random.randint(0, 1)
        cps = cps_99[i]
    m += 1
    if m > 50000 and m < 150001:
        is_dx = "1"
    print ("\t".join([pin, str(score), cps, is_dx]))
