import os 
import jieba
from collections import Counter

train_data = eval(open('train.list', 'r').read())
test_data = eval(open('test.list', 'r').read())
tmp_list = []

print(len(train_data))
temp = []
for d in train_data:
    temp += d['aspects']
print(Counter(temp).most_common())
lists = [asp[0] for asp in Counter(temp).most_common()]
print({l:i for i,l in enumerate(lists)})
