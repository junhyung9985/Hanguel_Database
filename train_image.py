import os
from os import walk
from os.path import isfile, join
from os import listdir
import glob

train_path = "../Hanguel_Database/Image_train"
test_path = "../Hanguel_Database/Image_test"
#ts = "/content/drive/My Drive/DeepLearning_group3/Hanguel_Database/Image_test"

def findFiles(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    return onlyfiles

test_file = findFiles(test_path)
train_file = findFiles(train_path)

from PIL import Image
import numpy as np
import csv

test_dir = "../Hanguel_Database/test"
train_dir = "../Hanguel_Database/train"

label = open('labels.csv', 'r', encoding='utf-8')
rdr = csv.reader(label)
for line in rdr:
    line = line[0]
    save_ts = test_dir+"/"+line
    save_tr = train_dir+"/"+line
    if (os.path.exists(save_ts)==False): os.makedirs(os.path.join(save_ts))
    if (os.path.exists(save_tr)==False): os.makedirs(os.path.join(save_tr))

for f in train_file:
    fi = f.split('_')
    class_name = fi[0]
    num = (fi[1].split('.'))[0]
    ls = list(csv.reader(open(join(train_path, f), newline=''), delimiter=','))
    lst = []
    for tmp in ls:
        temp = [a for a in tmp if a!='']
        lst.append(temp)
    print(lst)
    matrix = np.array(lst).astype("uint8")
    imgObj = Image.fromarray(matrix)
    resized_imgObj = imgObj.resize((28, 28))
    resized_imgObj.save("../Hanguel_Database/train/{}/{}.jpg".format(class_name,num))
    '''
    imgObj.show()
    resized_imgObj.show()
    resized_matrix = np.asarray(resized_imgObj)
    np.savetxt(test_dir+'/resized.csv', resized_matrix, fmt='%3.0d', delimiter=',')
    '''
