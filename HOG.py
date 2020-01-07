import numpy as np
import cv2
import os
from sklearn.neighbors import KNeighborsClassifier

count = 0
winSize = (60,60)
blockSize = (30,30)
blockStride = (15,15)
cellSize = (15,15)
nbins = 9

hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
#WinSize, BlockSize, BlockStride, CellSize, NBins

char_id = os.listdir('./Thai44Consonants/Train/')
label_train = np.zeros(1)
for i in range(len(char_id)):
    print(str(i)+'/'+str(len(char_id)), end="\r")
    num_char_id = os.listdir('./Thai44Consonants/Train/'+str(char_id[i]))
    for j in range(len(num_char_id)):
        im = cv2.imread('./Thai44Consonants/Train/'+str(char_id[i])+'/'+str(num_char_id[j]),0)

        im = cv2.resize(im, (50, 50))
        im = cv2.GaussianBlur(im, (3, 3), 0)
        h = hog.compute(im)

        if count == 0:
            features_train = h.reshape(1,-1)
            label_train = int(char_id[i])
        else:
            features_train = np.concatenate((features_train,h.reshape(1,-1)),axis = 0)
            label_train = np.append(label_train,int(char_id[i]))
        count = count+1

neigh = KNeighborsClassifier(n_neighbors=3,weights = 'distance',n_jobs =-1)
neigh.fit(features_train,label_train)

count = 0
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
                                #WinSize, BlockSize, BlockStride, CellSize, NBins

char_id = os.listdir('./Thai44Consonants/Test/')
label_test = np.zeros(1)
for i in range(len(char_id)):
    print(str(i)+'/'+str(len(char_id)), end="\r")
    num_char_id = os.listdir('./Thai44Consonants/Test/'+str(char_id[i]))
    for j in range(len(num_char_id)):
        im = cv2.imread('./Thai44Consonants/Test/'+str(char_id[i])+'/'+str(num_char_id[j]),0)

        im = cv2.resize(im, (50, 50))
        im = cv2.GaussianBlur(im, (3, 3), 0)
        h = hog.compute(im)

        if count == 0:
            features_test = h.reshape(1,-1)
            label_test = int(char_id[i])
        else:
            features_test = np.concatenate((features_test,h.reshape(1,-1)),axis = 0)
            label_test = np.append(label_test,int(char_id[i]))
        count = count+1

neigh.score(features_test, label_test)