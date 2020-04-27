from keras.preprocessing import text,sequence
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


np.random.seed(123)
torch.manual_seed(123)

corpus=[]
file=open('../OLID/olid-training-v1.0.tsv','r',encoding='UTF-8')
y_trainA=[];y_trainB=[];y_trainC=[]
for i in file:
    if(len(corpus)==10):print(corpus[-1])
    a=i.split('\t')
    corpus.append(a[1].lower())
    if a[2]=="OFF":y_trainA.append(1)
    else:y_trainA.append(0)
    if a[3]!="NULL":
        if a[3]=="TIN":y_trainB.append(1)
        else:y_trainB.append(0)
    if a[4]!="NULL\n":

        if a[4]=="GRP\n":y_trainC.append(2)
        elif a[4]=="IND\n":y_trainC.append(1)
        elif a[4]=="OTH\n":y_trainC.append(0)
        else:print(a[4],"fuck");continue;
file.close()


TP=0;FP=0;TN=0;FN=0
y_pred=[]
ct=0
for i in y_trainA:
    if i==1:ct+=1
    else:ct-=1
for i in y_trainA:
    if ct>=0:y_pred.append(1)
    else:y_pred.append(0)
    if y_pred[-1]==1:
        if i==1:TP+=1;
        else: FP+=1
    else:
        if i==0:TN+=1
        else: FN+=1
print(accuracy_score(y_pred,y_trainA))
print(y_pred)
all=TP+FP+TN+FN
print("TP:",TP/all,"FP:",FP/all,"TN:",TN/all,"FN:",FN/all)


TP=0;FP=0;TN=0;FN=0
y_pred=[]
ct=0
for i in y_trainB:
    if i==1:ct+=1
    else:ct-=1
for i in y_trainB:
    if ct>=0:y_pred.append(1)
    else:y_pred.append(0)
    if y_pred[-1]==1:
        if i==1:TP+=1;
        else: FP+=1
    else:
        if i==0:TN+=1
        else: FN+=1
print(accuracy_score(y_pred,y_trainB))
print(y_pred)
all=TP+FP+TN+FN
print("TP:",TP/all,"FP:",FP/all,"TN:",TN/all,"FN:",FN/all)


y_pred=[]
c0=0;c1=0;c2=0
for i in y_trainC:
    if i==0:c0+=1
    elif i==1:c1+=1
    else:c2+=1
for i in y_trainC:
    if max(c0,c1,c2)==c0:y_pred.append(0)
    elif max(c0,c1,c2)==c1:y_pred.append(1)
    else:y_pred.append(2)
print(accuracy_score(y_pred,y_trainC))