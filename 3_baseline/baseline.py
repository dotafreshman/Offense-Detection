import numpy
import re
import plotly.graph_objs as go
ngram=1

common=['I','me','my','you','your','it','they','we',
        'is','was','are','were','be','being',
        'like','can','do','have','has','make','get','go','want',
        'to','the','a','in','of','on','for','with','so','just','and','or','at','as','some','any']
common2=['','I','me','my','you','your','it','i','to',
        'is','be','being','do','have','the','a','in','of','at']
common3=['','I','the']

def tokenize(s):
    res=s.split(' ')
    return res


def ngram_tokenize(s):
    res = []
    step1 = s.split(' ')
    for i in range(len(step1) - ngram+1):
        res.append(tuple([step1[i+j] for j in range(ngram)]))
    res.append(tuple([step1[len(step1)-j] for j in reversed(range(1,ngram if ngram<len(step1) else len(step1)))]))
    return res


#remove common word





def train(lis_x,lis_y,alpha=0):
    res=[{},{}]      #word occurence for p/n
    wordC=[0,0]        #word count for p/n
    p=0                 #p case num
    for line in range(len(lis_x)):
        np=int(lis_y[line])
        p+=np
        for i in lis_x[line]:
            if i not in res[0]:res[0][i]=0
            if i not in res[1]:res[1][i]=0
            res[np][i]+=1
            wordC[np]+=1


    px=res[0].copy();pxy0=res[0].copy();pxy1=res[1].copy()
    py1=p/len(lis_y);py0=1-py1


    for i in res[0]:
        #print((res[0][i]+alpha)/(wordC[0]+alpha*(wordC[0]+wordC[1])),(res[1][i]+alpha)/(wordC[1]+alpha*(wordC[0]+wordC[1])))
        px[i]=(res[0][i]+res[1][i])/(wordC[0]+wordC[1])
        pxy0[i]=(res[0][i]+alpha)/(wordC[0]+alpha*(wordC[0]+wordC[1]))
        pxy1[i]=(res[1][i]+alpha)/(wordC[1]+alpha*(wordC[0]+wordC[1]))

    return [px,py0,py1,pxy0,pxy1]

def train3(lis_x,lis_y,alpha=0):
    res=[{},{},{}]      #word occurence for p/n
    wordC=[0,0,0]        #word count for p/n
    p0=0;p1=0                 #p case num
    for line in range(len(lis_x)):
        np=int(lis_y[line])
        if np==0:p0+=1;
        elif np==1:p1+=1;

        for i in lis_x[line]:
            if i not in res[0]:res[0][i]=0
            if i not in res[1]:res[1][i]=0
            if i not in res[2]: res[2][i] = 0
            res[np][i] += 1
            wordC[np] += 1


    px=res[0].copy();pxy0=res[0].copy();pxy1=res[1].copy();pxy2=res[2].copy()
    py1=p1/len(lis_y);py0=p0/len(lis_y);py2=1-py1-py0


    for i in res[0]:
        #print((res[0][i]+alpha)/(wordC[0]+alpha*(wordC[0]+wordC[1])),(res[1][i]+alpha)/(wordC[1]+alpha*(wordC[0]+wordC[1])))
        px[i]=(res[0][i]+res[1][i]+res[2][i])/(wordC[0]+wordC[1]+wordC[2])
        pxy0[i]=(res[0][i]+alpha)/(wordC[0]+alpha*(wordC[0]+wordC[1]+wordC[2]))
        pxy1[i]=(res[1][i]+alpha)/(wordC[1]+alpha*(wordC[0]+wordC[1]+wordC[2]))
        pxy2[i] = (res[2][i] + alpha) / (wordC[2] + alpha * (wordC[0] + wordC[1]+wordC[2]))

    return [px,py0,py1,py2,pxy0,pxy1,pxy2]


def classify(x,py0,py1,pxy0,pxy1):
    y=[]
    for line in x:

        a = py0
        b=py1
        for i in line:
            if (i not in pxy0):
                continue


            a *= pxy0[i]
            b*=pxy1[i]
        if (a > b)or((a==0)and(b==0)):
            y.append(0)
        else:
            y.append(1)
    return y

def classify3(x,py0,py1,py2,pxy0,pxy1,pxy2):
    y=[]
    for line in x:

        a = py0
        b=py1
        c=py2
        for i in line:
            if (i not in pxy0):
                continue


            a *= pxy0[i]
            b*=pxy1[i]
            c*=pxy2[i]
        if ((a>=b)and(a>=c))or((a==0)and(b==0)and(c==0)):
            y.append(0)
        elif((b>=a)and(b>=c)):
            y.append(1)
        elif((c>=a)and(c>=b)):y.append(2)
    return y

def alpha_test(x_use,y_use,st,ed,intv):
    step=(ed-st)/intv
    plotx=[];ploty=[]
    print("alpha","accuracy")
    for i in numpy.arange(st,ed,step):
        px, py0, py1, pxy0, pxy1 = train(lis_x, lis_y, i)
        y = classify(x_use, py0, py1, pxy0, pxy1)
        acc = 0
        for j in range(len(y_use)):

            if y[j] == int(y_use[j]):
                acc += 1

        acc = acc / len(y_use)
        print(i,acc)
        plotx.append(i);ploty.append(acc)


    fig=go.Figure()
    fig.add_trace(go.Scatter(x=plotx,y=ploty))

    fig.update_layout(
        title="Accuracy vs. Smoothing_alpha",
        xaxis_title="smoothing_alpha",
        yaxis_title="accuracy"
        )
    fig.show()
    return

def common_word_remove(dic1,dic2,para):
    while (True) and (len(dic1) > para) and (len(dic2) > para):
        akey = sorted(dic1.items(), key=lambda item: item[1], reverse=True)[:para]
        bkey = sorted(dic2.items(), key=lambda item: item[1], reverse=True)[:para]
        same_tf = False
        for i in range(para):
            for j in range(para):
                if akey[i][0] == bkey[j][0]:
                    dic1.pop(akey[i][0]);
                    dic2.pop(bkey[j][0])
                    same_tf = True
        if (not same_tf): break
    return dic1,dic2

#读取train.txt，训练
x_file=open('../OLID/olid-training-v1.0.tsv','r',encoding='UTF-8')

lis_x=[];lis_y=[]
first=0
for i in x_file:
    if(first==0):first=1;continue
    else:first=1
    line=i.split('\t')
    lis_x.append(tokenize(line[1]))
    if line[2]=="OFF":
        lis_y.append(1)
    else:lis_y.append(0)

px,py0,py1,pxy0,pxy1=train(lis_x,lis_y,0)
x_file.close();



##读取dev.txt
x_test_file=open('../OLID/testset-levela.tsv','r',encoding='UTF-8')
y_test_file=open('../OLID/labels-levela.csv','r',encoding='UTF-8')
test_x=[];test_y=[]
first=0
for i in x_test_file:
    if first==0:first=1;continue
    line=i.split('\t')
    test_x.append(tokenize(line[1]))
for i in y_test_file:
    line=i.split(',')
    if line[1]=="OFF":
        test_y.append(1)
    else:test_y.append(0)
x_test_file.close();y_test_file.close()



op="test"
if(op=="test"):

#进行classify
    x_use=test_x;y_use=test_y;
    y=classify(x_use,py0,py1,pxy0,pxy1)
    acc=0
    TP=0;FN=0;FP=0;TN=0
    print(len(y_use))
    print(y)
    for i in range(len(y_use)):
        if (y[i]==1)and(int(y_use[i])==1):TP+=1
        elif (y[i]==1)and(int(y_use[i])==0):FP+=1
        elif (y[i] == 0) and (int(y_use[i]) == 1):FN += 1
        elif (y[i] == 0) and (int(y_use[i]) == 0):TN += 1
        if y[i]==int(y_use[i]):
            acc+=1
    # alpha_test(x_use, y_use, 0, 1.3, 30)
    acc=acc/len(y_use)
    print("accuracy",acc)
    print('TP: {}'.format(TP))
    print('FP: {}'.format(FP))
    print('FN: {}'.format(FN))
    print('TN: {}'.format(TN))
    F1=2*TP/(2*TP+FN+FP);print('F1: {}'.format(F1))
elif(op=='output'):
    y_realTestFile=open('y_test_Bayes.csv','w')
    x_use = kaggle_x;
    y = classify(x_use, py0, py1, pxy0, pxy1)
    acc = 0
    y_realTestFile.writelines('Id,Category\n')
    for i in range(len(kaggle_x)):
        y_realTestFile.writelines('{},{}\n'.format(i,int(y[i])))
    y_realTestFile.close()
print("###########################################")
print("##########################################")
print("#######################target")
x_file=open('../OLID/olid-training-v1.0.tsv','r',encoding='UTF-8')

lis_x=[];lis_y=[]
first=0
for i in x_file:
    if(first==0):first=1;continue
    else:first=1
    line=i.split('\t')
    lis_x.append(tokenize(line[1]))
    if line[3]=="TIN":
        lis_y.append(1)
    else:lis_y.append(0)

px,py0,py1,pxy0,pxy1=train(lis_x,lis_y,0)
x_file.close();



##读取dev.txt
x_test_file=open('../OLID/testset-levelb.tsv','r',encoding='UTF-8')
y_test_file=open('../OLID/labels-levelb.csv','r',encoding='UTF-8')
test_x=[];test_y=[]
first=0
for i in x_test_file:
    if first==0:first=1;continue
    line=i.split('\t')
    test_x.append(tokenize(line[1]))
for i in y_test_file:
    line=i.split(',')
    if line[1]=="TIN":
        test_y.append(1)
    else:test_y.append(0)
x_test_file.close();y_test_file.close()



op="test"
if(op=="test"):

#进行classify
    x_use=test_x;y_use=test_y;
    y=classify(x_use,py0,py1,pxy0,pxy1)
    acc=0
    TP=0;FN=0;FP=0;TN=0
    print(len(y_use))
    print(y)
    for i in range(len(y_use)):
        if (y[i]==1)and(int(y_use[i])==1):TP+=1
        elif (y[i]==1)and(int(y_use[i])==0):FP+=1
        elif (y[i] == 0) and (int(y_use[i]) == 1):FN += 1
        elif (y[i] == 0) and (int(y_use[i]) == 0):TN += 1
        if y[i]==int(y_use[i]):
            acc+=1
    # alpha_test(x_use, y_use, 0, 1.3, 30)
    acc=acc/len(y_use)
    print("accuracy",acc)
    print('TP: {}'.format(TP))
    print('FP: {}'.format(FP))
    print('FN: {}'.format(FN))
    print('TN: {}'.format(TN))
    F1=2*TP/(2*TP+FN+FP);print('F1: {}'.format(F1))
elif(op=='output'):
    y_realTestFile=open('y_test_Bayes.csv','w')
    x_use = kaggle_x;
    y = classify(x_use, py0, py1, pxy0, pxy1)
    acc = 0
    y_realTestFile.writelines('Id,Category\n')
    for i in range(len(kaggle_x)):
        y_realTestFile.writelines('{},{}\n'.format(i,int(y[i])))
    y_realTestFile.close()

print("###########################################")
print("##########################################")
print("#######################target group")

x_file=open('../OLID/olid-training-v1.0.tsv','r',encoding='UTF-8')

lis_x=[];lis_y=[]
first=0
lis_y0=0;lis_y1=0
for i in x_file:
    if(first==0):first=1;continue
    else:first=1
    line=i.split('\t')
    lis_x.append(tokenize(line[1]))
    if line[4]=="NULL\n":lis_y.append(0)
    elif line[4]=="IND\n":lis_y.append(1)
    # elif line[4]=="GRP\n":lis_y.append(2)
    else:lis_y.append(2);


px,py0,py1,py2,pxy0,pxy1,pxy2=train3(lis_x,lis_y,0.1)
# file = open("px.txt", 'w')
# for i in px:file.write(str(i))
# file.close()
# print("px",px)
# file = open("py0.txt", 'w')
# for i in py0:file.write(str(i))
# file.close()
# file = open("py2.txt", 'w')
# for i in py2:file.write(str(i))
# file.close()
# file = open("py1.txt", 'w')
# for i in py1:file.write(str(i))
# file.close()
# file = open("pxy0.txt", 'w')
# for i in pxy0:file.write(str(i))
# file.close()
# file = open("pxy1.txt", 'w')
# for i in pxy1:file.write(str(i))
# file.close()
# file = open("pxy2.txt", 'w')
# for i in pxy2:file.write(str(i))
# file.close()
# x_file.close();



##读取dev.txt
x_test_file=open('../OLID/testset-levelc.tsv','r',encoding='UTF-8')
y_test_file=open('../OLID/labels-levelc.csv','r',encoding='UTF-8')
test_x=[];test_y=[]
first=0
for i in x_test_file:
    if first==0:first=1;continue
    line=i.split('\t')
    test_x.append(tokenize(line[1]))

for i in y_test_file:
    line=i.split(',')
    if line[1]=="NULL\n":test_y.append(0);
    elif line[1]=="IND\n":test_y.append(1)
    # elif line[1] == "GRP\n":test_y.append(2)
    else:test_y.append(2);





op="test"
if(op=="test"):

#进行classify
    x_use=test_x;y_use=test_y;
    y=classify3(x_use,py0,py1,py2,pxy0,pxy1,pxy2)
    acc=0
    TP=0;FN=0;FP=0;TN=0
    print(len(y_use))
    print(y)
    for i in range(len(y_use)):
        if y[i]==int(y_use[i]):
            acc+=1
    # alpha_test(x_use, y_use, 0, 1.3, 30)
    acc=acc/len(y_use)
    print("accuracy",acc)

elif(op=='output'):
    y_realTestFile=open('y_test_Bayes.csv','w')
    x_use = kaggle_x;
    y = classify3(x_use, py0, py1,py2, pxy0, pxy1,pxy2)
    acc = 0
    y_realTestFile.writelines('Id,Category\n')
    for i in range(len(kaggle_x)):
        y_realTestFile.writelines('{},{}\n'.format(i,int(y[i])))
    y_realTestFile.close()
