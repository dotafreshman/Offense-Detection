from keras.preprocessing import text,sequence
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

para_path="4_para/off.pkl"
glove_para="glove/glove.twitter.27B.25d.txt"
# para_path="4_para/off_self.pkl"
# glove_para="word2vec_self_para/word_embedding.txt"
vec_dim=25
mode="read"


np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.backends.cudnn.deterministic=True

corpus=[]
file=open('OLID/olid-training-v1.0.tsv','r',encoding='UTF-8')
y_train=[]
for i in file:
    if(len(corpus)==10):print(corpus[-1])
    a=i.split('\t')
    corpus.append(a[1].lower())
    if a[2]=="OFF":y_train.append(1)
    else:y_train.append(0)
file.close()

#print(corpus[1:4])

tokenizer = text.Tokenizer(num_words = 1000)
tokenizer.fit_on_texts(corpus)
word_index = tokenizer.word_index
x_train = tokenizer.texts_to_sequences(corpus)
x_train = sequence.pad_sequences(x_train, maxlen = 30)






EMBEDDING_FILE = glove_para
embeddings_index = {}
for i, line in enumerate(open(EMBEDDING_FILE,encoding="utf-8")):
    val = line.split()
    embeddings_index[val[0]] = np.asarray(val[1:], dtype='float32')
embedding_matrix = np.zeros((len(word_index) + 1, vec_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print("vocabulary num",len(word_index))
print("embed matrix",embedding_matrix.shape)

# from sklearn.decomposition import FactorAnalysis
# embedding_matrix = FactorAnalysis(n_components = 10).fit_transform(embedding_matrix)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(len(word_index), embedding_matrix.shape[1])
        et = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.embedding.weight = nn.Parameter(et)
        self.embedding.weight.requires_grad = False
        # self.embedding_dropout = nn.Dropout2d(0.1)
        self.lstm = nn.LSTM(embedding_matrix.shape[1], 40)
        self.linear = nn.Linear(40, 16)
        self.out = nn.Linear(16, 2)
        self.relu = nn.ReLU()


    def forward(self, x):
        h_embedding = self.embedding(x)
        h_lstm, _ = self.lstm(h_embedding)
        max_pool, _ = torch.max(h_lstm, 1)
        linear = self.relu(self.linear(max_pool))
        out = self.out(linear)

        return out



xtrain,xtest,ytrain,ytest=train_test_split(x_train,y_train,test_size=0.1)

x_tr = torch.tensor(xtrain,dtype=torch.long)
y_tr = torch.tensor(ytrain)
train = TensorDataset(x_tr, y_tr)
trainloader = DataLoader(train, batch_size=400)
x_val = torch.tensor(xtest,dtype=torch.long)
y_val = torch.tensor(ytest)
valid = TensorDataset(x_val, y_val)
validloader = DataLoader(valid, batch_size=800)

loss_function =nn.CrossEntropyLoss()  #nn.BCEWithLogitsLoss(reduction='mean')
if mode=="write":
    model=Model()


    optimizer = torch.optim.Adam(model.parameters())

    #######训练模型
    for epoch in range(15):
        train_loss, valid_loss = [], []
        ## training part
        model.train()
        for data, target in trainloader:

            output = model(data)
            loss = loss_function(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
    torch.save(model,para_path)
else:model=torch.load(para_path)

model.eval()
valid_loss=[]
for data, target in validloader:
    output = model(data)
    loss = loss_function(output, target)
    valid_loss.append(loss.item())


##########输出结果
dataiter = iter(validloader)
data, labels = dataiter.next()

output = model(data)
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy())
print("prediction",preds)
print("label",labels)
print(accuracy_score(preds,labels))


TP=0;FP=0;TN=0;FN=0

for i in range(len(preds)):

    if preds[i]==1:
        if labels[i]==1:TP+=1;
        else: FP+=1
    else:
        if labels[i]==0:TN+=1
        else: FN+=1

all=TP+FP+TN+FN
print("TP:",TP/all,"FP:",FP/all,"TN:",TN/all,"FN:",FN/all)