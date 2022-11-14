from typing import Tuple
import numpy as np
from sklearn import metrics
from time import time
import pdb
from typing import Tuple
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR, StepLR


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."

#print_top10(unique_word, clf, label_names)
def print_top10(topN,feature_names, clf, class_labels):
    
    restr = {}
    
    for i, class_label in enumerate(class_labels):
        top10_vec,top10 = torch.topk(clf.l1.weight[i],topN)
        
        top10=top10.tolist()
        top10_vec=top10_vec.tolist()
            
        dic = {key: val for key, val in zip(feature_names[top10],top10_vec)}
        restr[class_label] = dic
            
    return restr



def benchmark(model,x_train, x_test, y_train, y_test,numofword):

#    if torch.cuda.is_available():
#        device = 'cuda:0'
#    else:
#        device = 'cpu'
#        
#    model.to(device)
    
    outstr = {}
    t0 = time()
    
    X_train = torch.from_numpy(x_train.astype(np.float32))
    X_test = torch.from_numpy(x_test.astype(np.float32))
    Y_train = torch.from_numpy(y_train).long()
    Y_test = torch.from_numpy(y_test).long()

    train = TensorDataset(X_train,Y_train)
    train_loader = DataLoader(train,shuffle=False,batch_size=int(len(train)))
    
    test = TensorDataset(X_test,Y_test)
    test_loader = DataLoader(test,shuffle=False,batch_size=len(test))

    cls_sample_count = np.unique(Y_train,return_counts=True)[1]
    weight = torch.from_numpy((1. / cls_sample_count).astype(np.float32)) 
    
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    
    optimizer = torch.optim.SGD(model.parameters(),lr= model.lr_rate)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    loss_history = []

    iterations =0
    
    
    for epoch in tqdm(range(model.epoch)):
    
        total_loss = 0
        for x, y in train_loader:

        # 学習ステップ
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()        
            total_loss += loss.item()
            
        scheduler.step()
        loss_history.append(total_loss)
        if (epoch +1) % 100 == 0:
            print(epoch + 1, total_loss)
    
    train_time = time() - t0
    outstr["train time"] = "%0.3f" % train_time 
    
    t0 = time()
              
    for x_test,y_test in test_loader:
        outputs = model(x_test)
        
        pred_proba_vec,pred_proba= torch.topk(outputs.data,dim=1,k= numofword if outputs.shape[1] > numofword else outputs.shape[1])
        _,pred= torch.max(outputs.data,dim=1)
        
    test_time = time() - t0
    outstr["test time"] = "%0.3f" % test_time 
#正解率
    score = metrics.accuracy_score(y_test, pred)
    outstr["accuracy"] = "%0.3f" % score 

#再現性
    recall = metrics.recall_score(y_test, pred, average='macro')
    outstr["recall"] = "%0.3f" % recall

    return model,outstr,pred_proba_vec,pred_proba

class LogisticRegression(torch.nn.Module):
    def __init__(self,input_dim,output_dim,lr_rate=0.01,epoch=20):
        super(LogisticRegression,self).__init__()
        self.l1 = torch.nn.Linear(input_dim,output_dim)
        self.__lr_rate = lr_rate
        self.__epoch = epoch
        
    def forward(self,x):
        x=self.l1(x)
        return x
    
    @property
    def lr_rate(self):
        return self.__lr_rate
    @lr_rate.setter
    def lr_rate(self,value):
        if value is None:
            raise TypeError('invalid value')
        self.__lr_rate = value

    @property
    def epoch(self):
        return self.__epoch
    @epoch.setter
    def epoch(self,value):
        if value is None:
            raise TypeError('invalid value')
        self.__epoch = value