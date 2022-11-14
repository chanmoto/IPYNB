from typing import Tuple
import numpy as np
from sklearn import metrics
from time import time
import pdb
from typing import Tuple
from sklearn.metrics import confusion_matrix

def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."

#print_top10(unique_word, clf, label_names)
def print_top10(topN,feature_names, clf, class_labels):
    
    
    restr = {}

    if len(class_labels)>2:
        N= -1 * topN
        for i, class_label in enumerate(class_labels):
            
            top10 = np.argsort(clf.coef_[i])[N:]
            top10 = top10[::-1]
            top10_vec = np.sort(clf.coef_[i])[N:]
            top10_vec = top10_vec[::-1]
            dic = {key: val for key, val in zip(feature_names[top10],top10_vec)}
            restr[class_label] = dic
            
            #print("{} {} {}".format( class_label,len(top10),len(top10_vec)))

        return restr
    else:
        N= -1 * topN
        
        top10 = np.argsort(clf.coef_[0])[N:]
        top10 = top10[::-1]
        top10_vec = np.sort(clf.coef_[0])[N:]
        top10_vec = top10_vec[::-1]
        dic = {key: val for key, val in zip(feature_names[top10],top10_vec)}
        restr[class_labels[1]] = dic
        #print("{} {} {}".format( class_labels[0],len(top10),len(top10_vec)))

        top10 = np.argsort(clf.coef_[0])[:topN]
        top10_vec = np.sort(clf.coef_[0])[:topN]
        dic = {key: val * -1 for key, val in zip(feature_names[top10],top10_vec)}
        restr[class_labels[0]] = dic
        #print("{} {} {}".format( class_labels[1],len(top10),len(top10_vec)))

        return restr

def benchmark(clf,X_train, X_test, y_train, y_test):
    outstr = {}

    t0 = time()
    clf.fit(X_train, y_train)
    
    train_time = time() - t0
    outstr["train time"] = "%0.3f" % train_time 

    t0 = time()
    pred = clf.predict(X_test)

    #print(confusion_matrix(y_test, pred))

    test_time = time() - t0
    outstr["test time"] = "%0.3f" % test_time 
    
#正解率
    score = metrics.accuracy_score(y_test, pred)
    outstr["accuracy"] = "%0.3f" % score 

#再現性
    recall = metrics.recall_score(y_test, pred, average='macro')
    outstr["recall"] = "%0.3f" % recall

    return clf,outstr
