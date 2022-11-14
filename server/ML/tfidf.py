import pandas as pd
import tqdm as tqdm
import numpy as np
import pdb
from typing import Tuple
from sklearn.preprocessing import LabelEncoder

from time import time
from typing import Tuple
from tqdm import tqdm

from janome.tokenizer import Tokenizer
from collections import defaultdict
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer
from janome.tokenfilter import POSKeepFilter


def tfidf(df,head) -> Tuple[np.ndarray, np.ndarray, np.ndarray,np.ndarray, np.ndarray]:
    
    t0 = time()

    patentid_list=[] #特許のID配列
    tfidf_word=[] #TFIDFの語句配列
    tfidf_score=[] #TFIDFのスコア配列
    train_label=[]

    for id,tfidf in zip(df.patent_id,df.tfidf):
        df1 = pd.DataFrame(list(tfidf.items()),columns= ['word','tfidf'])
        df1['tfidf']= df1['tfidf'].astype(float)
       
        #df1 = df1[df1['tfidf']> head ]
        df1 = df1.sort_values(by='tfidf',ascending=False).head(head)
        patentid_list.append(id)
        tfidf_word.append(df1['word'].tolist())
        tfidf_score.append(df1['tfidf'].tolist())
    
    train_label = df['ai_train_label'].tolist()

    unique_word = set([x for row in tfidf_word for x in row])
    unique_idx = {w:i for i,w in enumerate(unique_word)}

    print('number of unique words = {}'.format(len(unique_word)))
    print('number of data lengths = {}'.format(len(tfidf_word)))

    to_unique_idx = np.vectorize(unique_idx.get)
    tfidf_arr = np.zeros((len(tfidf_word),len(unique_word)))

    for i,(words,scores) in enumerate(zip(tfidf_word,tfidf_score)):
        words = np.array(words)
        scores = np.array(scores,dtype=float)
    
        try:
            idx = to_unique_idx(words)
            tfidf_arr[i,idx] = scores
        except:
            pass

    
    # 変換前のデータ
    train_arr = np.array(train_label)
    le = LabelEncoder()
    le.fit(train_arr)
    label_names = le.classes_
    label_array = le.transform(train_arr)
    

    waist_time = time() - t0
    print("tfidf time: %0.3fs" % waist_time)

    unique_words = np.array(list(unique_word))

    return tfidf_arr, train_arr, label_names , unique_words,label_array


def get_token(text):
    t = Tokenizer()
    tokens = t.tokenize(text)
    
    word = ""
    for token in tokens:
        part_of_speech = token.part_of_speech.split(",")[0]
        if part_of_speech == "名詞":
            word +=token.surface + " "
        if part_of_speech == "動詞":
            word +=token.base_form+ " "
        if part_of_speech == "形容詞":
            word +=token.base_form+ " "
        if part_of_speech == "形容動詞":
            word +=token.base_form+ " "
    return word


def tfidf_calc(target,token,max_word):
    corpus=[]
    
    for item in tqdm(target): 
        token=get_token(item)
        corpus.append(token)

    vectorize = TfidfVectorizer(use_idf=True,max_df=0.9)
    
    tfidf = vectorize.fit_transform(corpus)
    
    output =[]
    words = vectorize.get_feature_names()
    for doc_id,vec in zip(range(len(corpus)),tfidf.toarray()):
        jstring=""
        for w_id,tfidf in sorted(enumerate(vec),key=lambda x:x[1],reverse=True)[:max_word]:
            lemma= words[w_id]
            jstring += '"{}"=>"{:.7g}",'.format(lemma,tfidf*100)
        output.append(jstring)
    
    return output

