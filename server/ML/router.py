from copyreg import pickle
import io
import typing
from ast import Bytes
from typing import List
import base64
import uuid
import json


from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,LeaveOneOut, cross_val_score, KFold, StratifiedKFold
from sklearn.linear_model import Perceptron
from time import time
from tqdm import tqdm
from ML.sklearn import benchmark as benchmark_sklearn, print_top10 as print_top10_sklearn
from ML.pytorch import benchmark as benchmark_pytorch, print_top10 as print_top10_torch ,LogisticRegression as LogisticRegression_pytorch
from urllib3 import HTTPResponse
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from sklearn.ensemble import RandomForestClassifier
from db.database import get_db
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from flask import jsonify
from sklearn.feature_extraction import img_to_graph
from sqlalchemy.orm import Session
import requests
from fastapi.responses import RedirectResponse
from .tfidf import tfidf
import numpy as np
import pandas as pd
from db import cruds, schemas
from sklearn.utils.extmath import density
import pdb
from io import BytesIO

from fastapi import Body

static_path = "./static/"


router = APIRouter(tags=["ML"])

# TFIDFマトリクスと、ラベルベクトルを作るエンドポイント
# データベースへの問い合わせは、一括に問い合わせる。
# 戻り値はdataframe、Patent、Labelのセットを内部でPickel化、IDはトークンとする
# 細かいcrud処理は考えないこと
# Patent集合をJSONで受け取るエンドポイント

@router.post('/patent_list', response_model=List[schemas.Patent])
async def post_patent_list(db: Session = Depends(get_db),
                           mode: str = Query("公開番号", enum=["出願番号", "公開番号"]),
                           body=Body(...)):

    if body:
        data = body['patent']
        tfidf_max_length= body['tfidf_max_length']
        df = pd.read_json(data)##CrossValidationで使うためにデータフレームにする
        patent_list = df['querystr'].tolist()##クエリストリングをリストにする

        # 処理が持たない気がする　→　まずはシングルで作る・・・
        item_patent = cruds.search_patents(
            db=db, mode=mode, patent_numbers=patent_list)
        # modelスケマからjson→pandasに変換する
        json_item_data = jsonable_encoder(item_patent)
        df_s = pd.DataFrame(json_item_data)
        
        for index,row in df_s.iterrows():
            label = df[df.querystr == row.koukai_number].ai_train_label
            df_s.at[index,'ai_train_label'] = label.values[0]
            
        # pickleにつける
        df_s.to_pickle(static_path + body['token'])

       # リダイレクト
        return RedirectResponse("/train_tfidf/?token={}&tfidf_max_length={}".format(body['token'], tfidf_max_length), status_code=303)

    return {"Error": "Upload file not found"}

@router.post('/patent_list2', response_model=List[schemas.Patent])
def post_patent_list2(db: Session = Depends(get_db),
                           mode: str = Query("公開番号", enum=["出願番号", "公開番号"]),
                           body=Body(...)):

    if body:
        data = body['patent']
        tfidf_max_length= body['tfidf_max_length']
        df = pd.read_json(data)##CrossValidationで使うためにデータフレームにする
        patent_list = df['querystr'].tolist()##クエリストリングをリストにする

        # 処理が持たない気がする　→　まずはシングルで作る・・・
        items = cruds.search_patents(
            db=db, mode=mode, patent_numbers=patent_list)
        return items
        
    return {"Error": "Upload file not found"}

# Patent集合をJSONで受け取るエンドポイント
# tokenを指定して、そのtokenのデータを取得する
@router.get('/train_tfidf/')
def post_train_tfidf(
        db: Session = Depends(get_db),
        token: str = "",
        tfidf_max_length: int = 10000):
    
    unpickled_df = pd.read_pickle(static_path + token)
    tfidf_arr, label_arr,label_names ,unique_word,label_array= tfidf(unpickled_df, tfidf_max_length)
    # pickleにつける

    np.save(static_path +'tfidf_arr.npy', tfidf_arr)
    np.save(static_path +'label_arr.npy', label_arr)
    np.save(static_path +'label_names.npy', label_names)
    np.save(static_path + 'unique_names.npy', unique_word)
    np.save(static_path + 'label_array.npy', label_array)
    
    # リダイレクト
    return {"token": token, 
            "tfidf_arr": tfidf_arr.shape, 
            "label_arr": label_arr.shape, 
            "label_names": label_names.shape, 
            "unique_word": unique_word.shape,
            }



@router.get('/calc_crossvalid/')
def get_mlflow(db: Session = Depends(get_db),
            token: str = "",numofword: int = 100,
            cvmode: str = Query("KFold", enum=["KFold", "LeaveOneOut"]),
            nsplit: int = Query(5, ge=2, le=100),
            n_candidate: int = Query(5, ge=1, le=100),
            ):

    
    tfidf_arr=np.load(static_path +'tfidf_arr.npy')
    label_arr=np.load(static_path +'label_arr.npy')
    label_names=np.load(static_path +'label_names.npy')
    unique_word = np.load(static_path +'unique_names.npy')

    if cvmode == "KFold":# kfoldでクロスバリデーションを行う
        CrossVaild = StratifiedKFold(n_splits=nsplit, shuffle=True)#,random_state=42)
    elif cvmode == "LeaveOneOut":# 1 vs 1 でクロスバリデーションを行う
        CrossVaild = LeaveOneOut()

    #predict-proba が使えるモデルを使う
    clf =  LogisticRegression(class_weight='balanced')

    ext = []
    mlresult =[]
    mlresult2=[]

    for train_index, test_index in tqdm(CrossVaild.split(tfidf_arr, label_arr)):
        x_train, x_test = tfidf_arr[train_index], tfidf_arr[test_index]
        y_train, y_test = label_arr[train_index], label_arr[test_index]

        ms = StandardScaler()
        #ms =  StandardScaler()
        X_train = ms.fit_transform(x_train)
        X_test = ms.transform(x_test)

        clf,ostr = benchmark_sklearn(clf,X_train, X_test, y_train, y_test)

#def print_top10(feature_names, clf, class_labels):
        if hasattr(clf, "coef_"):
            #ostr["dimensionality"] = "%d" % clf.coef_.shape[1] 
            #ostr["density"] = "%f" % density(clf.coef_) 
            ostr["top10"] = print_top10_sklearn(numofword,unique_word, clf, label_names)
            intercept = clf.intercept_.tolist()
            mlresult.append(ostr)
            mlresult2.append( dict(zip(label_names,intercept))) #mlresult.append(ostr)
        
        pred_proba_train = clf.predict_proba(X_train)
        pred_proba_test = clf.predict_proba(X_test)

        df = pd.read_pickle(static_path + token)
 
        labels = set()   # 空のセットを生成(set([]))
        for i,value in enumerate(test_index):
            labels.add( df.loc[value].ai_train_label)
        
        labels = list(labels) #listに戻す

        #pdb.set_trace()

        for ind, value in enumerate(test_index):
            output = {"id": str(uuid.uuid4())[:6]}
            output.update({"group": len(mlresult)-1})  #labels.index( df.loc[value].ai_train_label)})
            num = df.loc[value].koukai_number
            output.update({"querystr": num})
            output.update({"ai_train_label": df.loc[value].ai_train_label})
            output.update(
            {"ai_predict_label(pf)": df.loc[value].ai_predict_label})
            output.update(
            {"ai_predict_score(pf)": df.loc[value].ai_predict_score})
            
            ds = {key: val for key, val in zip(label_names, map(
            '{:f}'.format, pred_proba_test[ind].tolist()))}
            #pdb.set_trace()
            max_kv = max(ds.items(), key=lambda x: x[1])
            output.update({"ai_predict_label": max_kv[0]})
            output.update({"ai_predict_score": max_kv[1]})
            ds_sel = dict( sorted(ds.items(),key=lambda x:x[1],reverse=True)[0:n_candidate])
            output.update({"ai_predict_candidate": json.dumps(ds_sel)})
            
            #pdb.set_trace()
            
            ext.append(output)

    return JSONResponse(content={"crossvalid":ext,"mlresult":mlresult,"mlresult2":mlresult2})


@router.get('/calc_crossvalid2/')
def get_mlflow_pytorch(db: Session = Depends(get_db),
            token: str = "",
            numofword: int = 100,
            cvmode: str = Query("KFold", enum=["KFold", "LeaveOneOut"]),
            nsplit: int = Query(5, ge=2, le=100),
            lr_rate: float = 0.01,
            epoch: int = 20, 
            n_candidate: int = Query(5, ge=1, le=100),
            ):
    
    #pdb.set_trace()
    
    tfidf_arr=np.load(static_path +'tfidf_arr.npy')
    label_array=np.load(static_path +'label_array.npy')#encoding されたもの
    label_names=np.load(static_path +'label_names.npy')
    unique_word = np.load(static_path +'unique_names.npy')

    if cvmode == "KFold":# kfoldでクロスバリデーションを行う
        CrossVaild = StratifiedKFold(n_splits=nsplit, shuffle=True)#,random_state=42)
    elif cvmode == "LeaveOneOut":# 1 vs 1 でクロスバリデーションを行う
        CrossVaild = LeaveOneOut()

    #predict-proba が使えるモデルを使う
    input_dim = tfidf_arr.shape[1]
    output_dim = len(label_names)
    
    ext = []
    mlresult =[]
    mlresult2=[]

    for train_index, test_index in tqdm(CrossVaild.split(tfidf_arr, label_array)):
        
        #model selection
        clf = LogisticRegression_pytorch(
            input_dim= input_dim,
            output_dim = output_dim,
            lr_rate=lr_rate,
            epoch=epoch)
        
        
        x_train, x_test = tfidf_arr[train_index], tfidf_arr[test_index]
        y_train, y_test = label_array[train_index], label_array[test_index]

        ms = StandardScaler()
        X_train = ms.fit_transform(x_train)
        X_test = ms.transform(x_test)

        clf,ostr,pred_proba_vec,pred_proba = benchmark_pytorch(clf,X_train, X_test, y_train, y_test,numofword)    
        
        ostr["top10"] = print_top10_torch(numofword,unique_word, clf, label_names)
        intercept = clf.l1.bias.tolist()
        
        mlresult.append(ostr)
        mlresult2.append( dict(zip(label_names,intercept))) #mlresult.append(ostr)
              
        df = pd.read_pickle(static_path + token)
 
        for ind, value in enumerate(test_index):
            output = {"id": str(uuid.uuid4())[:6]}
            output.update({"group": len(mlresult)-1}) 
            num = df.loc[value].koukai_number
            output.update({"querystr": num})
            output.update({"ai_train_label": df.loc[value].ai_train_label})
            output.update(
            {"ai_predict_label(pf)": df.loc[value].ai_predict_label})
            output.update(
            {"ai_predict_score(pf)": df.loc[value].ai_predict_score})

            ds = {key: val for key, val in zip(label_names[pred_proba[ind].tolist()], pred_proba_vec[ind].tolist())}
                  
            max_kv = max(ds.items(), key=lambda x: x[1])
            output.update({"ai_predict_label": max_kv[0]})
            output.update({"ai_predict_score": max_kv[1]})
            ds_sel = dict( sorted(ds.items(),key=lambda x:x[1],reverse=True)[0:n_candidate])
            output.update({"ai_predict_candidate": json.dumps(ds_sel)})

            ext.append(output)

    return JSONResponse(content={"crossvalid":ext,"mlresult":mlresult,"mlresult2":mlresult2})
