# ------------------------------------------------------------------
#	database.py
#
#					May/02/2022
# ------------------------------------------------------------------
from mongo_db.model import CV
import shutil
import sys
import	datetime
#
import motor.motor_asyncio
from pymongo import MongoClient
import pymongo
import pdb
import uuid
from typing import Dict, List, Optional
from matplotlib.pyplot import title
import pandas as pd
from sqlalchemy.orm import Session
from db import models, schemas
import pdb
import re
import psycopg2.extras
import csv
import shutil
import os
import json
from tempfile import NamedTemporaryFile
from pathlib import Path
from fastapi.responses import RedirectResponse
from ML.tfidffreetext import tfidf_calc
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer
from vutil.dialogs import get_wordcloud,file_upload
from db import cruds

from bson.json_util import dumps
from bson.json_util import loads

# ------------------------------------------------------------------
client = motor.motor_asyncio.AsyncIOMotorClient('mongodb://root:root@172.26.106.79:27017')
database = client.datafolder
collection = database.localstrage
database_freetext = client.tfidf

# ------------------------------------------------------------------
async def fetch_one_crossValid(id):
	document = await collection.find_one({"id":id})
	return document
#
# ------------------------------------------------------------------
async def fetch_all_crossvalid():
	CrossValids = []
	cursor = collection.find({})
	async for document in cursor:
		#pdb.set_trace()
		CrossValids.append(CV(**document))

	return CrossValids
#
# ------------------------------------------------------------------
async def create_crossValid(CrossValid):
	document = CrossValid
	result = await collection.insert_one(document)
	return document

# ------------------------------------------------------------------
async def update_crossValid(id, population):
	sys.stderr.write("*** update_CrossValid ***\n")
	date_mod = datetime.date.today()
	await collection.update_one({"id":id}, {"$set": {
		'population': population,
		'date_mod': '%s' % date_mod
		}})
	document = await collection.find_one({"id": id})
	return document

# ------------------------------------------------------------------
async def remove_crossValid(id):
	await collection.delete_one({"id":id})
	return True
# ------------------------------------------------------------------


async def freetext_insert_to_database(filename=None):

    #file拡張子の取得
    fn = filename.filename
    suffix = Path(fn).suffix
    stem = Path(fn).stem
    
    with NamedTemporaryFile(delete=False,suffix=suffix,prefix = stem +"_") as f:
        shutil.copyfileobj(filename.file, f)
        tmp_path = Path(f.name)

    with open(tmp_path, encoding='utf-8_sig', newline='') as f: 
        csvreader = csv.reader(f)
        content = [row for row in csvreader]
    
    df=pd.DataFrame(content,columns=content[0])     
    df = df.drop(df.index[[0]])

    #freetextと言っても、PatentField仕様に準ずること
    df.columns=['text','label','t']
    jstring = df.to_dict(orient ='records')

    #collection名は一時ファイル名にする
    coll = Path(f.name).stem
    coll_dest = database_freetext[coll]
    for i in jstring:
        try:   
            coll_dest.insert_one(i)
        except:
            pass
    
    return coll

#tfidfの計算
#対象のコーパスを指定する
async def add_tfidf_text(token_str=None):
    corpus =[]
    label= []
    
    #cursor = database_freetext[token].find({})
    pdb.set_trace()
    
    async for document in cursor:
        corpus.append(document['text'])
        label.append(document['label'])
    
    #corpus = corpus[0:100]
    #label = label[0:100]

    tfidf = tfidf_calc(corpus,token,100)
       
    df = pd.DataFrame({'text':corpus,'label':label , 'tfidf':tfidf,})
    
    return df



#tfidfの計算
async def add_tfidf(token=None):
    corpus =[]
    label= []
    cursor = database_freetext[token].find({})
    pdb.set_trace()
    
    async for document in cursor:
        corpus.append(document['text'])
        label.append(document['label'])
    
    #corpus = corpus[0:100]
    #label = label[0:100]

    tfidf = tfidf_calc(corpus,token,100)
       
    df = pd.DataFrame({'text':corpus,'label':label , 'tfidf':tfidf,})
    
    return df

#PatentFiledから排出したcrossvalid.csvファイルを読み込み、データベースに格納する
#tokenは一時ファイル名でURLから取得する
#filenameはpostされたcrossvalid.csvファイル
#引数は単体ファイルであり、multi_file_uploadはrouter側で処理する

#Column fieldの対応付けをリスト化    
Clossvalid_fields_columns =  ['n', 'クエリ', 'AI教師ラベル', 'AI予測ラベル', 'AI予測スコア', 'AI予測ラベル候補',
       'AI予測ラベル候補スコア']

Clossvalid_database_columns = ['group','query', 'ai_train_label', 
'ai_predict_label', 'ai_predict_score', 'ai_predict_label_candidate', 'ai_predict_score_candidate']

async def upload_pf_crossvalid_data_format(db:Session,token=None,file_obj=None):
    
    #filenameはpostされたcrossvalid.csvファイル
    shutil.copyfileobj(file_obj.file, open("temp.csv",'wb+'))
    with open("temp.csv", encoding='utf8', newline='') as f:
        csvreader = csv.reader(f)
        content = [row for row in csvreader]

    #読み込みエラー対策として、ダミー列を追加している
    content[0].extend(['dummy','dummy','dummy'])

    df= pd.DataFrame(content,columns=content[0])     
    df = df.drop(df.index[[0]])

    #ゴミデータの除去　→　列にnが出たら、最後まで削除する
    try:    
        row = df.index[df[df.columns[0]]=='n'].tolist()  
        df = df.drop(range(row[0]-1,len(df.index)+1))
        df = df.dropna(subset=['n'])
    except:
        pass
     
    scorelist = [ re.findall(r'[^|]+',x) for x in df["AI予測ラベル候補スコア"]]
    labellist = [ re.findall(r'[^|]+',x) for x in df["AI予測ラベル候補"]]
    multilist = [dict(zip(itemj,scorelist[j])) for j,itemj in enumerate(labellist)]

    b=[]
    for i in multilist:
        a=""
        for j in i.keys():
            a = a + j+"=>"+i[j]+","
        b.append(str(a[:-1]))

    #csvのカラムをデータベースのカラムにrename
    dict_from_list = dict(zip(Clossvalid_fields_columns,Clossvalid_database_columns))
    df = df.rename(columns=dict_from_list,errors='ignore')

    for index, row in df.iterrows():
        crossvalid_id = str(uuid.uuid4())[:6]
        data = CV(
            id=crossvalid_id,
            group=row[0],#group
            querystr=row['query'],
            ai_train_label=row['ai_train_label'],
            ai_predict_label=row['ai_predict_label'],
            ai_predict_score=row['ai_predict_score'],
            ai_predict_candidate=str(b[index-1])
        )
        
        mode = "公開番号"
        
        #qurystrでpatentデータベースが検索される前提
        item_patent = cruds.search_patent(
            db=db, mode=mode, patent_number=row['query'])
        if item_patent == None:
            print("Not found {}".format(row['query']))
        
        else:
            await database[token].insert_one(data.dict())
                
    return 1