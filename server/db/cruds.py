import uuid
from typing import Dict, List, Optional
from matplotlib.pyplot import title
import pandas as pd
from sqlalchemy.orm import Session
from db import models, schemas
from db.database import Base
import pdb
import re
import psycopg2.extras
import csv
import shutil
import os
import json

#def select_crossvalid_all(db: Session,number: int ) -> List[schemas.CrossVaild]:
#    return db.query(models.CrossVaild).limit(number).all()
def select_patent_all_light(db: Session,number: int ) -> List[schemas.Patent]:
    return db.query(models.Patent.patent_id,
                    models.Patent.syutugan_number,
                    models.Patent.syutugan_date,
                    models.Patent.koukai_number,
                    models.Patent.koukai_date,
                    models.Patent.title,
                    models.Patent.ai_train_label,
                    models.Patent.ai_predict_score,
                    models.Patent.ai_predict_label,
                    models.Patent.url).limit(number).all()                

def select_patent_all(db: Session,number: int ) -> List[schemas.Patent]:
    return db.query(models.Patent).limit(number).all()

def select_patent_by_id(
    db: Session,
    patent_id: str,
) -> schemas.Patent:
    return db.query(models.Patent).filter(models.Patent.patent_id == patent_id).first()

def add_patent(
    db: Session,
    model: schemas.Patent,
    commit: bool = True,
) -> schemas.Patent:
    exists = select_patent_by_id(
        db=db,
        patent_id=model.patent_id,
    )
    if exists:
        return exists
    else:
        patent_id = str(uuid.uuid4())[:6]
        data = models.Patent(
            patent_id=patent_id,
            syutugan_number = model.syutugan_number,
            syutugan_date = model.syutugan_date,
            koukai_number = model.koukai_number,
            koukai_date = model.koukai_date,
            title = model.title,
            ai_train_label= model.ai_train_label,
            ai_predict_score= model.ai_predict_score,
            ai_predict_label = model.ai_predict_label,
            tfidf= model.tfidf
        )
        db.add(data)
        if commit:
            db.commit()
            db.refresh(data)
        return data

def delete_patent(
    db: Session,
    patent_id: str,
) -> schemas.Patent:
    
    data = db.query(models.Patent).filter(models.Patent.patent_id == patent_id).first()

    if not data:
        return 0
    db.delete(data)
    db.commit()
    return 1

def delete_all_patent(
    db: Session,
):
    
    db.query(models.Patent).delete()
    db.commit()
    return 1


Patent_fields_columns = [
    '出願番号', '出願日', '公開番号', '公開日','名称',
    'AI教師ラベル',
    'AI予測スコア', 'AI予測ラベル', '特徴キーワード(タイトル/要約/請求の範囲/明細書/審査官キーワード)',"URL"
    ]
Database_columns = ["syutugan_number",
 "syutugan_date", "koukai_number", "koukai_date", "title",
  "ai_train_label", "ai_predict_score", "ai_predict_label", "tfidf","url"]

Clossvalid_fields_columns =  ['n', 'クエリ', 'AI教師ラベル', 'AI予測ラベル', 'AI予測スコア', 'AI予測ラベル候補',
       'AI予測ラベル候補スコア']

Clossvalid_database_columns = ['group','query', 'ai_train_label', 
'ai_predict_label', 'ai_predict_score', 'ai_predict_label_candidate', 'ai_predict_score_candidate']

def convert_to_crossvalid(items):
    labels = set()   # 空のセットを生成(set([]))
    for item in items:
        labels.add(item.ai_train_label)
    labels = list(labels) #listに戻す
    outputs = []
    for item in items:
        output={
            'id': item.patent_id,
            'group':labels.index(item.ai_train_label),
            'querystr':item.koukai_number,
            'ai_train_label':item.ai_train_label,
            'ai_predict_label':item.ai_predict_label,
            'ai_predict_score':item.ai_predict_score,
            'ai_predict_label_candidate': "",
            'ai_predict_score_candidate': "",
            'ai_predict_candidate': ""
        }
        outputs.append(output)
    return json.dumps(outputs)

def crossvalid_insert_to_database(
    db: Session,
    filename=None):

    with open(filename, encoding='utf8', newline='') as f:
        csvreader = csv.reader(f)
        content = [row for row in csvreader]  
    content[0].append('dummy')
    df=pd.DataFrame(content,columns=content[0])     
    df = df.drop(df.index[[0]])
    

    #列にnが出たら、最後まで削除する
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

    
    dict_from_list = dict(zip(Clossvalid_fields_columns,Clossvalid_database_columns))
    
    df = df.rename(columns=dict_from_list)


    for index, row in df.iterrows():
        crossvalid_id = str(uuid.uuid4())[:6]
        data = models.CrossVaild(
            id=crossvalid_id,
            group=row[0],#group
            querystr=row['query'],
            ai_train_label=row['ai_train_label'],
            ai_predict_label=row['ai_predict_label'],
            ai_predict_score=row['ai_predict_score'],
            ai_predict_candidate=str(b[index-1])
        )
        
        
        db.add(data)
        db.commit()
        db.refresh(data)
    
    return 1


def patent_insert_to_database(
    db: Session,
    files=None):

    df_concat =pd.DataFrame()
    
    for file in files:
        shutil.copyfileobj(file.file, open("temp.csv",'wb+'))
        with open("temp.csv", encoding='cp932') as f:
            csvreader = csv.reader(f)
            content = [row for row in csvreader]
    
        df=pd.DataFrame(content,columns=content[0])
        dict_from_list = dict(zip(Patent_fields_columns,Database_columns))
        df = df.loc[:, Patent_fields_columns]
        df = df.drop(df.index[[0]])

        dict_from_list = dict(zip(Patent_fields_columns,Database_columns))
        df = df.rename(columns=dict_from_list)
        df_concat = df_concat.append(df)
        
    
    print(df_concat.shape)
    
    for index, row in df_concat.iterrows():
        patent_id = str(uuid.uuid4())[:6]
        data = models.Patent(
            patent_id=patent_id,
            syutugan_number = row['syutugan_number'],
            syutugan_date = row['syutugan_date'],
            koukai_number = row['koukai_number'],
            koukai_date = row['koukai_date'],
            title = row['title'],
            ai_train_label= row['ai_train_label'],
            ai_predict_score= row['ai_predict_score'],
            ai_predict_label = row['ai_predict_label'],
            tfidf= str(row['tfidf']).strip('{}'),
            url = row['url']
        )

        #if "US2014035735" in data.koukai_number:
        #    pdb.set_trace()
        #    print("stop")
        
        try:
            db.add(data)
            db.commit()
            db.refresh(data)
            print("OK ",index,data.koukai_number)
            
        except:
            #pdb.set_trace()
            print("error ",index,data.koukai_number)
            db.rollback()
    
    return 1


#トレーニングデータのUPLOAD　⇒　書式はPatentFieldと同じ形式とする
def train_csv_upload(
    db: Session,
    csv_file =None):

#csvファイルのアップロードルーチン
#    with open(contents.file, encoding='shift-jis') as f:
#        csvreader = csv.reader(f)
#        content = [row for row in csvreader]  
    
#いったんpandasに入れる
    df = pd.read_csv(csv_file.file, header=None, names=['number', 'label'])

#train データベースに格納する　⇒　データ仕様としては、ユーザーに紐付けした外部キーをつけること
    dict_from_list = dict(zip(Patent_fields_columns,Database_columns))
    df = df.rename(columns=dict_from_list)
 
    for index, row in df.iterrows():
        patent_id = str(uuid.uuid4())[:6]
        data = models.Patent(
            patent_id=patent_id,
            syutugan_number = row['syutugan_number'],
            syutugan_date = row['syutugan_date'],
            koukai_number = row['koukai_number'],
            koukai_date = row['koukai_date'],
            title = row['title'],
            ai_train_label= row['ai_train_label'],
            ai_predict_score= row['ai_predict_score'],
            ai_predict_label = row['ai_predict_label'],
            tfidf= str(row['tfidf']).strip('{}')
        )
        
        #pdb.set_trace()
        db.add(data)
        db.commit()
        db.refresh(data)
    
    return 1

def select_project_by_id(
    db: Session,
    project_id: str,
) -> schemas.Project:
    return db.query(models.Project).filter(models.Project.project_id == project_id).first()


def select_project_by_name(
    db: Session,
    project_name: str,
) -> schemas.Project:
    return db.query(models.Project).filter(models.Project.project_name == project_name).first()


def add_project(
    db: Session,
    project_name: str,
    description: Optional[str] = None,
    commit: bool = True,
) -> schemas.Project:
    exists = select_project_by_name(
        db=db,
        project_name=project_name,
    )
    if exists:
        return exists
    else:
        project_id = str(uuid.uuid4())[:6]
        data = models.Project(
            project_id=project_id,
            project_name=project_name,
            description=description,
        )
        db.add(data)
        if commit:
            db.commit()
            db.refresh(data)
        return data




def search_patent(db:Session,mode:str,patent_number:str)->Optional[models.Patent]:
    if mode == "出願番号":
        return  select_patent_by_syutugan_number(db=db,number=patent_number)
    elif mode == "公開番号":
        return  select_patent_by_koukai_number(db=db,number=patent_number)
    else:
        return False        

def search_patents(db:Session,mode:str,patent_numbers:List[str])->List[models.Patent]:
    if mode == "出願番号":
        return  select_patents_by_syutugan_numbers(db=db,numbers=patent_numbers)
    elif mode == "公開番号":
        return  select_patents_by_koukai_numbers(db=db,numbers=patent_numbers)
    else:
        return False

def select_patent_by_syutugan_number(db:Session,number:str):
    return db.query(models.Patent).filter(models.Patent.syutugan_number == number).first()

def select_patent_by_koukai_number(db:Session,number:str):
    return db.query(models.Patent).filter(models.Patent.koukai_number == number).first()

def select_patents_by_syutugan_numbers(db:Session,numbers:list):
    return db.query(models.Patent).filter(models.Patent.syutugan_number.in_(numbers)).all()

def select_patents_by_koukai_numbers(db:Session,numbers:list):
    return db.query(models.Patent).filter(models.Patent.koukai_number.in_(numbers)).all()


def model_table_all_delete(db:Session,model:Base):
    try:
        db.query(model).delete()
        
        db.commit()
    except Exception as err:
        db.rollback()
        
#see the sqlalchemy document
def df_write_to_sql(df,db,model):

    name = model.__tablename__  #see the sqlalchemy document
    df.to_sql(name=name,con=db.bind,if_exists='append',index = False)#2020/2/23 アペンドやめる　if_exists='append')
    #indexがFalseの場合はインデックスをインサートするカラムに含めない (デフォルトはTrue)
    #index_label='id' を追加した
    return 

def model_data_import(db:Session,filename:str,model:Base):
    try:
        df = model.data_import(filename)  #write to each model class, return value is pandas.dataframe 
        df_write_to_sql(df,db,model)
        return True
    except:
        return False

def model_data_view(db:Session,model):
    
    return db.query(model.__table__).all()

def exec_query(db:Session,query:str):
    
    return db.execute(query)




