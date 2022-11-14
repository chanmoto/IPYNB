from sqlalchemy import Column, DateTime, ForeignKey, String, Text,Integer
from sqlalchemy.sql.functions import current_timestamp
from sqlalchemy.types import JSON
from db.database import Base
from sqlalchemy.dialects.postgresql import HSTORE
import pdb
import pandas as    pd
import numpy as np
import os

"""    syutugan_number: str
    syutugan_date:datetime.date
    koukai_number: str
    koukai_date:datetime.date
    """


class Patent(Base):
    __tablename__ = 'patent'

    patent_id = Column(String(255), primary_key=True)
    syutugan_number = Column(String(255), unique=True)
    syutugan_date = Column(String(255))
    koukai_number = Column(String(255))
    koukai_date = Column(String(255))
    title = Column(String(400))
    ai_train_label = Column(String(255))
    ai_predict_score = Column(String(255))
    ai_predict_label = Column(String(255))
    tfidf = Column(HSTORE)
    url = Column(String(255))

# 総括的なプロジェクト
class Project(Base):
    __tablename__ = "projects"
    project_id = Column(String(255), primary_key=True, comment="主キー",)
    project_name = Column(String(255), nullable=False,
                          unique=True, comment="プロジェクト名",)
    description = Column(Text, nullable=True, comment="説明",)
    created_datetime = Column(
        DateTime(timezone=True), server_default=current_timestamp(), nullable=False,)

# トレインデータ
class TrainData(Base):
    __tablename__ = "trains"
    model_id = Column(String(255), primary_key=True,  comment="主キー",)
    project_id = Column(String(255), ForeignKey(
        "projects.project_id"), nullable=False, comment="外部キー",)
    patent_number = Column(String(255), comment="特許番号")
    label_name = Column(String(255), nullable=False, comment="ラベル名",)


class ClassifyMasterHistory(Base):
    __tablename__ = "classify_master_history"
    
    #db.Column情報の設定
    id = Column(Integer, primary_key=True,autoincrement=True)
    H10 = Column(String)      
    H20 = Column(String)                                
    H30 = Column(String)        
    H40 = Column(String)        
    Strategy = Column(String)   
    Comment = Column(String)
    Year = Column(String)
    
    #ファイルからDataframeを返す処理
    @staticmethod
    def data_import(file=None):
        try:
            #EXCEL以外の場合のエラー処理を追加すること
            df = pd.read_excel(file,sheet_name = 0,header=5,usecols=[0,1,2,3,4,5]) #"21F Product Classification    シート名を変更"
            #df = df[df['H40'].str.isalpha()]#ゴミ行削除　英数字以外を除去
            #df = df.astype(str)#ゴミ行削除　NANを省く
            
            df = df.replace(r'^\s*$',np.nan, regex=True).ffill()#空白をフィルインする
            df.columns=['H10', 'H20', 'H30', 'H40', 'Strategy','Comment']
            #末尾に年度を追加する
            df['Year'] = os.path.basename(file).replace(".xlsx","")

            return df
        except:
            return False
    
    def __init__(self,*a):
        super(Foo, self).__init__(**kwargs)
        # do custom stuff

            #特許情報をDBから取得して辞書型で返す
    def get_schema(self):
        schema = [
            self.id,
            self.H10,
            self.H20,
            self.H30,
            self.H40,
            self.Strategy,
            self.Comment,
            self.Year
        ]               
        return schema
    @staticmethod
    def col_name():
        return 'id,H10,H20,H30,H40,戦略分類,詳細,年度'

class PalcomMaster(Base):
    __tablename__ = "palcom_master"
    
    #db.Column情報の設定
    id = Column(Integer, primary_key=True,autoincrement=True)
    murata_number= Column(String) #ムラタ案件管理番号
    family_number= Column(String) #ﾌｧﾐﾘ管理番号
    country_code = Column(String)     #出願番号
    syutugan_number = Column(String)     #出願番号
    koukai_number = Column(String)       #公開/(再)公表番号
    IPC_hittou = Column(String) #IPC（公開/筆頭）
    IPC_zenken= Column(String) #IPC（公開/全件）
    key = Column(String)                  #製品区分
    value = Column(String)                 #製品コード
    classify = Column(String)             #分類筆頭のみ
    strategy = Column(String)             #戦略筆頭のみ

    #ファイルからDataframeを返す処理
    @staticmethod
    def data_import(file=None):
        #CSV以外のファイルはエラー返すこと
        #https://qiita.com/niwaringo/items/d2a30e04e08da8eaa643
        #pandasでread_csv時にUnicodeDecodeErrorが起きた時の対処 (pd.read_table())
        
        df=pd.read_csv(file,usecols =['ムラタ案件管理番号','ﾌｧﾐﾘ管理番号','国コード','庁：IPC（公開/筆頭）','庁：IPC（公開/全件）','製品区分','製品コード','出願番号','公開/(再)公表番号']    ,dtype=str, encoding="cp932")

        df.columns=['murata_number','family_number','country_code','IPC_hittou','IPC_zenken','key', 'value','syutugan_number','koukai_number']
        df_key = df['key'].str.split(',',expand=True).values.tolist()
        df_value = df['value'].str.split(',',expand=True).values.tolist()
        classify_list= []
        strategy_list = []

        for ii,( key, value) in enumerate(zip(df_key,df_value)):
            set_classify_flag = False
            classify_list.append('')
            strategy_list.append('')
            for i,k in enumerate(key):
                if type(k) is str:
                    if 'H30' in k and set_classify_flag == False:
                        classify_list[ii]= value[i]
                        set_classify_flag = True
                    elif '戦略分類' in k :
                        strategy_list[ii]= value[i]
                        break
        df['classify']=classify_list
        df['strategy']=strategy_list
        
        return df
    
    def __init__(self,*a):
        super(Foo, self).__init__(**kwargs)
        # do custom stuff

    #特許情報をDBから取得して辞書型で返す
    def get_schema(self):
        schema = [
            self.id,
            self.murata_number,
            self.family_number,
            self.country_code,
            self.syutugan_number,
            self.koukai_number,
            self.IPC_hittou, 
            self.IPC_zenken,
            self.classify,
            self.strategy,
            self.key,
            self.value
        ]               
        return schema
    @staticmethod
    def col_name():
        return 'id,ムラタ案件管理番号,ﾌｧﾐﾘ管理番号,国コード,出願番号,公開/(再)公表番号,IPC（公開/筆頭）,IPC（公開/全件）,製品区分,製品コード,品種分類,戦略分類' 
