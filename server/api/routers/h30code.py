from email.errors import MessageDefect
from fastapi import APIRouter, Depends, HTTPException,Query,UploadFile
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session
from db import cruds, schemas,models
from db.database import get_db
import pdb
from typing import List
from mongo_db.model import CV
from fastapi.responses import PlainTextResponse
from enum import Enum
from werkzeug.utils import secure_filename
import os
import  shutil

UPLOAD_FOLDER = "./static"

ALLOWED_EXTENSIONS = {'xlsx', 'csv'}

def allowed_file(filename):
    return '.' in filename and            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


router = APIRouter(tags=["H30"])


    
class ModelName(str,Enum):
    ClassifyMasterHistory = "CLASSIFY"
    PalcomMaster = "PALCOM"
      
      
@router.post('/import/{model_name}')
async def data_import(
                      model_name: ModelName, 
                      files:List[UploadFile],
                      db: Session = Depends(get_db),
                      mode: str = Query("replace", enum=["replace", "upsert"]),
                      ):
    if files:
        
        if model_name is ModelName.ClassifyMasterHistory:
            model = models.ClassifyMasterHistory
    
        if model_name is ModelName.PalcomMaster:
            model = models.PalcomMaster
    
        #チェックボタンが押されていたらデータ消去する
        if mode =="replace":
            cruds.model_table_all_delete(db,model)
                    
        file_ok=[]
        file_ng=[]

        for file in files:
            if file and allowed_file(file.filename): 

                origin_filename = secure_filename(file.filename)
                filename =os.path.join(UPLOAD_FOLDER, origin_filename)
                #file.save(filename)
                upload_dir = open(filename,'wb+')
                shutil.copyfileobj(file.file, upload_dir)
                upload_dir.close()
        
                if cruds.model_data_import(db,filename,model):   #Processing depends on the model
                    file_ok.append(origin_filename)
                else:
                    file_ng.append(origin_filename)

            else:
                return {"Error failed","ファイル形式は{}にしてください。".format(', '.join([str(i) for i in ALLOWED_EXTENSIONS]))
                }
                

        strout1=""
        strout2=""
        
        if len(file_ok)!=0:
            strout1 = "ファイル{}を読み込みました。".format(file_ok)
        if len(file_ng)!=0:
            strout2 = "ファイル{}を読み込みに失敗しました。".format(filename)
            
        return {"result":strout1 + strout2}
    
    return {"Error",'No file part'}


@router.get("/getalldata/{model_name}")
async def get_all_data(
                      model_name: ModelName,
                      db: Session = Depends(get_db)):

    if model_name is ModelName.ClassifyMasterHistory:
        model = models.ClassifyMasterHistory
    
    if model_name is ModelName.PalcomMaster:
        model = models.PalcomMaster

    obj = jsonable_encoder(cruds.model_data_view(db,model))
    
    return JSONResponse(content=obj)

@router.get("/exec_query/")
async def get_all_data(
                      query: str,
                      db: Session = Depends(get_db)):

    obj = jsonable_encoder([item for item in cruds.exec_query(db,query)])
    
    return JSONResponse(content=obj)
