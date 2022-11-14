import io
import typing
from ast import Bytes
from typing import List
import base64

from urllib3 import HTTPResponse

from db.database import get_db
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from flask import jsonify
from sklearn.feature_extraction import img_to_graph
from sqlalchemy.orm import Session
from vutil.dialogs import get_wordcloud,file_upload
from db import models, schemas,cruds

import pdb
from io import BytesIO

router = APIRouter(tags=["view"])

#patentリストの読み込み(excel)
@router.get('/upload_patent_clear')
def post_read_patent(db: Session = Depends(get_db)):
    return cruds.delete_all_patent(db)

#patentリストの読み込み(excel)
@router.post('/upload_patent')
async def post_read_patent(db: Session = Depends(get_db), files: List[UploadFile] = File(...)): #files: List[UploadFile] = File(...)):

    if files:
        patent = cruds.patent_insert_to_database(db=db,files = files)
        return patent
    return {"Error" : "Upload file not found"}
    #return excelset_file_read(db)

#CROSSVALIDリストの読み込み(csv)
@router.post('/read_crossvalid_postgres')
def post_read_crossvalid(db: Session = Depends(get_db), files:UploadFile= File(...)):
    if files:
        crossvalid = cruds.crossvalid_insert_to_database(db=db,filename = files)
        return crossvalid
    return {"Error" : "Upload file not found"}
    #return crossvalid_file_read(db)


@router.post("/upload/")
def test_multiple_upload(db: Session = Depends(get_db), files: List[UploadFile] = File(...)):
    return {"filenames": file_upload(db=db, files=files)}


@router.get('/wordcloud')
def getwordcloud(db: Session = Depends(get_db),
                 mode: str = Query("出願番号", enum=["出願番号", "公開番号"]),
                 patent_number: str = Query(
                     default="JP2018098190A", max_length=20),
                 word_num: int = Query(default=10, ge=1, le=100),
                 img: str = Query(default="svg", enum=["svg", "png"])
                 ):

    patent = cruds.search_patent(db=db, mode=mode, patent_number=patent_number)
   # pdb.set_trace()

    if not patent is None:
        print(patent.syutugan_number,word_num)

        wordcloud = get_wordcloud(patent.tfidf, cutoff=word_num, img=img)
        return wordcloud
        #data = base64.b64encode(wordcloud.encode())
        #return {"data": data}

    else:
        raise HTTPException(status_code=404)
        
