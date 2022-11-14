from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
import json
from ML.tfidf import tfidf
from mongo_db.database import (
	fetch_one_crossValid,
	fetch_all_crossvalid,
	create_crossValid,
	#update_crossValid,
	remove_crossValid,
	freetext_insert_to_database,
	add_tfidf,
	upload_pf_crossvalid_data_format)

from ML.tfidffreetext import morphological_analysis,tfidf_calc
from typing import List
import io
import hashlib
import datetime
from db.database import get_db
from fastapi import APIRouter, Depends, HTTPException,Query
from mongo_db.model import CV
import pdb
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
import pandas as pd
from fastapi.responses import ORJSONResponse,HTMLResponse
from fastapi import Body,Response
import numpy as np

router = APIRouter(tags=["mongo"])

static_path = "./static/"

# ------------------------------------------------------------------
@router.get("/CrossValid2")
async def get_CrossValid2():
	response = await fetch_all_crossvalid()
	return response
# ------------------------------------------------------------------
@router.get("/CrossValid")
async def get_CrossValid():
	response = await fetch_all_crossvalid()
	for item in response:
		item.ai_predict_candidate = ""
	return response
# ------------------------------------------------------------------
@router.get("/CrossValid/{id}", response_model=CV)
async def get_CrossValid_by_id(id):
	response = await fetch_one_CrossValid(id)
	if response:
		return response
	raise HTTPException(404, f"there is no CrossValid item with this id {id}")
# ------------------------------------------------------------------
@router.post("/CrossValid", response_model=CV)
async def post_CrossValid(CrossValid:CV):
	response = await create_crossValid(CrossValid.dict())
	if response:
		return response
	raise HTTPException(400, "Sometheng went wrong / Bad Request")
# ------------------------------------------------------------------
@router.post("/CrossValid_pf_upload")#, response_model=CV)
async def post_CrossValid_pf_upload(
	db: Session = Depends(get_db),
	token:str = None,
    files: List[UploadFile] = File(...)
):
	if len(files)==1:
	    response = await upload_pf_crossvalid_data_format(db,token, files[0])
	    if response:
		    return response
	raise HTTPException(400, "Sometheng went wrong / Bad Request")
# ------------------------------------------------------------------
#@router.put("/api/CrossValid/{id}/", response_model=CV)
#async def put_CrossValid(id:str, population:int):
#	response = await update_CrossValid(id, population)
#	if response:
#		return response
#	raise HTTPException(404, f"there is no CrossValid item with this id {id}")

# ------------------------------------------------------------------
@router.delete("/CrossValid/{id}")
async def delete_CrossValid(id):
	response = await remove_crossValid(id)
	if response:
		return "Successfully deleted CrossValid item!"
	raise HTTPException(404, f"there is no CrossValid item with this id {id}")

#patentリストの読み込み(excel)
@router.post('/read_freetext')
async def post_read_freetext(files: UploadFile = File(...)): #files: List[UploadFile] = File(...)):
    if files:
        coll = await freetext_insert_to_database(filename = files)
        return {"token" : coll}

    return {"Error" : "Upload file not found"}
    #return excelset_file_read(db)

#tfidf計算
@router.get('/add_tfidf')
async def add_tfidf_get(token: str = ""):
	if token:
		df = await add_tfidf(token)
		stream = io.StringIO()
		df.to_csv(stream, encoding="cp932",sep="\t")
		response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
		response.headers["Content-Disposition"] = "attachment; filename={}.txt".format(token+hashlib.md5(str(datetime.datetime.now()).encode()).hexdigest())
		return response	
	
	return {"Error" : "token not found"}

@router.get('/keitaiso',response_class=HTMLResponse)
def keitaiso_get(
    token_text: str = "",
    output : str = Query("csv", enum=["txt", "csv","json"])):
            
    if token_text:
        txt = morphological_analysis(token_text)
        df = pd.json_normalize(txt)
        if output == "csv":
            stream = io.StringIO()
            df.to_csv(stream, encoding="cp932",sep="\t")
            response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
            response.headers["Content-Disposition"] = "attachment; filename={}.txt".format(hashlib.md5(str(datetime.datetime.now()).encode('utf-8')).hexdigest())
            return response	
        
        if output == "json":
            return df.to_json(orient="records",force_ascii=False)
        if output == "txt":
            return df.to_string(index=False)
	
    return {"Error" : "token not found"}

#JSONで与えられた文字列から、tfidfを計算するAPI
@router.post('/add_tfidf_json')
def add_tfidf_json(
    body = Body(...)
 ):
    
    token_text= body["token_text"]
    grammar = body["grammar"]
    mode = body["mode"]
    
    text_input = []
    if token_text:
        texts = json.loads(token_text)
        for text in texts:
            text_input.append(text)
        
        output = tfidf_calc(target=text_input,max_word=1000,grammar=grammar,mode=mode )
        types = body["output"]
        
        if types == "json":
            return output
        if types == "txt":
            return df.to_string(index=False)
        
    return {"Error" : "token not found"}

@router.get('/add_tfidf')
async def add_tfidf_get(token: str = ""):
	if token:
		df = await add_tfidf(token)
		stream = io.StringIO()
		df.to_csv(stream, encoding="cp932",sep="\t")
		response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
		response.headers["Content-Disposition"] = "attachment; filename={}.txt".format(token+hashlib.md5(str(datetime.datetime.now()).encode()).hexdigest())
		return response	
	
	return {"Error" : "token not found"}


#JSONで与えられた文字列から、logistic回帰を計算するAPI
@router.post('/logistic')
def logistic(
	body = Body(...)
 ):
	
	token_text= body["token_text"]
	#grammar = body["grammar"]
	mode = body["mode"]
	head = body["head"] 
	df_s = pd.DataFrame()
	text_input = []
	if token_text:
		json_item_data = json.loads(token_text)
		for k,v in json_item_data.items():
			df_s = df_s.append({'patent_id':k,'ai_train_label':k , 'tfidf':v},ignore_index=True)

		tfidf_arr, label_arr,label_names ,unique_word= tfidf(df_s,head)

    # pickleにつける

		token = "logistic"
  
		np.save(static_path +'tfidf_arr.npy', tfidf_arr)
		np.save(static_path +'label_arr.npy', label_arr)
		np.save(static_path +'label_names.npy', label_names)
		np.save(static_path + 'unique_names.npy', unique_word)
    # リダイレクト
		return {"token": token, "tfidf_arr": tfidf_arr.shape, "label_arr": label_arr.shape, "label_names": label_names.shape, "unique_word": unique_word.shape}
    #return RedirectResponse("/calc_crossvalid/?token={}&numofword={}".format(token, numofword), status_code=303)
    
    #except:
#        return {"Error": "Upload file not found"}

