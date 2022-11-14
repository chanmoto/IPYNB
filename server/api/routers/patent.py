from email.errors import MessageDefect
from fastapi import APIRouter, Depends, HTTPException,Query
from sqlalchemy.orm import Session
from db import cruds, schemas
from db.database import get_db
import pdb
from typing import List
from mongo_db.model import CV
from fastapi.responses import PlainTextResponse

router = APIRouter(tags=["patent"])

@router.get("/crossvalid/all")#,response_model=List[CV])
async def crossvalid_all(db: Session = Depends(get_db),number: int = Query(..., gt=0)):
    items = cruds.select_patent_all(db=db, number=number)
    #convert to crossvalid model
    items_ = cruds.convert_to_crossvalid(items)

    return items_

@router.get("/patent/all",response_model=List[schemas.Patent])
async def patent_all(db: Session = Depends(get_db),number: int = Query(..., gt=0),
                     mode: str = Query("light", enum=["ligth", "full"])):
    if mode == "full":
        items = cruds.select_patent_all(db=db, number=number)
    else:
        items = cruds.select_patent_all_light(db=db, number=number)
    return items


@router.post("/patent")
async def add_patent(
    model: schemas.Patent,
    db: Session = Depends(get_db),
)->schemas.Patent:
    return cruds.add_patent(
        db=db,
        model=model,
        commit=True,
    )

@router.delete("/patent/all")
async def delete_all_patent(db: Session = Depends(get_db)):

    message = cruds.delete_all_patent(db=db)
    if not message:
        raise HTTPException(status_code=404, detail="Patent not found")
    return {"message": "Patent deleted"}

    
@router.delete("/patent/{patent_id}")
async def delete_patent(
    patent_id: str,
    db: Session = Depends(get_db),
)->schemas.Patent:
    
    message = cruds.delete_patent(patent_id=patent_id, db=db)
    
    if not message:
        raise HTTPException(status_code=404, detail="Patent not found")
    return {"message": "Patent deleted"}

#221005
#StreamlitからURLリンク表示用のAPI呼び出し
@router.get("/patent",response_model=schemas.Patent)
async def get_patent_one(
    db: Session = Depends(get_db),
    mode: str = Query("出願番号", enum=["出願番号", "公開番号"]),
    patent_number: str = Query(
                default="JPWO2019087753", max_length=20)
    ):
    
    patent = cruds.search_patent(db=db, mode=mode, patent_number=patent_number)
 
    if patent is None:
        raise HTTPException(status_code=404, detail="URL not found")
    else:
        return patent
      
