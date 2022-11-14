# ------------------------------------------------------------------
#	model.py
#
#					May/02/2022
# ------------------------------------------------------------------
from pydantic import BaseModel
from typing import Dict, Optional

class CV(BaseModel):
    id:str
    group :int
    querystr: str
    ai_train_label: str
    ai_predict_label: str
    ai_predict_score: str
    ai_predict_candidate: str
    class Config:
        orm_mode = True
