import datetime
from typing import Dict, Optional

from pydantic import BaseModel
from sqlalchemy.dialects.postgresql import hstore
from sqlalchemy.dialects.postgresql import HSTORE
"""
URL	スコア	出願番号	出願日	公開番号	公開日	登録番号	登録発行日	名称	AI教師ラベル	AI予測スコア	AI予測ラベル	特徴キーワード(名称)
"""
        
class Patent(BaseModel):
    patent_id: str
    syutugan_number: str
    syutugan_date:str
    koukai_number: str
    koukai_date:str
    title: str
    ai_train_label: str
    ai_predict_score: str
    ai_predict_label: str
    tfidf: Optional [Dict[str, str]] = None
    url: str

    class Config:
        orm_mode = True

        """	n	クエリ	AI教師ラベル	AI予測ラベル	AI予測スコア	AI予測ラベル候補	AI予測ラベル候補スコア
        """



"""
class ProjectBase(BaseModel):
    project_name: str
    description: Optional[str]


class ProjectCreate(ProjectBase):
    pass


class Project(ProjectBase):
    project_id: int
    created_datetime: datetime.datetime

    class Config:
        orm_mode = True


class ModelBase(BaseModel):
    project_id: str
    model_name: str
    description: Optional[str]


class ModelCreate(ModelBase):
    pass


class Model(ModelBase):
    model_id: int
    created_datetime: datetime.datetime

    class Config:
        orm_mode = True


class ExperimentBase(BaseModel):
    model_id: str
    model_version_id: str
    parameters: Optional[Dict]
    training_dataset: Optional[str]
    validation_dataset: Optional[str]
    test_dataset: Optional[str]
    evaluations: Optional[Dict]
    artifact_file_paths: Optional[Dict]


class ExperimentCreate(ExperimentBase):
    pass


class ExperimentEvaluations(BaseModel):
    evaluations: Dict


class ExperimentArtifactFilePaths(BaseModel):
    artifact_file_paths: Dict


class Experiment(ExperimentBase):
    experiment_id: int
    created_datetime: datetime.datetime

    class Config:
        orm_mode = True

"""


# 総括的なプロジェクト
class ProjectBase(BaseModel):
    project_name: str
    description: Optional[str]

class ProjectCreate(ProjectBase):
    pass

class Project(ProjectBase):
    project_id: int
    created_datetime: datetime.datetime

    class Config:
        orm_mode = True

# トレインデータ
class TrainData(BaseModel):
    model_id = str
    project_id = str
    patent_number = str
    label_name = str
    
    class Config:
        orm_mode = True