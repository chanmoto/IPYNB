from logging import getLogger

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import patent,view,h30code
from ML import router as ml
from mongo_db import router as mongo
from configurations import APIConfigurations
from db import initialize
from db.database import engine
import pdb


logger = getLogger(__name__)

initialize.initialize_table(engine=engine, checkfirst=True)

app = FastAPI(
    title=APIConfigurations.title,
    description=APIConfigurations.description,
    version=APIConfigurations.version,
)

origins=[
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )

app.include_router(ml.router)
app.include_router(mongo.router)
app.include_router(patent.router)
app.include_router(view.router)
app.include_router(h30code.router)

"""
app.include_router(health.router, prefix=f"/v{APIConfigurations.version}/health", tags=["health"])
app.include_router(api.router, prefix=f"/v{APIConfigurations.version}/api", tags=["api"])
"""

