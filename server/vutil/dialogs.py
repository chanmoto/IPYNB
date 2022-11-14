import uuid
from typing import Dict, List, Optional
from fastapi import Form
from db import models, schemas,cruds
from matplotlib.pyplot import title
from sqlalchemy.orm import Session
import requests
import shutil
import pdb 
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt
from io import BytesIO 
import base64
import pandas as pd
import numpy as np


# makes the circle using numpy
#x, y = np.ogrid[:300, :300]
#mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
#mask = 255 * mask.astype(int)

def get_wordcloud(dics, cutoff=5,img="svg")->WordCloud:

    df = pd.DataFrame(list(dics.items()),columns=['word', 'tfidf'])
    df = df.astype({'word': 'str', 'tfidf': 'float'})
    df = df.sort_values(by='tfidf', ascending=False).head(cutoff)
    dict_parse = dict(zip(df['word'], df['tfidf']))

    if dict_parse:
        wc = WordCloud(
            #mask=mask,
        width = 600,
        height =300,
        prefer_horizontal=1,
        background_color='white',
        include_numbers=True,
#        colormap='tab20',
       font_path='/usr/share/fonts/truetype/fonts-japanese-gothic.ttf',
        stopwords=STOPWORDS).generate_from_frequencies(dict_parse)
    
        if img == "svg":
            return wc.to_svg()  
        
        else:
            pass
 

def file_upload(db:Session,files):
    for file in files:
        with open(file.filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    return {"filenames": [file.filename for file in files]}


      