# PatentField 教師データ作成支援ツール

アプリケーション構成

* client　　cytoscape / vis.jsによるネットワーク図 ---> wordcloud表示はcytoscape.jsのみ対応
* server　　Fast-Apiによるバックエンド　--->　wordcloud / Sckitlearn / Postgres用CRUD
* ipynb　　　pythonノートブック　テストスニペット置き場


# POSTGRES
python-fast-API　参考コード　---> Dockerでのデータベース起動
9290d66f742d   shibui/ml-system-in-actions:model_db_0.0.1   "./run.sh"               2 months ago    Up 4 hours   0.0.0.0:8001->8000/tcp, :::8001->8000/tcp   model_db
6bcbcca2e47f   postgres:13.3                                "docker-entrypoint.s…"   2 months ago    Up 4 hours   0.0.0.0:5432->5432/tcp, :::5432->5432/tcp   postgres

# Patent Field
https://patentfield.com/


server
```
cd server
pip install -r requirement.txt
uvicorn api.app:app --reload --host 0.0.0.0 --port 7999'
```

ファイヤーウォール (UBUNTU)
```
sudo ufw allow 7999
```

fontインストール　(ubuntu)
```
sudo apt install -y fonts-ipafont
fc-cache -fv
fc-list

pythonコードに下記を追加すること
/usr/share/fonts/truetype/fonts-japanese-gothic.ttf: IPAGothic,IPAゴシック:style=Regular
```

client
```
docker compose up -d
# port "3001:3000"

```
