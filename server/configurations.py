import os


class DBConfigurations:
    postgres_username = "user"   # local C 変更後
    postgres_password = "password" # local C 変更後
    postgres_port = 5432
    postgres_db = "model_db" # local C 変更後
    postgres_server = "172.26.106.79" # local C 変更後
    sql_alchemy_database_url = (
        f"postgresql://{postgres_username}:{postgres_password}@{postgres_server}:{postgres_port}/{postgres_db}"
    )

class APIConfigurations:
    title = os.getenv("API_TITLE", "Model_DB_Service")
    description = os.getenv("API_DESCRIPTION", "machine learning system training patterns")
    version = os.getenv("API_VERSION", "0.1")
