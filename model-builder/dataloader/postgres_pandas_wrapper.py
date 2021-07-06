import pandas as pd
from sqlalchemy import create_engine
import psycopg2
from typing import List
from dataloader import PostgresWrapper

class PostgresPandasWrapper(PostgresWrapper):
    def __init__(self, **args):
        super().__init__(**args)

    def _create_db_url(self):
        db_url = "postgresql+psycopg2://" +  self.user + ":" + self.password + "@"+ self.host + ":" + str(self.port) + "/" + self.dbname
        return db_url

    def connect(self):
        db_url = self._create_db_url()
        alchemy_engine  = create_engine(db_url, pool_recycle=3600)
        self.connection = alchemy_engine.connect()

    def get_response(self, queries: List[str]):
        response = []
        for query in queries:
            response.append(pd.read_sql(query, self.connection))
        return response

    def save_response(self, response, filename):
        file_format = filename.split(".")[-1]
        if file_format == "csv":
            response.to_csv(filename, index=False)
        elif file_format == "xlsx" or file_format == "xls":
            response.to_excel(filename, index=False)
        elif file_format == "json":
            response.to_json(filename)
        else:
            raise IOError("File format either not implemented or not available")

    def disconnect(self):
        self.connection.close()
