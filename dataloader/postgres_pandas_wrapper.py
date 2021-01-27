import pandas as pd
from sqlalchemy import create_engine
import psycopg2
from typing import List

class PostgresPandasWrapper:
    def __init__(self, **args):
        self.user = args.get('user', 'postgres')
        self.password = args.get('password', "")
        self.port = args.get('port', 5432)
        self.dbname = args.get('dbname', 'opendb')
        self.host = args.get('host', 'localhost')
        self.connection = None

    def _create_db_url(self):
        db_url = "postgresql+psycopg2://" +  self.user + ":" + self.password + "@"+ self.host + ":" + str(self.port) + "/" + self.dbname
        return db_url

    def connect(self):
        db_url = self._create_db_url()
        alchemy_engine  = create_engine(db_url, pool_recycle=3600)
        pg_conn = alchemy_engine.connect()
        self.connection = pg_conn

    def get_response(self, cols:List[str ], dataset:str):
        query = self.querymaker(cols, dataset)
        response = pd.read_sql(query, self.connection)
        return response

    def save_response(self, response, filename):
        file_format = filename.split(".")[-1]
        if file_format == "csv":
            response.to_csv(filename, index=False)
        elif file_format == "xlsx" or file_format == "xls":
            response.to_excel(filename, index=False)
        elif file_format == "json":
            response.to_json(filename, index=False)
        else:
            raise IOError("File format either not implemented or not available")

    def get_and_save_response(self, cols:List[str ], dataset:str, filename):
        response = self.get_response(cols, dataset)
        self.save_response(response,filename)

    def querymaker(self, cols:List[str], dataset:str):
        query = "SELECT " + ", ".join(cols) + "FROM " + dataset
        return query

    def disconnect(self):
        self.connection.close()
        print("Database closed successfully")
