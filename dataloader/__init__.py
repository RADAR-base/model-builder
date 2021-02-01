from abc import ABC, abstractmethod
from typing import List

class PostgresWrapper(ABC):

    def __init__(self, **args):
        self.user = args.get('user', 'postgres')
        self.password = args.get('password', "")
        self.port = args.get('port', 5432)
        self.dbname = args.get('dbname', 'opendb')
        self.host = args.get('host', 'localhost')
        self.connection = None

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def get_response(self, cols: List[str ], tablename:str):
        pass

    @abstractmethod
    def save_response(self, response, filename):
        pass

    def get_and_save_response(self, cols:List[str ], tablename:str, filename: str):
        response = self.get_response(cols, tablename)
        self.save_response(response,filename)

    def querymaker(self, cols:List[str], tablename:str):
        query = "SELECT " + ", ".join(cols) + "FROM " + tablename
        return query

    @abstractmethod
    def disconnect(self):
        pass
