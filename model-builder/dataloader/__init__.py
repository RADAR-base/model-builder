from abc import ABC, abstractmethod
from typing import List

class PostgresWrapper(ABC):

    def __init__(self, **kwargs):
        self.user = kwargs.get('user', 'postgres')
        self.password = kwargs.get('password', "")
        self.port = kwargs.get('port', 5432)
        self.dbname = kwargs.get('dbname', 'opendb')
        self.host = kwargs.get('host', 'localhost')
        self.connection = None

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def get_response(self, queries: List[str]):
        pass

    @abstractmethod
    def save_response(self, responses, filenames):
        pass

    @abstractmethod
    def disconnect(self):
        pass
