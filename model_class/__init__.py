from abc import ABC, abstractmethod
import sys

class ModelClass(ABC):

    #def __init__(self):
    #   pass

    @abstractmethod
    def get_query_for_training(self):
        pass

    @abstractmethod
    def get_query_for_prediction(self, user_id, project_id, starttime, endtime):
        pass

    @abstractmethod
    def preprocess_data(self, data):
        pass
    @abstractmethod
    def create_return_obj(self):
        pass