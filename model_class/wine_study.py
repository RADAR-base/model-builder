from model_class import  ModelClass
import pandas as pd
from datetime import  timedelta
import datetime
from datetime import datetime as dt
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('../model-builder/'))
from dataloader.querybuilder import QueryBuilder

class WineStudy(ModelClass):

    def __init__(self):
        self.tablename = "wine_dataset"
        self.querybuilder = QueryBuilder(tablename=self.tablename)
        super().__init__()

    def get_query_for_training(self):
        return [self.querybuilder.get_all_columns()]

    def get_query_for_prediction(self, user_id, project_id, starttime, endtime):
        return [f"{self.querybuilder.get_all_columns()} limit 10"]

    def preprocess_data(self, data):
        return data[0], data[0].index

    def create_return_obj(self, indexes, model_name, model_version, inference_result):
        dateTimeObj = dt.now(tz=None)
        print(indexes)
        return_obj = pd.DataFrame({"idx":indexes})
        return_obj["invocation_result"] = [{"wine_quality": result} for result in inference_result]
        return_obj["model_name"] = model_name
        return_obj["model_version"] = model_version
        return_obj["timestamp"] = dateTimeObj.timestamp()
        return_obj['invocation_result'] = return_obj.invocation_result.map(self.dict2json)
        return return_obj