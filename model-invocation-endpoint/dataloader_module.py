
from pydantic import BaseModel,ValidationError, validator
from typing import List, Union, Dict
import datetime
from uuid import UUID

class DataInputModelSplit(BaseModel):
    columns: List[str]
    data: List[List[Union[int, float, str]]]
    format = "pandas_split"

    @validator('data')
    def data_size_must_be_equal_column(cls, v, values):
        column_len = len(values["columns"])
        for data_point in v:
            if len(data_point) != column_len:
                raise ValueError("Size of data point is different from that of column")
        return v

class DataInputModelRecord(BaseModel):
    record: List[Dict[str, Union[int, float, str]]]
    format = "pandas_record"

    @validator("record")
    def check_dict_keys(cls, v):
        if len(v) == 0:
            raise ValueError("Data is empty")
        first_dict_keys = set(v[0].keys())
        for dct in v:
            if first_dict_keys != set(dct.keys()):
                raise ValueError("Mismatch columns in the input")
        return v


class DataInputModelInputs(BaseModel):
    inputs: Dict[str, List[Union[int, float, str, List[Union[int, float, str]]]]]
    format = "tf-inputs"
    @validator("inputs")
    def check_input_format(cls, v):
        input_len = None
        for key, value in v.items():
            if input_len is None:
                input_len = len(value)
            elif input_len != len(value):
                raise ValueError("Inconsistent data size in input")
        return v

class DataInputModelInstances(BaseModel):
    instances: List[Dict[str, Union[int, float, str, List[Union[int, float, str]]]]]
    format = "tf-instances"
    @validator("instances")
    def check_instance_format(cls, v):
        if len(v) == 0:
            raise ValueError("Data is empty")
        first_dict_keys = set(v[0].keys())
        for dct in v:
            if first_dict_keys != set(dct.keys()):
                raise ValueError("Mismatch columns in the input")
        return v

DataInputModel = Union[DataInputModelSplit, DataInputModelInstances, DataInputModelInputs, DataInputModelRecord]

class DataLoaderClass(BaseModel):
    filename: str
    classname: str
    dbname: str
    starttime: datetime.datetime = None
    endtime: datetime.datetime = None
    user_id: str
    project_id: str
