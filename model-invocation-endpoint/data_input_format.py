
from pydantic import BaseModel,ValidationError, validator
from typing import List, Union, Dict

class DataInputModelSplit(BaseModel):
    columns: List[str]
    data: List[List[Union[int, float]]]

    @validator('data')
    def data_size_must_be_equal_column(cls, v, values):
        column_len = len(values["columns"])
        print(column_len)
        for data_point in v:
            if len(data_point) != column_len:
                raise ValueError("Size of data point is different from that of column")
        return v

class DataInputModelRecord(BaseModel):
    data: List[Dict[str, Union[int, float]]]

    @validator("data")
    def check_dict_keys(cls, v):
        if len(v) == 0:
            raise ValueError("Data is empty")
        first_dict_keys = set(v[0].keys())
        for dct in v:
            if first_dict_keys != set(dct.keys()):
                raise ValueError("Mismatch columns in the input")
        return v


class DataInputModelInputs(BaseModel):
    inputs: Dict[str, List[Union[int, float]]]

    @validator("inputs")
    def check_input_format(cls, v):
        input_len = None
        for key, value in v.iteritems():
            if input_len is None:
                input_len = len(value)
            elif input_len != len(value):
                raise ValueError("Inconsistent data size in input")
        return v

class DataInputModelInstances(BaseModel):
    instances: List[Dict[str, Union[int, float]]]

    @validator("instances")
    def check_instance_format(cls, v):
        if len(v) == 0:
            raise ValueError("Data is empty")
        first_dict_keys = set(v[0].keys())
        for dct in v:
            if first_dict_keys != set(dct.keys()):
                raise ValueError("Mismatch columns in the input")
        return v