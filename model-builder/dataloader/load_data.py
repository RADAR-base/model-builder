# Loading data from postgres

from typing import List
from postgres_json_wrapper import PostgresJsonWrapper
from postgres_pandas_wrapper import PostgresPandasWrapper


def main():
    # Testing Postgres json Wrapper
    dbconn = PostgresJsonWrapper(dbname="opendb", user="heet", password="password")
    dbconn.connect()
    #print(dbconn.get_response(cols=["*"], dataset="wine_dataset"))
    dbconn.get_and_save_response(cols=["*"], dataset="wine_dataset", filename="test_json.json")
    dbconn.disconnect()

    # Testing Postgres Pandas Wrapper
    dbconn = PostgresPandasWrapper(dbname="opendb", user="heet", password="password")
    dbconn.connect()
    #print(dbconn.get_response(cols=["*"], dataset="wine_dataset"))
    dbconn.get_and_save_response(cols=["*"], dataset="wine_dataset", filename="test_json.csv")
    dbconn.disconnect()

if __name__ == "__main__":
    main()