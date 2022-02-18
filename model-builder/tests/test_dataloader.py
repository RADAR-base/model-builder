import unittest
import testing.postgresql
import psycopg2
from dataloader.postgres_json_wrapper import PostgresJsonWrapper
from dataloader.postgres_pandas_wrapper import PostgresPandasWrapper
from dataloader.querybuilder import QueryBuilder
import json
import os
import pandas as pd
from pandas.testing import assert_frame_equal

def handler(postgresql):
        conn = psycopg2.connect(**postgresql.dsn())
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test(id int, value varchar(256))")
        cursor.execute("INSERT INTO test values(1, 'hello'), (2, 'ciao')")
        cursor.close()
        conn.commit()
        conn.close()

class TestPostgresJsonWrapper(unittest.TestCase):
    def setUp(self):
        self.name = "test"
        self.port = 5678
        self.postgresql = testing.postgresql.PostgresqlFactory(cache_initialized_db=True, on_initialized=handler, database= self.name,
                    port=self.port)()
        self.postgresql.initialize_database()
        self.querybuilder = QueryBuilder("test")
        self.queries = [self.querybuilder.get_all_columns()]
        self.expected = ['[{"id": 1, "value": "hello"}, {"id": 2, "value": "ciao"}]']

    def tearDown(self):
        self.postgresql.stop()

    def test_get_response(self):
        dbconn = PostgresJsonWrapper(dbname=self.name, port=self.port)
        dbconn.connect()
        responses=dbconn.get_response(self.queries)
        for response, expected_instances in zip(responses, self.expected):
            self.assertEqual(response, expected_instances)
        dbconn.disconnect()

class TestPostgresPandasWrapper(unittest.TestCase):
    def setUp(self):
        self.name = "test"
        self.port = 5678
        self.postgresql = testing.postgresql.PostgresqlFactory(cache_initialized_db=True, on_initialized=handler, database= self.name,
                    port=self.port)()
        self.postgresql.initialize_database()
        self.querybuilder = QueryBuilder("test")
        self.queries = [self.querybuilder.get_all_columns()]
        self.expected = [pd.DataFrame([{"id": 1, "value": "hello"}, {"id": 2, "value": "ciao"}])]
        self.expected_after_inserting = [pd.DataFrame([{"id": 1, "value": "hello"}, {"id": 2, "value": "ciao"}, {"id": 3, "value": "Hallo"}, {"id": 4, "value": "Bonjour"}])]

    def test_get_response(self):
        dbconn = PostgresPandasWrapper(dbname=self.name, port=self.port)
        dbconn.connect()
        responses=dbconn.get_response(self.queries)
        for response, expected_instance in zip(responses, self.expected):
            assert_frame_equal(response, expected_instance)
        dbconn.disconnect()

    def test_insert_data(self):
        dbconn = PostgresPandasWrapper(dbname=self.name, port=self.port)
        dbconn.connect()
        dataframe = pd.DataFrame([{"id": 3, "value": "Hallo"}, {"id": 4, "value": "Bonjour"}])
        responses=dbconn.insert_data(dataframe, "test")
        # Checking response
        responses = dbconn.get_response(self.queries)
        for response, expected_instance in zip(responses, self.expected_after_inserting):
            assert_frame_equal(response, expected_instance)
        dbconn.disconnect()


    def tearDown(self):
        self.postgresql.stop()
