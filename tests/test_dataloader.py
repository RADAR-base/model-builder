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
        self.path = "/tmp/my_test_db"
        self.postgresql = testing.postgresql.PostgresqlFactory(cache_initialized_db=False, on_initialized=handler, database= self.name,
                    port=self.port, base_dir=self.path)()
        self.postgresql.initialize_database()
        self.querybuilder = QueryBuilder("test")
        self.query = self.querybuilder.get_all_columns()
        self.expected = '[{"id": 1, "value": "hello"}, {"id": 2, "value": "ciao"}]'

    def tearDown(self):
        self.postgresql.stop()

    def test_get_response(self):
        dbconn = PostgresJsonWrapper(dbname=self.name, port=self.port)
        dbconn.connect()
        response=dbconn.get_response(self.query)
        self.assertEqual(response, self.expected)
        dbconn.disconnect()

    def test_get_and_save_response(self):
        dbconn = PostgresJsonWrapper(dbname=self.name, port=self.port)
        dbconn.connect()
        filename = "tmp.json"
        dbconn.get_and_save_response(self.query, filename=filename)
        with open(filename) as json_file:
            response = json.load(json_file)
        self.assertEqual(response, self.expected)
        os.remove(filename)
        dbconn.disconnect()

class TestPostgresPandasWrapper(unittest.TestCase):
    def setUp(self):
        self.name = "test"
        self.port = 5678
        self.path = "/tmp/my_test_db"
        self.postgresql = testing.postgresql.PostgresqlFactory(cache_initialized_db=False, on_initialized=handler, database= self.name,
                    port=self.port, base_dir=self.path)()
        self.postgresql.initialize_database()
        self.querybuilder = QueryBuilder("test")
        self.query = self.querybuilder.get_all_columns()
        self.expected = pd.DataFrame([{"id": 1, "value": "hello"}, {"id": 2, "value": "ciao"}])

    def test_get_response(self):
        dbconn = PostgresPandasWrapper(dbname=self.name, port=self.port)
        dbconn.connect()
        response=dbconn.get_response(self.query)
        assert_frame_equal(response, self.expected)
        dbconn.disconnect()

    def test_get_and_save_response_csv(self):
        dbconn = PostgresPandasWrapper(dbname=self.name, port=self.port)
        dbconn.connect()
        filename = "tmp.csv"
        dbconn.get_and_save_response(self.query, filename=filename)
        response = pd.read_csv(filename)
        assert_frame_equal(response, self.expected)
        os.remove(filename)
        dbconn.disconnect()

    def test_get_and_save_response_json(self):
        dbconn = PostgresPandasWrapper(dbname=self.name, port=self.port)
        dbconn.connect()
        filename = "tmp.json"
        dbconn.get_and_save_response(self.query, filename=filename)
        response = pd.read_json(filename)
        assert_frame_equal(response, self.expected)
        os.remove(filename)
        dbconn.disconnect()

    def test_get_and_save_response_excel(self):
        dbconn = PostgresPandasWrapper(dbname=self.name, port=self.port)
        dbconn.connect()
        filename = "tmp.xlsx"
        dbconn.get_and_save_response(self.query, filename=filename)
        response = pd.read_excel(filename)
        assert_frame_equal(response, self.expected)
        os.remove(filename)
        dbconn.disconnect()
