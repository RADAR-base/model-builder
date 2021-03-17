import simplejson as json
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List
from dataloader import PostgresWrapper

class PostgresJsonWrapper(PostgresWrapper):
    def __init__(self, **args):
        super().__init__(**args)

    def connect(self):
        self.connection = psycopg2.connect(
            user=self.user,
            password=self.password,
            port=self.port,
            dbname=self.dbname,
            host=self.host
        )

    def _get_json_cursor(self):
        return self.connection.cursor(cursor_factory=RealDictCursor)

    @staticmethod
    def _execute_and_fetch(cursor, query):
        cursor.execute(query)
        res = cursor.fetchall()
        cursor.close()
        return res

    def get_response(self, query: str):
        cursor = self._get_json_cursor()
        response = self._execute_and_fetch(cursor, query)
        return json.dumps(response)

    def save_response(self, response, filename):
        with open(filename, 'w') as outfile:
            json.dump(response, outfile)

    def disconnect(self):
        self.connection.close()
