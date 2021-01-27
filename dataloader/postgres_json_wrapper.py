import simplejson as json
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List

class PostgresJsonWrapper:
    def __init__(self, **args):
        self.user = args.get('user', 'postgres')
        self.password = args.get('password', "")
        self.port = args.get('port', 5432)
        self.dbname = args.get('dbname', 'opendb')
        self.host = args.get('host', 'localhost')
        self.connection = None

    def connect(self):
        pg_conn = psycopg2.connect(
            user=self.user,
            password=self.password,
            port=self.port,
            dbname=self.dbname,
            host=self.host
        )

        self.connection = pg_conn

    def _get_json_cursor(self):
        return self.connection.cursor(cursor_factory=RealDictCursor)

    @staticmethod
    def _execute_and_fetch(cursor, query):
        cursor.execute(query)
        res = cursor.fetchall()
        cursor.close()
        return res

    def get_response(self, cols:List[str ], dataset:str):
        cursor = self._get_json_cursor()
        query = self.querymaker(cols, dataset)
        response = self._execute_and_fetch(cursor, query)
        return json.dumps(response)

    def save_response(self, response, filename):
        with open(filename, 'w') as outfile:
            json.dump(response, outfile)

    def get_and_save_response(self, cols:List[str ], dataset:str, filename):
        response = self.get_response(cols, dataset)
        self.save_response(response,filename)

    def querymaker(self, cols:List[str], dataset:str):
        query = "SELECT " + ", ".join(cols) + "FROM " + dataset
        return query

    def disconnect(self):
        self.connection.close()
        print("Database closed successfully")
