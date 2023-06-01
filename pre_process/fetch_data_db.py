import json
import sqlite3


class FetchData(object):

    def __init__(self, db_path):
        self.path = db_path
        self.conn = sqlite3.connect(self.path)

    def close(self):
        self.conn.close()

    def fetch_doc_ids(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM wiki LIMIT 10")
        res = cursor.fetchall()
        return res
    
    def fetch_text(self, k_page):
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT data from wiki where id =  '{k_page}' ");
        result = cursor.fetchone()
        res = json.loads(result[0])

        return res
    