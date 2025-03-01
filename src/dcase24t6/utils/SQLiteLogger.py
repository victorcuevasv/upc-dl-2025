import sqlite3
import torch
import pickle
import codecs
import time

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    
class SQLiteLogger(metaclass=Singleton):

    dbName = None
    tableName = None
    colName = None
    conn = None

    def __init__(self):
        timestmp = int(time.time())
        self.dbName = f"attn{timestmp}.db"
        self.tableName = "weights"
        self.colName = "weights"
        self.conn = None

    def createDB(self):
        self.conn = sqlite3.connect(self.dbName)
        cursor = self.conn.cursor()
        sql = f"create table if not exists {self.tableName} ({self.colName} string)"
        cursor.execute(sql)
        cursor.close()

    def closeDB(self):
        self.conn.commit()

    def addTuple(self, tensor):
        cursor = self.conn.cursor()
        pickled = pickle.dumps(tensor, -1)
        pickledCoded = codecs.encode(pickled, "base64").decode()
        sql = f"insert into {self.tableName}({self.colName}) values (?)"
        cursor.execute(sql, (pickledCoded,))
        cursor.close()
