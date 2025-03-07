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
    trainLaunchedByTest = False
    inference = False

    def __init__(self):
        timestmp = int(time.time())
        dbName = f"attn-{timestmp}.db"
        self.dbName = dbName
        self.tableName = "weights"
        self.colName = "weights"
        self.conn = None

    def createDB(self):
        self.conn = sqlite3.connect(self.dbName)
        cursor = self.conn.cursor()
        sql = f"""create table if not exists {self.tableName} 
        (mode string, epoch int, batch_idx int, decoding string, {self.colName} string, frame_embs string, caps_in string, tok_embs_outs)"""
        cursor.execute(sql)
        sql = f"""create table if not exists output
        (mode string, epoch int, batch_idx int, decoding string, outs string, cands string)"""
        cursor.execute(sql)
        cursor.close()

    def closeDB(self):
        self.conn.commit()
        self.conn.close()

    def addTuple(self, mode, epoch, batch_idx, decoding, attn_tensor, frame_embs_tensor, caps_in_tensor, tok_embs_outs_tensor):
        if self.inference:
            mode = "inference"
        elif self.trainLaunchedByTest:
            mode = "test"
        cursor = self.conn.cursor()
        pickledAttn = pickle.dumps(attn_tensor, -1)
        pickledAttnCoded = codecs.encode(pickledAttn, "base64").decode()
        pickledFrameEmbs = pickle.dumps(frame_embs_tensor, -1)
        pickledFrameEmbsCoded = codecs.encode(pickledFrameEmbs, "base64").decode()
        pickledCapsIn = pickle.dumps(caps_in_tensor, -1)
        pickledCapsInCoded = codecs.encode(pickledCapsIn, "base64").decode()
        pickledTokEmbsOuts = pickle.dumps(tok_embs_outs_tensor, -1)
        pickledTokEmbsOutsCoded = codecs.encode(pickledTokEmbsOuts, "base64").decode()
        sql = f"insert into {self.tableName}(mode, epoch, batch_idx, decoding, {self.colName}, frame_embs, caps_in, tok_embs_outs) values (?, ?, ?, ?, ?, ?, ?, ?)"
        cursor.execute(sql, (mode, epoch, batch_idx, decoding, pickledAttnCoded, pickledFrameEmbsCoded, pickledCapsInCoded, pickledTokEmbsOutsCoded))
        cursor.close()

    def addOuputTuple(self, mode, epoch, batch_idx, decoding, outs, cands):
        if self.inference:
            mode = "inference"
        elif self.trainLaunchedByTest:
            mode = "test"
        cursor = self.conn.cursor()
        sql = f"insert into output(mode, epoch, batch_idx, decoding, outs, cands) values (?, ?, ?, ?, ?, ?)"
        cursor.execute(sql, (mode, epoch, batch_idx, decoding, outs, cands))
        cursor.close()
