import sqlite3
import torch
import pickle
import codecs
import sys

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

def countTuples(conn, tableName):
    cursor = conn.cursor()
    rowsCount = cursor.execute(f"select count(*) from {tableName}")
    nrows = next(rowsCount)[0]
    return nrows

def getTuples(conn, tableName, colName):
    nrows = countTuples(conn, tableName)
    print(f"nrows: {nrows}")
    cursor = conn.cursor()
    sql = f"select mode, epoch, batch_idx, decoding, {colName}, frame_embs, caps_in, tok_embs_outs from {tableName} order by batch_idx desc"
    rows = cursor.execute(sql)
    counter = 1
    limit = 10
    for row in rows:
        weightsTensor = pickle.loads(codecs.decode(row[4].encode(),'base64'))
        frameEmbsTensor = pickle.loads(codecs.decode(row[5].encode(),'base64'))
        capsInTensor = pickle.loads(codecs.decode(row[6].encode(),'base64'))
        tokEmbsOutsTensor = pickle.loads(codecs.decode(row[7].encode(),'base64'))
        # Average the heads
        # weightsTensor = weightsTensor.mean(dim=0, keepdim=True)
        print(f"""<mode, epoch, batch_idx, decoding, weights.shape, frame_embs.shape, caps_in.shape tok_embs_outs.shape>: 
              {row[0]}, {str(row[1])}, {str(row[2])}, {row[3]}, {weightsTensor.shape}, {frameEmbsTensor.shape}, {capsInTensor.shape}, {tokEmbsOutsTensor.shape}""")
        if counter == limit:    
            break
        counter += 1
    cursor.close()

def main(args):
    dbName = args[0]
    tableName = "weights"
    colName = "weights"
    conn = createDB(dbName, tableName, colName)
    print("Counting tensors")
    countTuples(conn, tableName)
    print("Retrieving tensors")
    getTuples(conn, tableName, colName)

if __name__ == "__main__":
    main(sys.argv[1:])
    



