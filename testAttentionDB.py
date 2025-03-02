import sqlite3
import torch
import pickle
import codecs
import sys

def createDB(dbName, tableName, colName):
    conn = sqlite3.connect(dbName)
    cursor = conn.cursor()
    sql = f"create table if not exists {tableName} ({colName} string)"
    cursor.execute(sql)
    cursor.close()
    return conn

def countTuples(conn, tableName):
    cursor = conn.cursor()
    sql = f"select count(*) from {tableName}"
    rows = cursor.execute(sql)
    for row in rows:
        print(row)

def getTuples(conn, tableName, colName):
    cursor = conn.cursor()
    sql = f"select mode, epoch, batch_idx, decoding, {colName}, frame_embs, caps_in, tok_embs_outs from {tableName} order by batch_idx desc"
    rows = cursor.execute(sql)
    counter = 0
    limit = 40
    for row in rows:
        weightsTensor = pickle.loads(codecs.decode(row[4].encode(),'base64'))
        frameEmbsTensor = pickle.loads(codecs.decode(row[5].encode(),'base64'))
        capsInTensor = pickle.loads(codecs.decode(row[6].encode(),'base64'))
        tokEmbsOutsTensor = pickle.loads(codecs.decode(row[7].encode(),'base64'))
        print(f"""<mode, epoch, batch_idx, decoding, weights.shape, frame_embs.shape, caps_in.shape tok_embs_outs.shape>: 
              {row[0]}, {str(row[1])}, {str(row[2])}, {row[3]}, {weightsTensor.shape}, {frameEmbsTensor.shape}, {capsInTensor.shape}, {tokEmbsOutsTensor.shape}""")
        # print(tensor)
        counter += 1
        if counter == limit:
            break

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
    



