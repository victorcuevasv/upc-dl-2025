import sqlite3
import torch
import pickle
import codecs
import time
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
    sql = f"select epoch, batch_idx, {colName} from {tableName} order by batch_idx desc"
    rows = cursor.execute(sql)
    counter = 0
    limit = 20
    for row in rows:
        tensor = pickle.loads(codecs.decode(row[2].encode(),'base64'))
        print(f"epoch, batch_idx, tensor.shape: {str(row[0])}, {str(row[1])}, {tensor.shape}")
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
    



