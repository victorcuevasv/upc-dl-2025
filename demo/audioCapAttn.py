from transformers import pipeline
import gradio as gr
import torchaudio
import torch
import uuid
from dcase24t6.nn.hub import baseline_pipeline
from dcase24t6.utils.SQLiteLogger import SQLiteLogger
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sqlite3
import pickle
import codecs
import numpy as np

def createDB(dbName, tableName, colName):
    conn = sqlite3.connect(dbName)
    cursor = conn.cursor()
    sql = f"create table if not exists {tableName} ({colName} string)"
    cursor.execute(sql)
    cursor.close()
    return conn

def countTuples(conn, tableName):
    cursor = conn.cursor()
    rowsCount = cursor.execute(f"select count(*) from {tableName}")
    nrows = next(rowsCount)[0]
    return nrows

def outputTuples(conn, tableName):
    cursor = conn.cursor()
    rows = cursor.execute(f"select * from output")
    for row in rows:
        print(row)

def getWeightsTensor(nTokens):
    global dbName
    global weightsTensor
    tableName = "weights"
    colName = "weights"
    conn = createDB(dbName, tableName, colName)
    cursor = conn.cursor()
    sql = f"select mode, epoch, batch_idx, decoding, {colName}, frame_embs, caps_in, tok_embs_outs from {tableName} order by batch_idx desc"
    rows = cursor.execute(sql)
    counter = 1
    for row in rows:
        weightsTensor = pickle.loads(codecs.decode(row[4].encode(),'base64'))
        frameEmbsTensor = pickle.loads(codecs.decode(row[5].encode(),'base64'))
        capsInTensor = pickle.loads(codecs.decode(row[6].encode(),'base64'))
        tokEmbsOutsTensor = pickle.loads(codecs.decode(row[7].encode(),'base64'))
        # Average the heads
        weightsTensor = weightsTensor.mean(dim=0, keepdim=True)
        if counter == nTokens:
            print(f"""<mode, epoch, batch_idx, decoding, weights.shape, frame_embs.shape, caps_in.shape tok_embs_outs.shape>: 
              {row[0]}, {str(row[1])}, {str(row[2])}, {row[3]}, {weightsTensor.shape}, {frameEmbsTensor.shape}, {capsInTensor.shape}, {tokEmbsOutsTensor.shape}""")
            break
        counter += 1
    cursor.close()
    conn.close()
    return weightsTensor.detach().cpu()

def caption_audio(filepath):
    global dbName
    global outputs
    global maxIndex
    global weightsTensor
    sr = 44100
    # waveform, sample_rate = torchaudio.load("/home/ubuntu/dcase2024-task6-baseline/data/CLOTHO_v2.1/clotho_audio_files/development/woodpecker, wind and dogs.wav")
    waveform, sample_rate = torchaudio.load(filepath)
    # model = baseline_pipeline()
    model_name_or_path = "epoch_228-step_001832-mode_min-val_loss_3.3725.ckpt"
    # model_name_or_path = "baseline_weights"
    dbUUID = str(uuid.uuid4())
    dbName = f"attn-{dbUUID}.db"
    print(f"dbName: {dbName}")
    sqliteLogger = SQLiteLogger()
    sqliteLogger.dbName = dbName
    sqliteLogger.createDB()
    sqliteLogger.inference = True
    model = baseline_pipeline(model_name_or_path)
    item = {"audio": waveform, "sr": sr}
    outputs = model(item)
    print(f"outputs: {outputs}")
    print(outputs["candidates"][0])
    sqliteLogger.closeDB()
    maxIndex = getMaxIndex(outputs)
    print(f"maxIndex: {maxIndex}")
    # Consider the existence of the <eos> token
    beamCandsList = outputs["beam_candidates"][0]
    topBeamCand = beamCandsList[maxIndex]
    nTokens = len(getWordList(topBeamCand)) + 1
    weightsTensor = getWeightsTensor(nTokens)
    return topBeamCand

def getMaxIndex(outputs):
    logProbs = outputs['beam_log_probs']
    maxIndex = torch.argmax(logProbs)
    return maxIndex

def getWordList(sentence):
    sentence = sentence.replace(',', '')
    return sentence.split()

def plot_attention(topBeamCand):
    """ Plots the attention map
    Args:
        att (torch.FloatTensor): Attention map (T_q x T_k)
        queries (List[str]): Query Tensor
        keys (List[str]): Key Tensor
    """
    global weightsTensor
    global maxIndex
    xtitle="Keys"
    ytitle="Queries"
    sns.set(rc={'figure.figsize':(20, 8)})
    # Consider the case in which a candidate dropped in position along the beam
    beamSize = weightsTensor.size(dim=1)
    ax = sns.heatmap(
        weightsTensor[0][min(maxIndex, beamSize-1)],
        cmap="coolwarm",)
        # linewidth=0.5,
        # xticklabels=keys,
        # yticklabels=queries,
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    plt.savefig("corr.png")
    plt.clf()
    plots = ["corr.png"]
    return plots

baseDir="/home/usuario/workspace/PosgradoDeepLearningAI/upc-dl-2025"
dbName = None
outputs = None
weightsTensor = None
maxIndex = 0

with gr.Blocks(css=".preview { width: 100%; }") as demo:
    audio = gr.Audio(type="filepath")
    textBox = gr.Textbox()
    gallery = gr.Gallery(label="Attention weights")
    textBox.change(fn=plot_attention, inputs=textBox, outputs=gallery,)
    gr.Interface(
        fn=caption_audio, inputs=audio, outputs=[textBox],
    )

demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)
