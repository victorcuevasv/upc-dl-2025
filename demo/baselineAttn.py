from transformers import pipeline
import gradio as gr
import torchaudio
import torch
from dcase24t6.nn.hub import baseline_pipeline
import os
from dcase24t6.utils.SQLiteLogger import SQLiteLogger
import uuid


# os.environ['FFMPEG_BIN'] = r'C:\Users\victor.cuevas\workspace\gradioapp\ffmpeg-master-latest-win64-gpl-shared\bin'
# os.environ['PATH'] = f"{os.environ['FFMPEG_BIN']};{os.environ['PATH']}"

def caption_audio(filepath):
    sr = 44100
    # waveform, sample_rate = torchaudio.load("/home/ubuntu/dcase2024-task6-baseline/data/CLOTHO_v2.1/clotho_audio_files/development/woodpecker, wind and dogs.wav")
    waveform, sample_rate = torchaudio.load(filepath)
    # model = baseline_pipeline()
    model_name_or_path = "epoch_228-step_001832-mode_min-val_loss_3.3725.ckpt"
    # model_name_or_path = "baseline_weights"
    dbUUID = str(uuid.uuid4())
    dbName = f"attn-{dbUUID}.db"
    sqliteLogger = SQLiteLogger()
    sqliteLogger.dbName = dbName
    sqliteLogger.createDB()
    sqliteLogger.inference = True
    model = baseline_pipeline(model_name_or_path)
    item = {"audio": waveform, "sr": sr}
    outputs = model(item)
    print(outputs["candidates"][0])
    sqliteLogger.closeDB()
    return outputs["candidates"][0]

demo = gr.Interface(
    fn=caption_audio, inputs=gr.Audio(type="filepath"), outputs="textbox"
)
demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)
