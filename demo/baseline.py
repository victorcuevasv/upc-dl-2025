from transformers import pipeline
import gradio as gr
import torchaudio
import torch
from dcase24t6.nn.hub import baseline_pipeline
import os


# os.environ['FFMPEG_BIN'] = r'C:\Users\victor.cuevas\workspace\gradioapp\ffmpeg-master-latest-win64-gpl-shared\bin'
# os.environ['PATH'] = f"{os.environ['FFMPEG_BIN']};{os.environ['PATH']}"

def caption_audio(filepath):
    sr = 44100
    # waveform, sample_rate = torchaudio.load("/home/ubuntu/dcase2024-task6-baseline/data/CLOTHO_v2.1/clotho_audio_files/development/woodpecker, wind and dogs.wav")
    waveform, sample_rate = torchaudio.load(filepath)
    # model = baseline_pipeline()
    checkpoint_file = "/home/victorcuevasv/dcase2024-task6-baseline-original/logs/train-2025.01.17-09.43.29-baseline/checkpoints/epoch_232-step_001864-mode_min-val_loss_3.3752.ckpt"
    # checkpoint_file = "/home/victorcuevasv/dcase2024-task6-baseline-original/logs/train-2025.01.17-09.43.29-baseline/checkpoints/fake.ckpt"
    model = baseline_pipeline(checkpoint_file)
    item = {"audio": waveform, "sr": sr}
    outputs = model(item)
    print(outputs["candidates"][0])
    return outputs["candidates"][0]

demo = gr.Interface(
    fn=caption_audio, inputs=gr.Audio(type="filepath"), outputs="textbox"
)
demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)
