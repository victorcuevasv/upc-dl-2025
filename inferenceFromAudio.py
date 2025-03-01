import torchaudio
from dcase24t6.nn.hub import baseline_pipeline

sr = 44100
file = "/home/victorcuevasv/upc-dl-2025/data/CLOTHO_v2.1/clotho_audio_files/development/woodpecker, wind and dogs.wav"
waveform, sample_rate = torchaudio.load(file)

# model_name_or_path = "baseline_weights"
# model_name_or_path= "/home/victorcuevasv/upc-dl-2025/logs/train-2025.01.17-09.43.29-baseline/checkpoints/epoch_232-step_001864-mode_min-val_loss_3.3752.ckpt"
model_name_or_path="/home/victorcuevasv/upc-dl-2025/logs/train-2025.03.01-13.49.44-baseline/checkpoints/epoch_003-step_000032-mode_min-val_loss_5.7540.ckpt"
model = baseline_pipeline(model_name_or_path=model_name_or_path)
item = {"audio": waveform, "sr": sample_rate}
outputs = model(item)
candidate = outputs["candidates"][0]

print(candidate)