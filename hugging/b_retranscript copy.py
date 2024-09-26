from transformers import pipeline

transcriber = pipeline(task="automatic-speech-recognition",
                       model="openai/whisper-small")
result = transcriber(
    "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")

print(result)