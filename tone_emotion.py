import torch
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU is NOT ON")

# Step 1: Benchmarking
!pip install -q opendatasets datasets transformers torch librosa soundfile scikit-learn pandas accelerate

import opendatasets as od
import glob
import os
import pandas as pd
import numpy as np
from datasets import Dataset, Audio
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
import torch
from sklearn.metrics import accuracy_score, classification_report

print("Downloading dataset...")
od.download("https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio")

print("Reading audio files...")

files = []
labels =[]

for path in glob.glob("./ravdess-emotional-speech-audio/**/*.wav", recursive=True):
    name = os.path.basename(path)
    parts = name.split("-")

    if len(parts) == 7:
        emotion = int(parts[2])
        label = -1

        # 0 = neutral/calm, 1 = happy, 2 = angry, 3 = sad
        if emotion in[1,2]:
            label = 0
        elif emotion == 3:
            label = 1
        elif emotion == 5:
            label = 2
        elif emotion == 4:
            label = 3

        if label != -1:
            files.append(path)
            labels.append(label)

df = pd.DataFrame({
    "audio": files,
    "label": labels
})

dataset = Dataset.from_pandas(df)
dataset = dataset.shuffle(seed=42).select(range(250))
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

print("Dataset ready:", len(dataset), "samples")

model_name = "superb/wav2vec2-base-superb-er"

print("Loading model...")
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
processor = AutoFeatureExtractor.from_pretrained(model_name)

def predict(batch):
    audio = []
    for x in batch["audio"]:
        audio.append(x["array"])

    inputs = processor(
        audio,
        sampling_rate=16000,
        padding=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    preds = outputs.logits.argmax(dim=-1)
    predicted_list = preds.tolist()

    result = {
        "predicted": predicted_list
    }
    return result

print("Running baseline test...")

results = dataset.map(predict, batched=True, batch_size=4)

y_true = results["label"]
y_pred = results["predicted"]
acc = accuracy_score(y_true, y_pred)

print("Baseline accuracy:", round(acc*100,2), "%")

print("\nDetailed results:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Neutral","Happy","Angry","Sad"]
))

# Step 2: Training
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
from google.colab import drive
from datasets import Dataset, Audio
import numpy as np
import gc
import torch
import os
from sklearn.metrics import accuracy_score, classification_report

print("Connecting to Google Drive...")
drive.mount('/content/drive')

torch.cuda.empty_cache()
gc.collect()

print("Preparing dataset...")
full_dataset = Dataset.from_pandas(df)
full_dataset = full_dataset.cast_column("audio", Audio(sampling_rate=16000))

split = full_dataset.train_test_split(test_size=0.2, seed=42)
train_data = split["train"]
test_data = split["test"]

model_name = "superb/wav2vec2-base-superb-er"
processor = AutoFeatureExtractor.from_pretrained(model_name)

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4,
    ignore_mismatched_sizes=True,
    label2id={"neutral": 0, "happy": 1, "angry": 2, "sad": 3},
    id2label={0: "neutral", 1: "happy", 2: "angry", 3: "sad"}
)

def prepare_train_data(batch):
    audio_list = []
    for x in batch["audio"]:
        speech = np.array(x["array"])

        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.0005, speech.shape)
            speech = speech + noise

        audio_list.append(speech)

    inputs = processor(
        audio_list,
        sampling_rate=16000,
        padding="max_length",
        max_length=80000,
        truncation=True
    )
    inputs["labels"] = batch["label"]
    return inputs

def prepare_test_data(batch):
    audio_list = [x["array"] for x in batch["audio"]]
    inputs = processor(
        audio_list,
        sampling_rate=16000,
        padding="max_length",
        max_length=80000,
        truncation=True
    )
    inputs["labels"] = batch["label"]
    return inputs

print("Processing data...")
processed_train = train_data.map(prepare_train_data, batched=True, batch_size=4, remove_columns=train_data.column_names)
processed_test = test_data.map(prepare_test_data, batched=True, batch_size=4, remove_columns=test_data.column_names)

training_args = TrainingArguments(
    output_dir="./emotion_model_pro",
    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.02,
    num_train_epochs=8,
    seed=42,

    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    fp16=True,
    gradient_checkpointing=True,

    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to="none"
)

def compute_accuracy(eval_pred):
    predictions = eval_pred.predictions
    if type(predictions) == tuple:
        predictions = predictions[0]

    preds = predictions.argmax(axis=-1)
    acc = accuracy_score(eval_pred.label_ids, preds)
    return {"accuracy": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_train,
    eval_dataset=processed_test,
    processing_class=processor,
    compute_metrics=compute_accuracy,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("\nStarting Training...")
trainer.train()

print("\nRunning final exam...")
test_results = trainer.predict(processed_test)
predictions = test_results.predictions

if type(predictions) == tuple:
    predictions = predictions[0]

y_pred = predictions.argmax(axis=-1)
y_true = test_results.label_ids

print(f"\nFINAL AUDIO ACCURACY: {round(accuracy_score(y_true, y_pred) * 100, 2)}%")
print("\nDetailed results:")
print(classification_report(y_true, y_pred, target_names=["Neutral", "Happy", "Angry", "Sad"]))

drive_path = "/content/drive/MyDrive/Final_Wav2Vec2_Tone_Final"

trainer.save_model(drive_path)
processor.save_pretrained(drive_path)
print(f"Model saved to {drive_path}")

# Step 3: Testing/Demo
from google.colab import drive
drive.mount("/content/drive", force_remount=True)

from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
import os
import subprocess
import librosa
import torch
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor

print("Loading our custom AI model...")
my_custom_model_path = "/content/drive/MyDrive/AI_MODELS_DEMO/Final_Wav2Vec2_Tone_Final"

my_audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(my_custom_model_path)
my_processor = AutoFeatureExtractor.from_pretrained(my_custom_model_path)

def record_audio_only(filename='test_audio.webm'):
    js_code = '''
    async function recordAudio() {
        const div = document.createElement('div');
        const start_button = document.createElement('button');
        const stop_button = document.createElement('button');
        const status_text = document.createElement('p');

        start_button.textContent = 'START RECORDING';
        stop_button.textContent = 'STOP & ANALYZE';
        stop_button.style.display = 'none';
        status_text.textContent = 'Ready.';

        div.appendChild(start_button);
        div.appendChild(stop_button);
        div.appendChild(status_text);
        document.body.appendChild(div);

        const stream = await navigator.mediaDevices.getUserMedia({audio: true});
        const recorder = new MediaRecorder(stream);
        let audio_data =[];

        recorder.ondataavailable = function(e) {
            audio_data.push(e.data);
        };

        start_button.onclick = function() {
            recorder.start();
            start_button.style.display = 'none';
            stop_button.style.display = 'inline';
            status_text.textContent = 'RECORDING... Speak now!';
        };

        const stop_promise = new Promise(function(resolve) {
            stop_button.onclick = function() {
                recorder.stop();
                status_text.textContent = 'Processing...';
                stream.getTracks().forEach(function(track) {
                    track.stop();
                });
                div.remove();
            };
            recorder.onstop = function() {
                const blob = new Blob(audio_data, {type: 'audio/webm'});
                const reader = new FileReader();
                reader.readAsDataURL(blob);
                reader.onload = function() {
                    resolve(reader.result);
                };
            };
        });

        return await stop_promise;
    }
    '''

    display(Javascript(js_code))

    js_result = eval_js('recordAudio()')
    encoded_audio = js_result.split(',')[1]
    decoded_audio = b64decode(encoded_audio)

    with open(filename, 'wb') as file:
        file.write(decoded_audio)

    return filename

print("Click the button below and speak into your microphone...")

recorded_file = record_audio_only()
clean_audio_file = "clean_test.wav"

try:
    subprocess.call(['ffmpeg', '-y', '-i', recorded_file, '-ac', '1', '-ar', '16000', clean_audio_file],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    speech_data, sample_rate = librosa.load(clean_audio_file, sr=16000)

    highest_volume = max(abs(speech_data))
    speech_data = speech_data / highest_volume

    model_inputs = my_processor(
        speech_data,
        sampling_rate=16000,
        padding=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        model_outputs = my_audio_model(**model_inputs)

    probabilities = torch.nn.functional.softmax(model_outputs.logits, dim=-1)

    predicted_number = torch.argmax(probabilities, dim=-1).item()

    emotion_names = {0: "Neutral", 1: "Happy", 2: "Angry", 3: "Sad"}

    print("\nEmotion probabilities:")

    for index in range(4):
        current_emotion = emotion_names[index]
        confidence_score = probabilities[0][index].item() * 100
        print(f"{current_emotion}: {confidence_score:.1f}%")

    final_emotion = emotion_names[predicted_number]

    print("FINAL PREDICTION:", final_emotion.upper())

except Exception as error:
    print("An error happened:", error)

if os.path.exists(recorded_file):
    os.remove(recorded_file)

if os.path.exists(clean_audio_file):
    os.remove(clean_audio_file)