# Step 1: Benchmarking
!pip install -q opendatasets transformers torch scikit-learn pandas pillow torchvision

import opendatasets as od
import glob
import os
import pandas as pd
from PIL import Image
import torch
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report

print("Step 1: Downloading colored dataset from Kaggle...")
od.download("https://www.kaggle.com/datasets/sujaykapadnis/emotion-recognition-dataset")

print("Step 2: Organizing image files...")

files = []
labels =[]

image_types = ["*.jpg", "*.jpeg", "*.png"]

for ext in image_types:
    for path in glob.glob(f"./emotion-recognition-dataset/**/{ext}", recursive=True):

        label = -1
        path_lower = path.lower()

        if "neutral" in path_lower:
            label = 0
        elif "happy" in path_lower:
            label = 1
        elif "angry" in path_lower:
            label = 2
        elif "sad" in path_lower:
            label = 3

        if label != -1:
            files.append(path)
            labels.append(label)

df = pd.DataFrame({
    "image_path": files,
    "label": labels
})

test_data = df.sample(n=250, random_state=42).reset_index(drop=True)

print(f"Dataset ready. Testing baseline on {len(test_data)} colored images.")

if torch.cuda.is_available():
    my_device = 0
    print("GPU is ON!")
else:
    my_device = -1
    print("GPU is OFF. This will be very slow!")

model_name = "dima806/facial_emotions_image_detection"

print("Step 3: Loading the base vision model...")
classifier = pipeline("image-classification", model=model_name, device=my_device)

print("Step 4: Running baseline test...")

predictions = []
actual_answers =[]

for index, row in test_data.iterrows():
    path = row["image_path"]
    correct_label = row["label"]

    try:
        img = Image.open(path).convert("RGB")
        results = classifier(img)

        best_guess = results[0]["label"]
        predicted_label = -1

        if best_guess == "neutral":
            predicted_label = 0
        elif best_guess == "happy":
            predicted_label = 1
        elif best_guess == "angry":
            predicted_label = 2
        elif best_guess == "sad":
            predicted_label = 3
        else:
            predicted_label = -1

        predictions.append(predicted_label)
        actual_answers.append(correct_label)

    except Exception as e:
        print("Could not read image:", path)

print("\nFinished checking images!")

acc = accuracy_score(actual_answers, predictions)

print("BASELINE VISION ACCURACY:", round(acc * 100, 2), "%")

print("\nDetailed results before training:")
print(classification_report(
    actual_answers,
    predictions,
    labels=[0, 1, 2, 3],
    target_names=["Neutral", "Happy", "Angry", "Sad"],
    zero_division=0
))

# Step 2: Training
import torch
import gc
import os
import pandas as pd
from PIL import Image
from datasets import Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from torchvision import transforms
from sklearn.metrics import accuracy_score, classification_report
from google.colab import drive

print("Step 1: Connecting to Google Drive and clearing memory...")
drive.mount('/content/drive')

torch.cuda.empty_cache()
gc.collect()

print("Step 2: Preparing data...")
if len(df) >= 12000:
    print("Selecting 12,000 random images...")
    df_train = df.sample(n=12000, random_state=42).reset_index(drop=True)
else:
    print("Using all available images...")
    df_train = df.sample(frac=1, random_state=42).reset_index(drop=True)

full_dataset = Dataset.from_pandas(df_train)

split = full_dataset.train_test_split(test_size=0.2, seed=42)
train_data = split["train"]
test_data = split["test"]

print("Training images:", len(train_data))
print("Testing images:", len(test_data))

model_name = "dima806/facial_emotions_image_detection"
print(f"\nStep 3: Loading base model ({model_name})...")

processor = AutoImageProcessor.from_pretrained(model_name)

model = AutoModelForImageClassification.from_pretrained(
    model_name,
    num_labels=4,
    ignore_mismatched_sizes=True,
    label2id={"neutral": 0, "happy": 1, "angry": 2, "sad": 3},
    id2label={0: "neutral", 1: "happy", 2: "angry", 3: "sad"}
)

print("\nStep 4: Setting up Image Scrambling (Data Augmentation)...")


image_scrambler = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2)
])

def transform_train_images(batch):
    pixel_values = []
    for path in batch["image_path"]:
        img = Image.open(path).convert("RGB")
        img = image_scrambler(img)

        processed_img = processor(img, return_tensors="pt")
        pixel_values.append(processed_img["pixel_values"][0])

    return {"pixel_values": pixel_values, "labels": batch["label"]}

def transform_test_images(batch):
    pixel_values =[]
    for path in batch["image_path"]:
        img = Image.open(path).convert("RGB")
        processed_img = processor(img, return_tensors="pt")
        pixel_values.append(processed_img["pixel_values"][0])

    return {"pixel_values": pixel_values, "labels": batch["label"]}

train_data.set_transform(transform_train_images)
test_data.set_transform(transform_test_images)

print("\nStep 5: Setting up strict training rules...")
training_args = TrainingArguments(
    output_dir="./temp_vision_model",
    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=1e-5,
    weight_decay=0.01,
    num_train_epochs=10,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    fp16=True,
    remove_unused_columns=False,

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

    predictions = predictions.argmax(axis=-1)
    labels = eval_pred.label_ids

    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    processing_class=processor,
    compute_metrics=compute_accuracy,

    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("\nStarting training...")
trainer.train()

print("\nTraining finished! Running final exam on the 20% unseen test data...")

test_results = trainer.predict(test_data)
predictions = test_results.predictions

if type(predictions) == tuple:
    predictions = predictions[0]

y_pred = predictions.argmax(axis=-1)
y_true = test_results.label_ids

final_accuracy = accuracy_score(y_true, y_pred)

print("FINAL ROBUST VISION ACCURACY:", round(final_accuracy * 100, 2), "%")

print("\nDetailed results:")
print(classification_report(
    y_true, y_pred, target_names=["Neutral", "Happy", "Angry", "Sad"]
))

drive_folder_path = "/content/drive/MyDrive/FINAL_ViT_Vision_Model"

print(f"\nSaving this robust model directly to: {drive_folder_path}...")
trainer.save_model(drive_folder_path)
processor.save_pretrained(drive_folder_path)

print("SUCCESS! Model is saved.")

# Step 3: Testing/Demo
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
import cv2
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from google.colab import drive
import os

print("Step 1: Connecting to Google Drive...")
drive.mount('/content/drive')

my_vision_model_path = "/content/drive/MyDrive/AI_MODELS_DEMO/FINAL_ViT_Vision_Model"

if not os.path.exists(my_vision_model_path):
    print(f"ERROR: Cannot find the folder '{my_vision_model_path}'!")
else:
    print(f"Found your Vision Model! Loading it into memory...")

    processor = AutoImageProcessor.from_pretrained(my_vision_model_path)
    model = AutoModelForImageClassification.from_pretrained(my_vision_model_path)
    print("Model loaded successfully!")

    def record_video(filename='test_video.webm'):
      js = Javascript('''
        async function recordVideo() {
          const div = document.createElement('div');
          const startBtn = document.createElement('button');
          const stopBtn = document.createElement('button');
          const video = document.createElement('video');
          const status = document.createElement('p');

          video.style.display = 'block';
          video.muted = true;
          startBtn.textContent = 'START RECORDING';
          stopBtn.textContent = 'STOP & ANALYZE';
          stopBtn.style.display = 'none';
          status.textContent = 'Status: Ready.';

          div.appendChild(startBtn);
          div.appendChild(stopBtn);
          div.appendChild(status);
          div.appendChild(video);
          document.body.appendChild(div);

          const stream = await navigator.mediaDevices.getUserMedia({video: true, audio: false});
          video.srcObject = stream;
          await video.play();

          const recorder = new MediaRecorder(stream);
          let data =[];

          recorder.ondataavailable = (e) => data.push(e.data);

          startBtn.onclick = () => {
            recorder.start();
            startBtn.style.display = 'none';
            stopBtn.style.display = 'inline';
            status.textContent = 'Status: RECORDING... Make a face!';
          };

          const stopPromise = new Promise((resolve) => {
            stopBtn.onclick = () => {
              recorder.stop();
              status.textContent = 'Processing Video... Please Wait.';
              stream.getTracks().forEach(track => track.stop());
              div.remove();
            };
            recorder.onstop = () => {
              const blob = new Blob(data, {type: 'video/webm'});
              const reader = new FileReader();
              reader.readAsDataURL(blob);
              reader.onload = () => resolve(reader.result);
            };
          });
          return await stopPromise;
        }
      ''')
      display(js)
      data = eval_js('recordVideo()')
      binary = b64decode(data.split(',')[1])
      with open(filename, 'wb') as f:
        f.write(binary)
      return filename

    print("VIDEO CAMERA IS READY!")
    print("Click START, make an emotion for 3-4 seconds, then click STOP.")

    try:
        video_file = record_video()
        print("\nVideo captured! Extracting frames for the AI...")

        cap = cv2.VideoCapture(video_file)

        total_scores = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        frame_count = 0
        frames_checked = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 5 == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)

                inputs = processor(pil_img, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)

                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

                total_scores[0] += probs[0].item()
                total_scores[1] += probs[1].item()
                total_scores[2] += probs[2].item()
                total_scores[3] += probs[3].item()

                frames_checked += 1

            frame_count += 1

        cap.release()

        if frames_checked > 0:
            print(f"Checked {frames_checked} frames in the video.")

            avg_scores = {
                0: (total_scores[0] / frames_checked),
                1: (total_scores[1] / frames_checked),
                2: (total_scores[2] / frames_checked),
                3: (total_scores[3] / frames_checked)
            }


            avg_scores[2] = avg_scores[2] * 1.5
            if avg_scores[2] > avg_scores[3]:
                avg_scores[3] = avg_scores[3] * 0.8

            if avg_scores[1] > 0.35 and avg_scores[0] > avg_scores[1]:
                avg_scores[1] = avg_scores[1] * 1.4
                avg_scores[0] = avg_scores[0] * 0.7

            grand_total = sum(avg_scores.values())
            for i in range(4):
                avg_scores[i] = (avg_scores[i] / grand_total) * 100

            predicted_id = max(avg_scores, key=avg_scores.get)
            emotion_names = {0: "Neutral", 1: "Happy", 2: "Angry", 3: "Sad"}

            print("\nAverage Video Emotion Probabilities:")
            for i in range(4):
                print(f"  - {emotion_names[i]:<10}: {avg_scores[i]:.1f}%")

            print(f" FINAL VIDEO PREDICTION: {emotion_names[predicted_id].upper()}")

        else:
            print("Video was too short to analyze!")

    except Exception as e:
        print("Error capturing or analyzing video:", e)

    if os.path.exists('test_video.webm'):
        os.remove('test_video.webm')