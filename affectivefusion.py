import os
import cv2
import librosa
import moviepy.editor as mp
from transformers import pipeline
import glob
import json
import csv
from collections import Counter
import torch
import torchaudio
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Wav2Vec2Processor, Wav2Vec2Model, AutoModelForSeq2SeqLM
from deepface import DeepFace
from detoxify import Detoxify
from itertools import chain


def get_audio_from_video(video_path, output_path='temp_audio.wav'):
    clip = mp.VideoFileClip(video_path)
    clip.audio.write_audiofile(output_path)
    return output_path


def get_frames(video_path, output_dir='frames', frame_interval=30):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_folder = os.path.join(output_dir, video_name)
    os.makedirs(frame_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(frame_folder, f'frame_{frame_count}.jpg')
            cv2.imwrite(frame_path, frame)
            saved_frames.append(frame_path)
        frame_count += 1

    cap.release()
    return saved_frames, frame_folder


def get_transcription(audio_path):
    whisper = pipeline("automatic-speech-recognition", model="openai/whisper-base")
    result = whisper(audio_path)
    return result['text']


def analyze_faces(frame_folder):
    emotions = Counter()
    confidence_scores = []
    
    for file in sorted(os.listdir(frame_folder)):
        if file.endswith('.jpg'):
            path = os.path.join(frame_folder, file)
            try:
                result = DeepFace.analyze(img_path=path, actions=['emotion'], enforce_detection=False)[0]
                emotion = result['dominant_emotion']
                score = result['emotion'][emotion]
                emotions[emotion] += 1
                confidence_scores.append(score)
            except:
                continue

    if not emotions:
        return "unknown", 0.0
    
    top_emotion = emotions.most_common(1)[0][0]
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
    return top_emotion, avg_confidence


def get_consensus(text_emotion, audio_emotion, face_emotion):
    votes = []
    if text_emotion:
        votes.append(text_emotion.lower())
    if audio_emotion:
        votes.append(audio_emotion.lower())
    if face_emotion:
        votes.append(face_emotion.lower())
    
    counts = Counter(votes)
    winner = counts.most_common(1)[0]
    return winner[0], winner[1] / len(votes) * 100


def process_videos(input_folder='assets/allemotion'):
    extensions = ('*.mp4', '*.MP4', '*.mkv', '*.avi', '*.mov', '*.webm')
    video_files = list(chain.from_iterable(glob.glob(os.path.join(input_folder, ext)) for ext in extensions))

    for video_file in video_files:
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        
        audio_path = get_audio_from_video(video_file)
        frames, frame_dir = get_frames(video_file)
        text = get_transcription(audio_path)
        
        os.makedirs('outputs_1', exist_ok=True)
        with open(f'outputs_1/{video_name}_transcription.txt', 'w') as f:
            f.write(text)

        # Check for harmful content
        detox = Detoxify('original')
        detox_results = detox.predict(text)
        flags = {k: float(f"{v:.4f}") for k, v in detox_results.items() if v > 0.5}
        threat_detected = any(cat in flags for cat in ["toxicity", "threat", "insult", "hate"])
        
        with open("outputs_1/rai_results.json", "w") as f:
            json.dump({
                "input_text": text,
                "risk_flags": flags,
                "threat_detected": threat_detected
            }, f, indent=2)

        #text emotion
        text_model = "j-hartmann/emotion-english-distilroberta-base"
        tokenizer = AutoTokenizer.from_pretrained(text_model)
        model = AutoModelForSequenceClassification.from_pretrained(text_model)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(model.device)
        
        with torch.no_grad():
            logits = model(**inputs).logits
            text_emotion = model.config.id2label[logits.argmax().item()]

        with open("outputs_1/text_sentiment.json", "w") as f:
            json.dump({"label": text_emotion}, f, indent=2)

        # facial emotions
        face_emotion, face_score = analyze_faces(frame_dir)

        # audio emotion
        audio_model = "facebook/wav2vec2-base"
        labels = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]
        max_length = 16000 * 10

        processor = Wav2Vec2Processor.from_pretrained(audio_model)
        base_model = Wav2Vec2Model.from_pretrained(audio_model)
        classifier = torch.nn.Linear(base_model.config.hidden_size, len(labels))

        if os.path.exists("classifier_head.pt"):
            state_dict = torch.load("classifier_head.pt")
            if classifier.weight.shape == state_dict['weight'].shape:
                classifier.load_state_dict(state_dict)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model = base_model.to(device)
        classifier = classifier.to(device)
        base_model.eval()
        classifier.eval()

        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform.squeeze()
        if waveform.shape[0] > max_length:
            waveform = waveform[:max_length]

        inputs = processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            features = base_model(input_values).last_hidden_state
            pooled = features.mean(dim=1)
            logits = classifier(pooled)
            probs = torch.softmax(logits, dim=-1).squeeze()

        pred = torch.argmax(probs).item()
        confidence = probs[pred].item() * 100
        audio_emotion = labels[pred]

        #final consensus
        final_emotion, final_confidence = get_consensus(text_emotion, audio_emotion, face_emotion)

        with open("outputs_1/audio_sentiment.json", "w") as f:
            json.dump({"emotion": audio_emotion, "confidence": confidence}, f, indent=2)

        with open("outputs_1/fused_output.json", "w") as f:
            json.dump({
                "text": text_emotion,
                "audio": audio_emotion,
                "image": face_emotion,
                "fused_emotion": final_emotion,
                "confidence": round(final_confidence, 2)
            }, f, indent=2)

        
        with open("outputs_1/rai_results.json", "r") as f:
            rai = json.load(f)

        threat_status = "No threat detected" if not rai['threat_detected'] else "Threat detected"
        bias_status = "No bias detected" if not rai['risk_flags'] else f"Bias risk: {list(rai['risk_flags'].keys())}"

        prompt = f"""
        The video analysis results:
        - Text sentiment: {text_emotion}
        - Audio emotion: {audio_emotion}
        - Facial emotion: {face_emotion} (confidence {face_score:.2f})
        - Final emotion: {final_emotion} ({final_confidence:.2f}% confidence)
        - {threat_status}
        - {bias_status}

        Provide a brief summary of this analysis.
        """

        gen_model_name = "google/flan-t5-large"
        tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
        gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name).to(device)
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(device)
        outputs = gen_model.generate(input_ids, max_new_tokens=200)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        with open("outputs_1/final_summary_.json", "w") as f:
            json.dump({"prompt": prompt.strip(), "summary": summary.strip()}, f, indent=2)

        
        csv_file = 'outputs_1/summary_log.csv'
        serial = 1 if not os.path.isfile(csv_file) else sum(1 for _ in open(csv_file, 'r', encoding='utf-8'))

        row = [serial, video_name, text_emotion, face_emotion, audio_emotion, 
               final_emotion, threat_status, bias_status, text, summary.strip()]

        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if serial == 1:
                writer.writerow(['Serial', 'Video Name', 'Text Emotion', 'Image Emotion', 
                               'Audio Emotion', 'Final Emotion', 'Threat Status', 
                               'Bias Status', 'Input Text', 'Summary'])
            writer.writerow(row)


if __name__ == '__main__':
    process_videos()