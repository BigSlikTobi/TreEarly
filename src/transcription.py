import sys
import time
import threading
import queue
import json
import sounddevice as sd
import numpy as np
import spacy
from vosk import Model, KaldiRecognizer
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

class TranscriptionStreaming:
    def __init__(self, model_path, language, samplerate, device, translate_queue=None, tts_queue=None):
        self.model = self.initialize_transcription_model(model_path)
        self.recognizer = self.initialize_recognizer(self.model, samplerate)
        self.nlp = self.load_spacy_model(language)
        self.q = queue.Queue()
        self.audio_callback = self.create_audio_callback(self.q)
        self.samplerate = samplerate
        self.device = device
        self.buffer_text = ""
        self.last_audio_time = time.time()
        self.pause_threshold = 0.3
        self.partial_words = []
        self.final_text = ""
        self.translate_queue = translate_queue
        self.tts_queue = tts_queue
        self.processing_thread = threading.Thread(target=self.process_audio_worker, daemon=True)

    def initialize_transcription_model(self, model_path):
        try:
            model = Model(model_path)
            print(f"Vosk model loaded from {model_path}")
            return model
        except Exception as e:
            print(f"Could not load Vosk model from {model_path}: {e}")
            sys.exit(1)

    def initialize_recognizer(self, model, samplerate):
        return KaldiRecognizer(model, samplerate)

    def load_spacy_model(self, language):
        model_name = "en_core_web_sm" if language == 'en' else None
        if model_name:
            try:
                nlp = spacy.load(model_name)
                print("spaCy model loaded successfully.")
                return nlp
            except OSError:
                print(f"spaCy model '{model_name}' not found. Attempting to download...")
                spacy.cli.download(model_name)
                nlp = spacy.load(model_name)
                return nlp
        else:
            print("Punctuation using spaCy is only implemented for English ('en') language.")
            return None

    def create_audio_callback(self, q):
        def audio_callback(indata, frames, time_info, status):
            if status:
                print(status, file=sys.stderr)
            q.put(bytes(indata))
        return audio_callback

    def process_audio_worker(self):
        print("Processing worker thread started.")
        try:
            while True:
                data = self.q.get()
                if data is None:
                    print("Processing worker received exit signal.")
                    break

                current_time = time.time()

                if self.recognizer.AcceptWaveform(data):
                    result = self.recognizer.Result()
                    result_dict = json.loads(result)
                    text = result_dict.get("text", "")
                    if text:
                        print(f"Final result: {text}")
                        self.final_text += " " + text
                        self.buffer_text = ""
                        self.partial_words = []
                        self.last_audio_time = current_time
                else:
                    partial_result = self.recognizer.PartialResult()
                    partial_dict = json.loads(partial_result)
                    partial_text = partial_dict.get("partial", "")
                    if partial_text:
                        current_partial_words = partial_text.strip().split()
                        new_word_index = 0
                        for i in range(min(len(self.partial_words), len(current_partial_words))):
                            if self.partial_words[i] != current_partial_words[i]:
                                new_word_index = i
                                break
                            new_word_index = i + 1
                        new_words = current_partial_words[new_word_index:]
                        if new_words:
                            self.buffer_text += " " + " ".join(new_words)
                        self.partial_words = current_partial_words
                        self.last_audio_time = current_time

                # Check for pause
                if (current_time - self.last_audio_time > self.pause_threshold) and (self.final_text or self.buffer_text):
                    full_text = (self.final_text + " " + self.buffer_text).strip()
                    if self.nlp:
                        punctuated_text = self.add_punctuation_spacy(full_text)
                    else:
                        punctuated_text = full_text

                    print(f"Punctuated Text: {punctuated_text}")

                    # Send to translation queue if enabled
                    if self.translate_queue:
                        self.translate_queue.put(punctuated_text)

                    self.final_text = ""
                    self.buffer_text = ""
                    self.partial_words = []

                self.q.task_done()
        except Exception as e:
            print(f"An error occurred during processing: {e}")

    def add_punctuation_spacy(self, transcription):
        doc = self.nlp(transcription)
        punctuated_text = ""
        for sents in doc.sents:
            sentence_text = sents.text.strip()
            if sentence_text:
                sentence_text = sentence_text[0].upper() + sentence_text[1:]
                if not sentence_text.endswith(('.', '?', '!')):
                    sentence_text += '.'
                punctuated_text += " " + sentence_text
        return punctuated_text.strip()

    def start_processing(self):
        self.processing_thread.start()

    def stop_processing(self):
        self.q.put(None)
        self.processing_thread.join()

class TranscriptionChunking:
    def __init__(self, model_name='openai/whisper-tiny', language='english', chunk_duration=5, device=None, translate_queue=None, tts_queue=None):
        self.processor = WhisperProcessor.from_pretrained(model_name, language=language, task="transcribe")
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()
        self.language = language
        self.chunk_duration = chunk_duration  # in seconds
        self.q = queue.Queue()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.samplerate = 16000  # Whisper expects 16kHz audio
        self.audio_callback = self.create_audio_callback(self.q)
        self.audio_buffer = []
        self.processing_thread = threading.Thread(target=self.process_audio_worker, daemon=True)
        self.model.to(self.device)
        self.translate_queue = translate_queue
        self.tts_queue = tts_queue

    def create_audio_callback(self, q):
        def audio_callback(indata, frames, time_info, status):
            if status:
                print(status, file=sys.stderr)
            q.put(indata.copy())
        return audio_callback

    def process_audio_worker(self):
        print("Processing worker thread started.")
        try:
            while True:
                data = self.q.get()
                if data is None:
                    print("Processing worker received exit signal.")
                    break
                self.audio_buffer.append(data)
                total_duration = len(self.audio_buffer) * data.shape[0] / self.samplerate
                if total_duration >= self.chunk_duration:
                    audio_chunk = np.concatenate(self.audio_buffer, axis=0).flatten()
                    self.audio_buffer = []
                    print("Transcribing chunk...")
                    # Convert audio to float32 and normalize
                    if audio_chunk.dtype != np.float32:
                        audio_chunk = audio_chunk.astype(np.float32) / np.iinfo(np.int16).max
                    # Preprocess audio
                    input_features = self.processor(audio_chunk, sampling_rate=self.samplerate, return_tensors="pt").input_features
                    input_features = input_features.to(self.device)
                    # Generate transcription
                    with torch.no_grad():
                        predicted_ids = self.model.generate(input_features)
                    transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                    print(f"Transcribed text: {transcription}")
                    # Send to translation queue if enabled
                    if self.translate_queue:
                        self.translate_queue.put(transcription)
                self.q.task_done()
        except Exception as e:
            print(f"An error occurred during processing: {e}")

    def start_processing(self):
        self.processing_thread.start()

    def stop_processing(self):
        self.q.put(None)
        self.processing_thread.join()