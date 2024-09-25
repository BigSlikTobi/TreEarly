import argparse
import queue
import sys
import threading
import time
import tempfile

import sounddevice as sd
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from transformers import MarianMTModel, MarianTokenizer

from src.transcription import TranscriptionStreaming, TranscriptionChunking

class Translation:
    @staticmethod
    def load_translation_model(src_lang="en", tgt_lang="de"):
        model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            print(f"Translation model {model_name} loaded successfully.")
            return tokenizer, model
        except Exception as e:
            print(f"Could not load translation model {model_name}: {e}")
            sys.exit(1)

    @staticmethod
    def translate_text(text, tokenizer, model):
        try:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            translated_tokens = model.generate(**inputs)
            tgt_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            return tgt_text
        except Exception as e:
            print(f"Translation error: {e}")
            return ""

    @staticmethod
    def translation_worker(translate_queue, tokenizer, model, target_language, tts_queue=None):
        while True:
            text = translate_queue.get()
            if text is None:
                print("Translation worker received exit signal.")
                break
            translated_text = Translation.translate_text(text, tokenizer, model)
            if translated_text:
                print(f"\nTranslated ({target_language}): {translated_text}\n")
                if tts_queue:
                    tts_queue.put(translated_text)
            translate_queue.task_done()

class TextToSpeech:
    @staticmethod
    def synthesize_speech(text, target_lang):
        try:
            tts = gTTS(text, lang=target_lang)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as fp:
                tts.save(fp.name)
                audio = AudioSegment.from_file(fp.name, format="mp3")
                play(audio)
        except Exception as e:
            print(f"TTS error: {e}")

    @staticmethod
    def tts_worker(tts_queue, target_lang):
        while True:
            text = tts_queue.get()
            if text is None:
                print("TTS worker received exit signal.")
                break
            TextToSpeech.synthesize_speech(text, target_lang)
            tts_queue.task_done()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Realtime Speech to Text with Translation")
    parser.add_argument('-l', '--language', type=str, required=True, help="Language code (e.g., en)")
    parser.add_argument('--model_path', type=str, default="models", help="Path to the model directory")
    parser.add_argument('--device', type=str, help="Input device (numeric ID or substring)")
    parser.add_argument('--samplerate', type=int, help="Sampling rate")
    parser.add_argument('--streaming', action='store_true', help="Use streaming transcription (Vosk)")
    parser.add_argument('--translate', action='store_true', help="Enable translation of transcribed text")
    parser.add_argument('--target_lang', type=str, default="de", help="Target language code for translation (e.g., de)")
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    # Initialize translation and TTS queues and threads if needed
    if args.translate:
        tokenizer, translation_model = Translation.load_translation_model(src_lang=args.language, tgt_lang=args.target_lang)
        translate_queue = queue.Queue()
        tts_queue = queue.Queue()

        # Start the translation worker thread
        translator_thread = threading.Thread(
            target=Translation.translation_worker,
            args=(translate_queue, tokenizer, translation_model, args.target_lang, tts_queue),
            daemon=True
        )
        translator_thread.start()
        print("Translation worker thread started.")

        # Start the TTS worker thread
        tts_thread = threading.Thread(
            target=TextToSpeech.tts_worker,
            args=(tts_queue, args.target_lang),
            daemon=True
        )
        tts_thread.start()
        print("TTS worker thread started.")
    else:
        translate_queue = None
        tts_queue = None

    if args.streaming:
        # Initialize Vosk + spaCy transcription
        if args.samplerate is None:
            device_info = sd.query_devices(args.device, "input")
            args.samplerate = int(device_info["default_samplerate"])
        transcription = TranscriptionStreaming(
            model_path=args.model_path,
            language=args.language,
            samplerate=args.samplerate,
            device=args.device,
            translate_queue=translate_queue,
            tts_queue=tts_queue,
        )
        transcription.start_processing()
        try:
            with sd.RawInputStream(
                samplerate=args.samplerate,
                blocksize=8000,
                device=args.device,
                dtype="int16",
                channels=1,
                callback=transcription.audio_callback,
            ):
                print("Audio stream started.")
                print("Press Ctrl+C to stop.")
                print("Listening...")
                while True:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping audio stream...")
        finally:
            transcription.stop_processing()
    else:
        # Initialize Whisper transcription
        transcription = TranscriptionChunking(
            model_name='openai/whisper-tiny',
            language=args.language,
            chunk_duration=5,
            device=args.device,
            translate_queue=translate_queue,
            tts_queue=tts_queue,
        )
        transcription.start_processing()
        try:
            with sd.InputStream(
                samplerate=transcription.samplerate,
                blocksize=0,
                device=args.device,
                channels=1,
                callback=transcription.audio_callback,
            ):
                print("Audio stream started.")
                print("Press Ctrl+C to stop.")
                print("Listening...")
                while True:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping audio stream...")
        finally:
            transcription.stop_processing()

    # Clean up translation and TTS threads
    if args.translate:
        if translate_queue:
            translate_queue.put(None)
        if tts_queue:
            tts_queue.put(None)

if __name__ == "__main__":
    main()