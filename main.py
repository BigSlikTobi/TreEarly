import argparse
import queue
import sys
import threading
import time
import sounddevice as sd

from src.transcription import TranscriptionStreaming, TranscriptionChunking
from src.translation import Translation
from src.tts import TextToSpeech

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
    if args.language == "en": 
        modelName = "openai/whisper-tiny.en"
    elif args.language == "de":
        modelName = "daveni/whisper-tiny-commonvoice_v11-de"
    elif args.language == "es":
        modelName = "zuazo/whisper-tiny-es"
    else: 
        modelName = "openai/whisper-tiny"

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
        try: 
            translator_thread.start()
            print("Translation worker thread started.")
        except Exception as e:
            print(f"Could not load translation model {e}")
            sys.exit(1)

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
            model_name = modelName,
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