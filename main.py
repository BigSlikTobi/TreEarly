import argparse
import queue
import sys
import threading
import json
import time
import tempfile

import spacy
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from transformers import MarianMTModel, MarianTokenizer

class Transcription: 
    @staticmethod
    def _int_or_str(text):
        """
        Helper function to convert a string to an integer if possible.
        """
        try:
            return int(text)
        except ValueError:
            return text       
   
    @staticmethod
    def construct_model_path():
        try:
            model_path = './models/transcription/en/vosk-model-small-en-us-0.15 2'
            return model_path
        except Exception as e:
            print(f"Could not construct model path: {e}")
            sys.exit(1)
       
    @staticmethod
    def initialize_transcription_model(model_path):
        """
        Initializes the Vosk model from the specified path.
        """
        try:
            model = Model(model_path)
            print(f"Vosk model loaded from {model_path}")
            return model
        except Exception as e:
            print(f"Could not load Vosk model from {model_path}: {e}")
            sys.exit(1)

    @staticmethod
    def create_audio_callback(q):
        def audio_callback(indata, frames, time_info, status):
            """Callback function to put audio data into the queue."""
            if status:
                print(status, file=sys.stderr)
            q.put(bytes(indata))
        return audio_callback

    @staticmethod
    def initialize_recognizer(model, samplerate):
        recognizer = KaldiRecognizer(model, samplerate)
        return recognizer

class Translation:
    @staticmethod
    def load_translation_model(src_lang="en", tgt_lang="de"):
        """
        Loads the MarianMT model for translation from src_lang to tgt_lang.
        """
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
        """
        Translates the given text using the provided tokenizer and model.
        """
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
        """
        Worker thread that handles translation of transcribed text.
        """
        while True:
            text = translate_queue.get()
            if text is None:
                print("Translation worker received exit signal.")
                break  # Exit signal
            translated_text = Translation.translate_text(text, tokenizer, model)
            if translated_text:
                print(f"\nTranslated ({target_language}): {translated_text}\n")
                if tts_queue:
                    tts_queue.put(translated_text)
            translate_queue.task_done()

class TextProcessing: 
    @staticmethod
    def load_spacy_model(language):
        """
        Loads the spaCy model for the specified language.
        """
        model_name = "en_core_web_sm" if language == 'en' else None
        if model_name:
            try:
                nlp = spacy.load(model_name)
                print("spaCy model loaded successfully.")
                return nlp
            except OSError:
                print(f"spaCy model '{model_name}' not found. Attempting to download...")
                try:
                    spacy.cli.download(model_name)
                    nlp = spacy.load(model_name)
                    print("spaCy model downloaded and loaded successfully.")
                    return nlp
                except Exception as e:
                    print(f"Could not download spaCy model: {e}")
                    sys.exit(1)
        else:
            print("Punctuation using spaCy is only implemented for English ('en') language.")
            return None

    @staticmethod
    def add_punctuation_spacy(transcription, nlp, min_words=4):
        """
        Adds punctuation to the transcribed text using spaCy and extracts complete sentences.
        """
        doc = nlp(transcription)
        punctuated_text = ""
        for sents in doc.sents:
            sentence_text = sents.text.strip()
            # Capitalize the first character
            if sentence_text:
                sentence_text = sentence_text[0].upper() + sentence_text[1:]
                # Add a period if missing
                if not sentence_text.endswith(('.', '?', '!')):
                    sentence_text += '.'
                punctuated_text += " " + sentence_text
        return punctuated_text.strip()

class TextToSpeech:
    @staticmethod
    def synthesize_speech(text, target_lang):
        """
        Synthesizes speech.
        """
        try:
            tts = gTTS(text, lang = target_lang)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as fp:
                tts.save(fp.name)
                audio = AudioSegment.from_file(fp.name, format="mp3")
                play(audio)
        except Exception as e:
            print(f"TTS error: {e}")

    @staticmethod
    def tts_worker(tts_queue, target_lang):
        """
        Worker thread that handles text-to-speech synthesis and playback.
        """
        while True:
            text = tts_queue.get()
            if text is None:
                print("TTS worker received exit signal.")
                break  # Exit signal
            TextToSpeech.synthesize_speech(text, target_lang)
            tts_queue.task_done()

class Utilities:
    @staticmethod
    def determine_sample_rate(args):
        if args.samplerate is None:
            try:
                device_info = sd.query_devices(args.device, "input")
                args.samplerate = int(device_info["default_samplerate"])
                print(f"Determined sample rate: {args.samplerate}")
            except Exception as e:
                print(f"Could not determine sample rate: {e}")
                sys.exit(1)
        return args.samplerate

    @staticmethod
    def initialize_translation_and_tts(args):
        translator_thread = None
        tts_thread = None
        if args.translate:
            tokenizer, translation_model = Translation.load_translation_model(src_lang="en", tgt_lang=args.target_lang)
            translate_queue = queue.Queue()
            tts_queue = queue.Queue()

            # Start the translation worker thread, passing the TTS queue
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
        return translate_queue, tts_queue

    @staticmethod
    def initialize_buffering_variables():
        buffer_text = ""
        last_audio_time = time.time()
        pause_threshold = 0.3
        last_processed_text = ""
        processed_partials = set()
        return buffer_text, last_audio_time, pause_threshold, last_processed_text, processed_partials  

    @staticmethod
    def parse_arguments():
        """
        Parses the command-line arguments.
        """
        parser = argparse.ArgumentParser(description="Realtime Speech to Text with Translation")

        parser.add_argument(
            '-l', '--language', type=str, required=True, help="Language code (eg. en)"
        )
        parser.add_argument(
            '-m', '--model', type=str, default="models", help="Path to the model directory"
        )
        parser.add_argument(
            '-d', '--device', type=Transcription._int_or_str, help="Input device (numeric ID or substring)"
        )
        parser.add_argument(
            '-r', '--samplerate', type=int, help="Sampling rate"
        )
        parser.add_argument(
            '--translate', action='store_true', help="Enable translation of transcribed text"
        )
        parser.add_argument(
            '--target_lang', type=str, default="de", help="Target language code for translation (e.g., de)"
        )
        args = parser.parse_args()
        return args

def main():

    # Initialize variables
    model_path = Transcription.construct_model_path() # Initialize model path for transcription
    args = Utilities.parse_arguments()  # Parse the command-line arguments
    model = Transcription.initialize_transcription_model(model_path) # Initialize transcription model
    nlp = TextProcessing.load_spacy_model(args.language) # initialize the punctuation model
    args.samplerate = Utilities.determine_sample_rate(args)  # Determine the appropriate sample rate
    q = queue.Queue() # Create a queue to store audio data
    audio_callback = Transcription.create_audio_callback(q) # Define the audio callback function
    recognizer = Transcription.initialize_recognizer(model, args.samplerate) # Initialize the Vosk recognizer
    buffer_text, last_audio_time, pause_threshold, last_processed_text, processed_partials = Utilities.initialize_buffering_variables() # Initialize variables for buffering audio and text
    translate_queue, tts_queue = Utilities.initialize_translation_and_tts(args) # Initialize translation model

    try:
        with sd.RawInputStream(
            samplerate=args.samplerate,
            blocksize=8000,
            device=args.device,
            dtype="int16",
            channels=1,
            callback=audio_callback,
        ):
            print("#" * 80)
            print("Press Ctrl+C to stop the transcription")
            print("#" * 80)

            while True:
                data = q.get()
                current_time = time.time()
                # print(f"Processing audio chunk at {current_time}")  # Debug print

                if recognizer.AcceptWaveform(data):
                    result = recognizer.Result()
                    # print(f"Recognizer result: {result}")  # Debug print
                    try:
                        result_dict = json.loads(result)
                        text = result_dict.get("text", "")
                        if text:
                            print(f"Final result: {text}")
                            buffer_text += " " + text
                            last_audio_time = current_time
                            last_processed_text = ""
                            processed_partials.clear() # Clear set after final result
                            #print(f"Reset last_processed_text: {last_processed_text}")  # Debug print
                    except json.JSONDecodeError as e:
                        print(f"Error parsing recognizer result: {e}")
                else:
                    partial_result = recognizer.PartialResult()
                    #print(f"Partial result: {partial_result}")  # Debug print
                    try:
                        partial_dict = json.loads(partial_result)
                        partial_text = partial_dict.get("partial", "")
                        # print(f"Partial text: {partial_text}")  # Debug print
                        # print(f"Last processed text: {last_processed_text}")  # Debug print
                        if partial_text and partial_text not in processed_partials:
                            if partial_text.startswith(last_processed_text):
                                new_text = partial_text[len(last_processed_text):].strip()
                            else:
                                new_text = partial_text.strip()
                            buffer_text += " " + new_text
                            last_processed_text = partial_text
                            processed_partials.add(partial_text)  # Add to set of processed partials
                            last_audio_time = current_time
                            print(f"Updated last_processed_text: {last_processed_text}")  # Debug print
                    except json.JSONDecodeError as e:
                        print(f"Error parsing partial result: {e}")

                # Check for pause
                if current_time - last_audio_time > pause_threshold and buffer_text:
                    #print(f"\nDetected pause. Processing buffer...\n")  # Debug print
                    if nlp:
                        punctuated_text = TextProcessing.add_punctuation_spacy(buffer_text, nlp)
                    else:
                        punctuated_text = buffer_text.strip()

                    #print(f"Transcribed: {punctuated_text}")

                    if args.translate and translate_queue:
                        translate_queue.put(punctuated_text)
                        #print("Enqueued text for translation.")

                    buffer_text = ""  # Clear the buffer after processing
                    last_processed_text = ""  # Reset the last processed text

    except KeyboardInterrupt:
        print("\nTranscription stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if args.translate and translate_queue:
            if translate_queue:
                translate_queue.put(None)  # Signal the translator thread to exit
            if tts_queue:
                tts_queue.put(None)
 
if __name__ == "__main__":
    main()