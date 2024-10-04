import tempfile

from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play

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