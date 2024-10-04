import sys
from transformers import MarianMTModel, MarianTokenizer

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