from deep_translator import GoogleTranslator, MyMemoryTranslator
import pykakasi
from gtts import gTTS
import io

def generate_audio(text, lang='ja'):
    """Generates TTS audio and returns it as a bytes-like object."""
    try:
        if not text:
            return None
        tts = gTTS(text=text, lang=lang)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None


# Initialize pykakasi once
kks = pykakasi.kakasi()

def get_kakasi_details(japanese_text):
    """
    Takes Japanese text (Kanji/Kana) and returns a tuple of (original, hiragana, romaji).
    """
    result = kks.convert(japanese_text)
    
    kanji = ""
    kana = ""
    romaji = ""
    
    for item in result:
        kanji += item['orig']
        kana += item['hira']
        
        # Add space between words for Romaji, if not empty
        if item['hepburn']:
            romaji += item['hepburn'] + " "
            
    # Remove the trailing space from Romaji
    romaji = romaji.strip()
        
    return kanji, kana, romaji

def safe_translate(text, source, target):
    """Helper function to try GoogleTranslator first, then fallback to MyMemoryTranslator."""
    try:
        translated = GoogleTranslator(source=source, target=target).translate(text)
        if not translated:
            raise ValueError("Empty translation from Google")
        return translated
    except Exception as e:
        print(f"GoogleTranslator failed: {e}. Falling back to MyMemoryTranslator...")
        # MyMemoryTranslator uses full language names or specific codes
        mm_source = 'english' if source == 'en' else ('japanese' if source == 'ja' else source)
        mm_target = 'japanese' if target == 'ja' else ('english' if target == 'en' else target)
        return MyMemoryTranslator(source=mm_source, target=mm_target).translate(text)

def process_input(text, input_type):
    """
    Processes the input text and returns a dictionary with English, Kanji, Kana, and Romaji.
    
    text: The input text (English or Romaji)
    input_type: "english" or "romaji" (or "japanese")
    """
    result_dict = {
        "english": "",
        "kanji": "",
        "kana": "",
        "romaji": ""
    }
    
    try:
        if input_type.lower() == "english":
            # 1. Translate English to Japanese
            japanese_text = safe_translate(text, source='en', target='ja')
            
            # 2. Extract components
            kanji, kana, romaji = get_kakasi_details(japanese_text)
            
            result_dict["english"] = text
            result_dict["kanji"] = kanji
            result_dict["kana"] = kana
            result_dict["romaji"] = romaji
            
        elif input_type.lower() in ["romaji", "japanese"]:
            # If the user inputs Romaji, we can use Google Translator from 'ja' to 'en'
            # because Google Translator natively handles Romaji text when set to Japanese.
            
            # 1. Translate the Romaji/Japanese to English
            english_text = safe_translate(text, source='ja', target='en')
            
            # 2. Translate English back to Japanese to get standard Kanji/Kana representation.
            # This helps standardize Romaji input into properly written Japanese.
            japanese_text = safe_translate(english_text, source='en', target='ja')
            
            # 3. Extract components
            kanji, kana, romaji = get_kakasi_details(japanese_text)
            
            result_dict["english"] = english_text
            result_dict["kanji"] = kanji
            result_dict["kana"] = kana
            result_dict["romaji"] = romaji
            
    except Exception as e:
        print(f"Error processing text: {e}")
        
    return result_dict
