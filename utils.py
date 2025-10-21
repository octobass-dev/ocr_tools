from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

font_mapping = {
    'ta':'./fonts/Hind_Madurai/HindMadurai-Regular.ttf',
    'en':'./fonts/Poppins/Poppins-Regular.ttf',
    'cn':'./fonts/CactusClassicalSerif/CactusClassicalSerif-Regular.ttf',
    'hi':'./fonts/Sahitya/Sahitya-Regular.ttf',
}

indic_translate_mapping = {
    'ta':"tam_Taml",
    'hi':"hin_Deva", 
}

indic_transliterate_mapping = {
    'hi':sanscript.DEVANAGARI,
    'ta':sanscript.TAMIL,
    'bn':sanscript.BENGALI,
    'gu':sanscript.GUJARATI,
    'kn':sanscript.KANNADA,
    'ml':sanscript.MALAYALAM,
    'or':sanscript.ORIYA,
    'pa':sanscript.GURMUKHI,
    'te':sanscript.TELUGU,
    'en':sanscript.ITRANS,
}

def get_font_path(language_code):
    """Retrieve the font path for a given language code."""
    return font_mapping.get(language_code, './fonts/Poppins/Poppins-Regular.ttf')


def indic_transliterate(text, source_language='ta', target_language='hi'):
    """Transliterate text between Indic scripts.

    Args:
        text (str): Text to be transliterated.
        source_language (str): Source language code.
        target_language (str): Target language code.

    Returns:
        str: Transliterated text.
    """

    src_script = indic_transliterate_mapping[source_language]
    tgt_script = indic_transliterate_mapping[target_language]

    transliterated_text = transliterate(text, src_script, tgt_script)
    return transliterated_text

def local_libre_translate(text, source_language='auto', target_language='en'):
    """Translate text using LibreTranslate API.
    Needs a local server instance of LibreTranslate running.

    Args:
        text (str): Text to be translated.
        source_language (str): Source language code.
        target_language (str): Target language code.

    Returns:
        str: Translated text.
    """
    import requests
    import json

    libretranslate_url = "http://localhost:5000/translate"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "q": text,
        "source": source_language,
        "target": target_language,
        "format": "text"
    }
    response = requests.post(libretranslate_url, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        translated_text = response.json().get('translatedText', '')
        return translated_text
    else:
        raise Exception(f"Translation failed with status code {response.status_code}: {response.text}")

def load_translation_model(name="ai4bharat/indictrans2-indic-indic-dist-320M"):
    """Load the translation model and tokenizer.

    Args:
        name (str): Model name.
    """
    
    model_name = "ai4bharat/indictrans2-indic-indic-dist-320M"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        torch_dtype=torch.float16, # performance might slightly vary for bfloat16
        attn_implementation="flash_attention_2"
    ).to(DEVICE)
    return model, tokenizer, DEVICE

def indic_translate(text, model, tokenizer, source_language='auto', target_language='en'):
    """Translate text using HuggingFace transformers.

    Args:
        text (str): Text to be translated.
        source_language (str): Source language code.
        target_language (str): Target language code.

    Returns:
        str: Translated text.
    """

    src_lang, tgt_lang = indic_mapping.get(source_language, source_language), indic_mapping.get(target_language, target_language)  
    print(src_lang, tgt_lang)     
    print(text)
    ip = IndicProcessor(inference=True)

    batch = ip.preprocess_batch(
        text,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )

    # Tokenize the sentences and generate input encodings
    inputs = tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(DEVICE)

    # Generate translations using the model
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )

    # Decode the generated tokens into text
    generated_tokens = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    # Postprocess the translations, including entity replacement
    translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)
    return translations
