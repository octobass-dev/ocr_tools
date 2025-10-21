import subprocess
import os
import tempfile
import time
from typing import Optional
from PIL import Image
from pdf_ocr import ScannedPDFOCR
import uroman

class OCRWithCliRomanize(ScannedPDFOCR):
    """
    Extended OCR processor that uses Android Translate app for visual verification
    """
    
    def __init__(self, input_pdf: str, output_dir: str = "ocr_output",
                 use_android_translate: bool = True):
        """
        Initialize with Termux Android Translate integration
        
        Args:
            input_pdf: Path to scanned PDF
            output_dir: Output directory
            use_android_translate: Use native Android Translate app
        """
        super(OCRWithCliRomanize, self).__init__(input_pdf, output_dir)
        self.uroman = uroman.Uroman()


    def romanize_text(self, text: str, source_language:str = "tam") -> str:
        """
        Romanize text (needed for finetuning english LLMs with other languages
        Requires: pip install uroman
        
        Args:
            text: Text to romanize
            source_language: Target language code (e.g., 'es', 'fr', 'de')
        
        Returns:
            Romanized text
        """
        try:
            return self.uroman.romanize_string(text, lcode=source_language)
        except subprocess.CalledProcessError as e:
            return f"Error during translation: {e.stderr.strip()}"
        except FileNotFoundError:
            return "Error: 'trans' command not found. Ensure translate-shell is installed."
