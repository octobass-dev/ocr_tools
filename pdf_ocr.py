import os
import json
from typing import List, Dict, Tuple
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from pdf2image import convert_from_path

class ScannedPDFOCR:
    """Apply OCR to scanned PDF files and preserve text layout"""
    
    def __init__(self, input_pdf: str, output_dir: str = "ocr_output"):
        """
        Initialize OCR processor
        
        Args:
            input_pdf: Path to input scanned PDF
            output_dir: Directory to store output files
        """
        self.input_pdf = input_pdf
        self.output_dir = output_dir
        try:
            self.reader = PdfReader(input_pdf)
            self.page_count = len(self.reader.pages)
        except Exception as e:
            print(f"Tried opening PDF: {e}. Ignore if not a pdf file.")
            self.page_count = 0
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def extract_page_as_image(self, page_num: int, dpi: float = 300) -> Image.Image:
        """
        Extract single page as PIL Image and rotate if needed
        
        Args:
            page_num: Page number (0-indexed)
            dpi: Resolution in DPI
        
        Returns:
            PIL Image object (rotated to horizontal if needed)
        """
        # Convert PDF page to image using pdf2image
        images = convert_from_path(self.input_pdf, first_page=page_num + 1, 
                                   last_page=page_num + 1, dpi=int(dpi))
        image = images[0]
        
        # Check if image needs rotation to horizontal
        #image = self._rotate_to_horizontal(image)
        
        return image
    
    def _rotate_to_horizontal(self, image: Image.Image) -> Image.Image:
        """
        Rotate image to horizontal orientation if needed
        
        Args:
            image: PIL Image object
        
        Returns:
            Rotated image in horizontal orientation
        """
        width, height = image.size
        
        if height > width:
            image = image.rotate(-90, expand=True)
        
        return image
    
    def extract_text_with_bbox(self, image: Image.Image) -> List[Dict]:
        """
        Extract text from image with bounding boxes
        
        Args:
            image: PIL Image object
        
        Returns:
            List of dicts with text, bbox, and confidence
        """
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        results = []
        n_boxes = len(data['level'])
        
        for i in range(n_boxes):
            # Only include detected text (confidence > 0)
            if int(data['conf'][i]) > 0:
                #text = data['text'][i].strip()
                if text:
                    results.append({
                        'text': text,
                        'x': int(data['left'][i]),
                        'y': int(data['top'][i]),
                        'width': int(data['width'][i]),
                        'height': int(data['height'][i]),
                        'confidence': int(data['conf'][i]),
                        'block_num': int(data['block_num'][i]),
                        'line_num': int(data['line_num'][i]),
                        'word_num': int(data['word_num'][i])
                    })
        
        return results
    
    def translate_text(self, text: str, target_language: str = "es", source_language:str = "auto") -> str:
        """
        Translate text to target language
        Requires: pip install google-cloud-translate
        
        Args:
            text: Text to translate
            target_language: Target language code (e.g., 'es', 'fr', 'de')
        
        Returns:
            Translated text
        """
        try:
            from google.cloud import translate_v2
            
            client = translate_v2.Client()
            if source_language is None or source_language=="auto":
                result = client.translate_text(text, target_language_code=target_language)
            else:
                result = client.translate_text(text, target_language_code=target_language, source_language_code=source_language)
            return result['translatedText']
        
        except ImportError:
            print("Install Google Cloud Translation: pip install google-cloud-translate")
            return text
        except Exception as e:
            print(f"Translation error: {e}")
            return text
    
    def overlay_text_on_image(self, image: Image.Image, 
                             text_data: List[Dict],
                             font_path: str = 'fonts/Poppins/Poppins-Regular.ttf',
                             font_size: int = 30,
                             text_color: Tuple = (0, 0, 0),
                             bg_color: Tuple = (255, 255, 255),
                             transparency: float = 0.8) -> Image.Image:
        """
        Overlay OCR text on original image
        
        Args:
            image: Original image
            text_data: List of text with bounding boxes
            font_size: Font size for text
            text_color: RGB color for text
            bg_color: Background color
            transparency: Transparency level (0-1)
        
        Returns:
            Image with overlaid text
        """
        result = image.copy()
        #print(font_path)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            print(f"Error loading font '{font_path}': {e}. Using default font.")
            font = ImageFont.load_default(font_size)
        draw = ImageDraw.Draw(result)
        
        for item in text_data:
            x, y = item['x'], item['y']
            w, h = item['width'], item['height']
            text = item['text']
            
            # Draw semi-transparent background
            overlay = Image.new('RGBA', result.size, (255, 255, 255, 0))
            overlay_draw = ImageDraw.Draw(overlay)

            overlay_draw.rectangle(
                [(x, y), (x + w, y + h)],
                fill=(*bg_color, int(255 * transparency))
            )
            
            # Draw text
            if h > w:
                text_overlay = Image.new('RGBA', (h,w), (255, 255, 255, 0))
                text_overlay_draw = ImageDraw.Draw(text_overlay)
                text_overlay_draw.text((0, 0), text, font=font, fill=text_color)
                text_overlay = text_overlay.rotate(90, expand=True)
            else:
                text_overlay = Image.new('RGBA', (w,h), (255, 255, 255, 0))
                text_overlay_draw = ImageDraw.Draw(text_overlay)
                text_overlay_draw.text((0, 0), text, font=font, fill=text_color)

            overlay.paste(text_overlay, (x,y), text_overlay)
            result = Image.alpha_composite(result.convert('RGBA'), overlay).convert('RGB')
        return result
    
    def process_page(self, page_num: int,
                    romanize:bool = False,
                    translate: bool = False,
                    target_language: str = "en",
                    source_language:str = 'en',
                    save_ocr_json: bool = False,
                    font_size:int = 60,
                    transparency:float = 0.8,
                    ignore_existing:bool = False) -> Dict:
        """
        Process single page with OCR
        
        Args:
            page_num: Page number (0-indexed)
            translate: Whether to translate extracted text
            target_language: Target language code
            save_ocr_json: Save OCR data as JSON
        
        Returns:
            Dict with page info and results
        """
        print(f"Processing page {page_num + 1}/{self.page_count}...")
        
        # Extract page as image
        image = self.extract_page_as_image(page_num)
        
        # Extract text with bounding boxes
        text_data = self.extract_text_with_bbox(image)
        
        # Translate if requested
        if translate:
            for item in text_data:
                item['original_text'] = item['text']
                item['text'] = self.translate_text(item['text'], target_language, source_language)
        
        # Overlay text on image
        ocr_image = self.overlay_text_on_image(image, text_data, font_size=font_size, transparency=transparency)
        
        # Save OCR image
        ocr_path = os.path.join(self.output_dir, f"page_{page_num + 1:04d}_ocr.png")
        ocr_image.save(ocr_path)
        
        # Save OCR data as JSON
        if save_ocr_json:
            json_path = os.path.join(self.output_dir, f"page_{page_num + 1:04d}_ocr.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(text_data, f, ensure_ascii=False, indent=2)
        
        return {
            'page_num': page_num + 1,
            'text_count': len(text_data),
            'ocr_image_path': ocr_path,
            'ocr_json_path': json_path if save_ocr_json else None,
            'text_data': text_data
        }
    
    def process_all_pages(self, 
                          romanize:bool = False,
                         translate: bool = False,
                         target_language: str = "es",
                         source_language: str = None,
                         start_page: int = 0,
                         end_page: int = None,
                         font_size = 60,
                         transparency = 0.8,
                         ignore_existing:bool = True,
                         **kwargs) -> List[Dict]:
        """
        Process all pages in PDF
        
        Args:
            translate: Whether to translate text
            target_language: Target language code
            start_page: Starting page number (0-indexed)
            end_page: Ending page number (0-indexed, inclusive)
        
        Returns:
            List of processing results for each page
        """
        if end_page is None:
            end_page = self.page_count - 1
        
        results = []
        for page_num in range(start_page, min(end_page + 1, self.page_count)):
            result = self.process_page(page_num, romanize, translate, target_language, source_language,
                                       font_size=font_size, transparency=transparency,ignore_existing=ignore_existing)
            results.append(result)
        
        return results
    
    def create_searchable_pdf(self, output_pdf: str = None,
                            romanize:bool = False,
                            translate: bool = False,
                            target_language: str = "es",
                            source_language:str = None,
                            font_size = 60,
                            transparency = 0.8,
                            ignore_existing=True,
                            **kwargs) -> str:
        """
        Create searchable PDF with text overlay
        Requires: pip install pypdf
        
        Args:
            output_pdf: Output PDF path
            translate: Whether to translate text
            target_language: Target language code
        
        Returns:
            Path to output PDF
        """
        try:
            from pypdf import PdfWriter, PdfReader
            from reportlab.pdfgen import canvas
            import io
        except ImportError:
            print("Install required: pip install pypdf reportlab")
            return None
        
        if output_pdf is None:
            output_pdf = os.path.join(self.output_dir, "searchable_output.pdf")
        
        # Process all pages
        results = self.process_all_pages(romanize, translate, target_language, source_language, 
                                         font_size=font_size, transparency=transparency, ignore_existing=ignore_existing)
        
        writer = PdfWriter()
        
        print("Creating searchable PDF...")
        for result in results:
            page_num = result['page_num'] - 1
            original_page = self.reader.pages[page_num]
            box = original_page.mediabox
            
            # Create text overlay
            text_overlay = io.BytesIO()
            c = canvas.Canvas(text_overlay, pagesize=(box.width, box.height))
            if result['ocr_image_path'] is not None and os.path.exists(result['ocr_image_path']):
                print(result['ocr_image_path'])
                c.drawImage(result['ocr_image_path'], x=0, y=0, width=box.width, height=box.height)

            c.setFont("Helvetica", font_size)
            for item in result['text_data']:
                c.drawString(item['x'], item['y'], item['text'])
            c.save()
            text_overlay.seek(0)

            image_pdf_reader = PdfReader(io.BytesIO(text_overlay.getvalue()))
            new_page = image_pdf_reader.pages[0]
            #original_page.merge_page(new_page, over=True)

            writer.add_page(new_page)
        
        with open(output_pdf, 'wb') as f:
            writer.write(f)
        
        print(f"Searchable PDF saved to: {output_pdf}")
        return output_pdf
    
    def save_summary(self, results: List[Dict], filename: str = "ocr_summary.json"):
        """
        Save processing summary
        
        Args:
            results: List of processing results
            filename: Output filename
        """
        summary = {
            'input_pdf': self.input_pdf,
            'total_pages': self.page_count,
            'processed_pages': len(results),
            'total_text_items': sum(r['text_count'] for r in results),
            'pages': results
        }
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"Summary saved to: {filepath}")
    
    def __del__(self):
        """Clean up resources"""
        pass

