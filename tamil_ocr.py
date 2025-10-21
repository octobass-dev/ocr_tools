from ocr_romanize import OCRWithCliRomanize
from ocr_tamil.ocr import OCR
from ocr_tamil.craft_text_detector import (
    get_prediction,
    export_detected_regions
)

import requests
from PIL import Image
from typing import List, Dict, Tuple
import torch
from pathlib import Path
import cv2
import os
import json
import numpy as np
from googletrans import Translator
from utils import *

class TamilOCR(OCR):
    def __init__(self,  assume_straight_page=True, **kwargs):
        super(TamilOCR, self).__init__(**kwargs)
    
    def craft_detect(self, image,regions = True, **kwargs):
        """Text detection predict

        Args:
            image (numpy array): image numpy array

        Returns:
            list: list of cropped numpy arrays for text detected
            list: Bbox informations
        """
        size = max(image.shape[0],image.shape[1],640)

        # Reshaping to the nearest size
        size = min(size,2560)
        
        # perform prediction
        prediction_result = get_prediction(
            image=image,
            craft_net=self.craft_net,
            text_threshold=self.text_threshold,
            link_threshold=self.link_threshold,
            low_text=self.low_text,
            cuda=self.gpu,
            long_size=size,
            poly = False,
            half=self.fp16
        )

        # print(prediction_result)

        new_bbox = []

        for bb in prediction_result:
            xs = bb[:,0]
            ys = bb[:,1]

            min_x,max_x = min(xs),max(xs)
            min_y,max_y = min(ys),max(ys)
            x,y,w,h = min_x,min_y,max_x-min_x, max_y-min_y
            if w>0 and h>0:
                new_bbox.append([x,y,w,h])
        ordered_new_bbox = []
        if len(new_bbox):
            ordered_new_bbox,line_info = self.sort_bboxes(new_bbox)

            updated_prediction_result = []
            for ordered_bbox in ordered_new_bbox:
                index_val = new_bbox.index(ordered_bbox)
                updated_prediction_result.append(prediction_result[index_val])

            # export detected text regions
            exported_file_paths = export_detected_regions(
                image=image,
                regions=updated_prediction_result ,#["boxes"],
                # output_dir=self.output_dir,
                #method=self.method
            )

            updated_prediction_result = [(i,line) for i,line in zip(updated_prediction_result,line_info)]

        else:
            updated_prediction_result = []
            exported_file_paths = []

        torch.cuda.empty_cache()

        if regions:
            return exported_file_paths,updated_prediction_result, ordered_new_bbox
        else: 
            return exported_file_paths,updated_prediction_result    

class TamilPDFOCR(OCRWithCliRomanize):
    def __init__(self, input_pdf: str, output_dir: str = "ocr_output"):
        super(TamilPDFOCR, self).__init__(input_pdf, output_dir)
        self.ocr = TamilOCR(detect=True, lang=['tamil'], assume_straight_page=True)
        self.translator = Translator()
        
    def extract_text_with_bbox(self, image_file:str) -> list[dict]:
        """
        Extract text from image with bounding boxes
        
        Args:
            image: PIL Image object
        
        Returns:
            List of dicts with text, bbox, and confidence
        """
        image =  self.ocr.read_image_input(image_file)
        exported_regions,updated_prediction_result, bbox = self.ocr.craft_detect(image, regions=True)
        text_list,conf = self.ocr.text_recognize_batch(exported_regions)
        #text_list = [self.ocr.output_formatter(inter_text_list,conf,updated_prediction_result)]

        results = []
        #text_list, conf, bbox = self.ocr.predict(image_file, regions=True)
        nboxes = len(bbox)
        for i in range(nboxes): 
            text = text_list[i]
            if text is not None and len(text)>0:
                results.append({
                        'text': text,
                        'x': int(bbox[i][0]),
                        'y': int(bbox[i][1]),
                        'width': int(bbox[i][2]),
                        'height': int(bbox[i][3]),
                        'confidence': conf[i],
                    })
        return results
    
    def page_image_process(self, image, translate: bool = False,
                    romanize:bool = False, target_language: str = "en",
                    source_language:str = 'en',
                    font_size:int = 60,
                    transparency:float = 0.8) -> Tuple[Image.Image, List[Dict]]:
            # Extract text with bounding boxes  
        np_image = np.asarray(image)
        cv2_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        text_data = self.extract_text_with_bbox(cv2_image)
        
        # Translate if requested
        font_path = get_font_path(target_language)
        if translate:
            model, tokenizer, DEVICE = load_translation_model("ai4bharat/indictrans2-indic-indic-dist-320M")
            translations = indic_translate([t['text'] for t in text_data], 
                                           model, tokenizer, source_language, target_language)
            #translations = local_libre_translate(text_data, source_language, target_language)

            for i, item in enumerate(text_data):
                item['original_text'] = item['text']
                item['text'] = translations[i]
        # Romanize if requested
        elif romanize:
            #font_path = get_font_path('en)
            for item in text_data:
                item['original_text'] = item['text']
                #item['text'] = self.romanize_text(item['text'], source_language)
                item['text'] = indic_transliterate(item['text'], source_language, target_language)
        else:
            font_path = get_font_path(source_language)
        
        # Overlay text on image
        ocr_image = self.overlay_text_on_image(image, text_data, font_size=font_size, 
                                               transparency=transparency, font_path=font_path)
        return ocr_image, text_data

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
        ocr_path = os.path.join(self.output_dir, f"page_{page_num + 1:04d}_ocr.jpg")
        if ignore_existing and os.path.exists(ocr_path):
            print(f"OCR image for page {page_num + 1} already exists. Skipping...")
            return {
                'page_num': page_num + 1,
                'text_count': 0,
                'ocr_image_path': ocr_path,
                'ocr_json_path': None,
                'text_data': []
            }
        image = self.extract_page_as_image(page_num)
        ocr_image, text_data = self.page_image_process(
            image,
            translate=translate,
            romanize=romanize,
            target_language=target_language,
            source_language=source_language,
            font_size=font_size,
            transparency=transparency
        )
        # Save OCR image
        print(ocr_path)
        ocr_image.save(ocr_path, quality=70, optimize=True)
        
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
    
    def create_searchable_pdf(self, output_pdf: str = None,
                            romanize:bool = False,
                            translate: bool = False,
                            target_language: str = "es",
                            source_language:str = None,
                            font_size = 60,
                            transparency = 0.8,
                            ignore_existing=False,
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
            from fpdf import FPDF
        except ImportError:
            print("Install required: pip install pypdf reportlab")
            return None
        
        if output_pdf is None:
            output_pdf = os.path.join(self.output_dir, "searchable_output.pdf")
        
        # Process all pages
        results = self.process_all_pages(romanize, translate, target_language, source_language,font_size=font_size,
                                        transparency=transparency, ignore_existing=ignore_existing,**kwargs)
        
        pdf = FPDF()
        pdf.set_compression(True) 

        if romanize:
            out_lang = 'en'

        if translate:
            out_lang = target_language

        # For indian languages
        if out_lang in ['ta', 'hi']:
            pdf.add_font(family="Hind_Madurai", fname="./fonts/Hind_Madurai/", uni=True)
            pdf.set_font("Hind_Madurai", size=font_size)
        # Chinese
        # pdf.add_font("CactusClassicalSerif", fname="./fonts/CactusClassicalSerif/CactusClassicalSerif-Regular.ttf", uni=True) 
        # English
        
        pdf.set_text_color(0, 0, 0) # Black color (RGB)

        print("Creating searchable PDF...")
        for result in results[:10]:
            page_num = result['page_num'] - 1
            original_page = self.reader.pages[page_num]
            box = original_page.mediabox
            pdf.add_page(format=(box.width, box.height))
            
            # Create text overlay
            if result['ocr_image_path'] is not None and os.path.exists(result['ocr_image_path']):
                print(result['ocr_image_path'])
                pdf.image(result['ocr_image_path'], x=0, y=0, w=pdf.w, h=pdf.h)

            for item in result['text_data']:
                pdf.set_xy(item['x'], item['y'])
                pdf.write(font_size, item['text'])
        
        pdf.output(output_pdf)
        
        print(f"Searchable PDF saved to: {output_pdf}")
        return output_pdf
    
    
IMAGES = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
DOCS = ['.pdf']#, '.epub', '.mobi']

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--file", type=str, help="File to translate")
    parser.add_argument('-o', '--output-dir', default='./output/books', help="Output directory")
    parser.add_argument('--romanize', action='store_true', help="Romanize the extracted text")
    parser.add_argument('--translate', action='store_true', help="Translate the extracted text")
    parser.add_argument('-t', '--target-language', default='en', type=str, help="Target language for translation (default: en)")
    parser.add_argument('-s', '--source-language', default='auto', type=str, help="Source language for translation (default: en)")
    parser.add_argument('--font-size', default=40, type=int, help="Font size for overlay text in PDF")
    parser.add_argument('--transparency', default=0.8, type=float, help="Transparency for overlay text in PDF")
    parser.add_argument('--ignore-existing', default=False, action='store_true', help="Ignore existing OCR images when processing PDF")
    #parser.add_argument('-i', '--image-file', default = None, type=str, help="Optional image file for testing")
    args = parser.parse_args()
    
    # Open image, test text reading on the image
    fname = Path(args.file).stem
    ext = Path(args.file).suffix
    outfile = Path(f'{args.output_dir}/{fname}_out{ext}')
    print(outfile)
    if ext.lower() in IMAGES:
        ocr = TamilPDFOCR(None, args.output_dir)
        image = Image.open(args.file) 
        out_image, text_data = ocr.page_image_process(
            image,
            translate=args.translate,
            romanize=args.romanize,
            target_language=args.target_language,
            source_language=args.source_language,
            font_size=args.font_size,
            transparency=args.transparency
        )
         # Save output image
        out_image.save(outfile)
    elif ext.lower() in DOCS:
        if args.source_language == 'ta' and args.romanize:
            ocr = TamilPDFOCR(args.file, args.output_dir)
            ocr.create_searchable_pdf(output_pdf=outfile, **vars(args))

    # process the whole 
if __name__=="__main__":
    #asyncio.run(main())
    main()

