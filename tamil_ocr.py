from ocr_romanize import OCRWithCliRomanize
from ocr_tamil.ocr import OCR
from ocr_tamil.craft_text_detector import (
    get_prediction,
    export_detected_regions
)
from PIL import Image
from typing import List, Dict, Tuple
import torch
from pathlib import Path
import cv2
import os
import json
import numpy as np

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
        return exported_file_paths,updated_prediction_result    

class TamilPDFOCR(OCRWithCliRomanize):
    def __init__(self, input_pdf: str, output_dir: str = "ocr_output"):
        super(TamilPDFOCR, self).__init__(input_pdf, output_dir)
        self.ocr = TamilOCR(detect=True, lang=['tamil'], assume_straight_page=True)
        
    
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
    
    def process_page(self, page_num: int,
                    romanize:bool = False,
                    translate: bool = False,
                    target_language: str = "en",
                    source_language:str = 'en',
                    save_ocr_json: bool = False,
                    font_size:int = 60,
                    transparency:float = 0.8) -> Dict:
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
        np_image = np.asarray(image)
        cv2_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        
        # Extract text with bounding boxes
        text_data = self.extract_text_with_bbox(cv2_image)
        
        # Translate if requested
        if translate:
            for item in text_data:
                item['original_text'] = item['text']
                item['text'] = self.translate_text(item['text'], target_language, source_language)

        # Romanize if requested
        if romanize:
            for item in text_data:
                item['original_text'] = item['text']
                item['text'] = self.romanize_text(item['text'], source_language)
        
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
    
    
IMAGES = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
DOCS = ['.pdf']#, '.epub', '.mobi']

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--file", type=str, help="File to translate")
    parser.add_argument('-o', '--output-dir', default='./output/books', help="Output directory")
    parser.add_argument('--romanize', action='store_true', help="Romanize the extracted text")
    parser.add_argument('--translate', action='store_true', help="Translate the extracted text")
    parser.add_argument('-t', '--target-language', default='en', type=str, help="Target language for translation (default: en)")
    parser.add_argument('-s', '--source-language', default='auto', type=str, help="Source language for translation (default: en)")
    parser.add_argument('--font-size', default=60, type=int, help="Font size for overlay text in PDF")
    parser.add_argument('--transparency', default=0.8, type=float, help="Transparency for overlay text in PDF")
    #parser.add_argument('-i', '--image-file', default = None, type=str, help="Optional image file for testing")
    args = parser.parse_args()
    
    # Open image, test text reading on the image
    fname = Path(args.file).stem
    ext = Path(args.file).suffix
    outfile = Path(f'{args.output_dir}/{fname}_out{ext}')
    print(outfile)
    if ext.lower() in IMAGES:
        ocr = TamilPDFOCR(None, args.output_dir)
        results = ocr.extract_text_with_bbox(args.file)

        for item in results:
            item['original_text'] = item['text']
            item['text'] = ocr.romanize_text(item['text'], 'ta')

        image = Image.open(args.file) 
        out_image = ocr.overlay_text_on_image(image, results, font_size = 60, transparency=0.8) 
        out_image.save(outfile)
    elif ext.lower() in DOCS:
        if args.source_language == 'ta' and args.romanize:
            ocr = TamilPDFOCR(args.file, args.output_dir)
            ocr.create_searchable_pdf(output_pdf=outfile, **vars(args))

    # process the whole 

