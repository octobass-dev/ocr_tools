from ocr_romanize import OCRWithCliRomanize
from ocr_tamil.ocr import OCR
from PIL import Image
import os


class TamilPDFOCR(OCRWithCliRomanize):
    def __init__(self, input_pdf: str, output_dir: str = "ocr_output"):
        super(TamilPDFOCR, self).__init__(input_pdf, output_dir)
        self.ocr = OCR(detect=True, batch_size=1, lang=['tamil'])

    def extract_text_with_bbox(self, image_file:str) -> list[dict]:
        """
        Extract text from image with bounding boxes
        
        Args:
            image: PIL Image object
        
        Returns:
            List of dicts with text, bbox, and confidence
        """
        results = []
        text_list, conf, bbox = self.ocr.predict(image_file, regions=True)
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

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--file", type=str, help="File to translate")
    parser.add_argument('-o', '--output-dir', default='./output/books', help="Output directory")
    parser.add_argument('-i', '--image-file', type=str, help="Optional image file for testing")
    args = parser.parse_args()
    ocr = TamilPDFOCR(args.file, args.output_dir)
    
    # Open image, test text reading on the image
    results = ocr.extract_text_with_bbox(args.image_file)

    for item in results:
        item['original_text'] = item['text']
        item['text'] = ocr.romanize_text(item['text'], 'ta')

    image = Image.open(args.image_file) 
    out_image = ocr.overlay_text_on_image(image, results, font_size = 50) 
    dn, fn = os.path.split(args.image_file)
    out_image.save(f'{dn}/out_{fn}')

