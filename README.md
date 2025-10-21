
## Requirements
- Install python packages as follows
`pip install -r requirements.txt`
- You'll need to install `poppler`, `pdfinfo` in your environment before you can process pdf files. Tested on linux, m1 mac.

TODO
- Font size autoscaling
- Sentence detection, overlay
- Add local translation server with Libre for free translation
- Test pdf,djvu ocr with image translation and romanization
- Add support for more languages
- Refactor and generalize code

How to use tamil OCR
```
python tamil_ocr.py -f <path to pdf/image> --romanize -t en -s ta --font-size 35 --transparency 0.75
```
