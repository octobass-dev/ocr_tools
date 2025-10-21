The goal is to have a completely free setup for booking scanning, romanization, translation, tokenization, and other dataset preprocessing tasks running mostly offline. Basis for three personal projects
- Digitization and preservation of Indian regional traditions
- Training regional LLMs
- Self-contained agent orchestration setup running on a personal server

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

## Notice
Please note that the fonts do not belong to me. I have chosen fonts from Google Fonts for convenience. You are responsible for all the stipulations associated with use and distribution of these fonts. Please let me know if you are the owner of a particular font and would like me to remove the font.