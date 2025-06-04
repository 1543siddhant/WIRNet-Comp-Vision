import os
import requests
from PIL import Image

class OCRTool:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('OCR_SPACE_API_KEY')
        if not self.api_key:
            raise ValueError("Set OCR_SPACE_API_KEY in env or pass to constructor.")
        self.api_url = "https://api.ocr.space/parse/image"

    def from_pytesseract(self, image):
        import pytesseract
        gray = image.convert("L")
        return pytesseract.image_to_string(gray, config='--psm 6').strip()

    def from_ocr_space(self, image_path, language='eng'):
        payload = {'apikey': self.api_key, 'language': language, 'isOverlayRequired': False}
        with open(image_path, 'rb') as f:
            r = requests.post(self.api_url, files={image_path: f}, data=payload)
        data = r.json()
        text = ""
        for res in data.get("ParsedResults", []):
            text += res.get("ParsedText", "")
        return text.strip()
