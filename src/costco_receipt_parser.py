import os
import pytesseract
from PIL import Image, ImageFilter, ImageOps
import cv2
import numpy as np
from src.models.receipt_details import ReceiptDetails
from src.costco_pdf_receipt_parser import CostcoPdfReceiptParser
from src.costco_json_receipt_parser import CostcoJsonReceiptParser
import shutil

class CostcoReceiptParser:
    def __init__(self):
        self.pdf_parser = CostcoPdfReceiptParser()
        self.json_parser = CostcoJsonReceiptParser()

    def _resize_image(self, image, target_width=1500):
        """
        Resize the image to a target width while maintaining aspect ratio.
        """
        width, height = image.size
        if width < target_width:
            aspect_ratio = height / width
            new_height = int(target_width * aspect_ratio)
            return image.resize((target_width, new_height), Image.Resampling.LANCZOS)
        return image

    def _deskew_image(self, image):
        """
        Deskew the image to align text properly.
        """
        image_cv = np.array(image)
        # Ensure the image has 3 channels (RGB) before converting to grayscale
        if len(image_cv.shape) == 2:  # Already grayscale
            gray = image_cv
        elif len(image_cv.shape) == 3 and image_cv.shape[2] == 3:  # RGB
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Unsupported image format for deskewing.")
        coords = np.column_stack(np.where(gray > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = image_cv.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(image_cv, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(deskewed)

    def _sharpen_image(self, image):
        """
        Sharpen the image to enhance text edges.
        """
        return image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

    def _preprocess_image(self, image):
        """
        Preprocess the image to improve OCR accuracy.
        """
        # Resize the image
        image = self._resize_image(image)

        # Convert to grayscale
        image = image.convert("L")

        # Enhance contrast
        image = ImageOps.autocontrast(image)

        # Sharpen the image
        image = self._sharpen_image(image)

        # Denoise
        image = image.filter(ImageFilter.MedianFilter(size=3))

        # Convert to OpenCV format for thresholding
        image_cv = np.array(image)
        _, image_cv = cv2.threshold(image_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Convert back to PIL format
        return Image.fromarray(image_cv)

    def _extract_text_with_ocr(self, image):
        """
        Extract text from an image using Tesseract OCR.
        """
        return pytesseract.image_to_string(image, config="--psm 6")

    def parse_receipt_pdf(self, pdf_path):
        """Parse a PDF receipt."""
        return self.pdf_parser.parse_receipt_pdf(pdf_path)

    def parse_receipt_pdf_with_ocr(self, pdf_path):
        """Parse a PDF receipt using OCR."""
        pages = self.pdf_parser.parse_receipt_pdf_with_ocr(pdf_path)
        text = ''
        for page_num, image in enumerate(pages, 1):
            image = self._preprocess_image(image)
            page_text = self._extract_text_with_ocr(image)
            text += page_text + '\n'
        return self.pdf_parser.parse_receipt_text(text)

    def parse_receipt_image(self, image_path):
        """Parse an image receipt using OCR."""
        try:
            image = Image.open(image_path)
            image = self._preprocess_image(image)
            text = self._extract_text_with_ocr(image)
            receipt_details = self.pdf_parser.parse_receipt_text(text)
            receipt_details.receipt_path = image_path
            return receipt_details
        except Exception as e:
            print(f"Error reading image {image_path} with OCR: {e}")
            return ReceiptDetails(items=[], receipt_path=image_path, subtotal=None, tax=None)

    def parse_json_receipt(self, json_path: str) -> list[ReceiptDetails]:
        """Parse a JSON receipt file."""
        return self.json_parser.parse_json_receipt(json_path)

    def parse_receipts_from_directory(self, directory_path, use_ocr=False) -> list[ReceiptDetails]:
        """Parse all receipts in the directory."""
        receipt_details_list = []
        for file_name in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file_name)
            if file_name.lower().endswith('.json'):
                receipts = self.parse_json_receipt(file_path)
                receipt_details_list.extend(receipts)
            elif file_name.lower().endswith('.pdf'):
                if use_ocr:
                    receipt_details = self.parse_receipt_pdf_with_ocr(file_path)
                else:
                    receipt_details = self.parse_receipt_pdf(file_path)
                receipt_details_list.append(receipt_details)
            elif file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                receipt_details = self.parse_receipt_image(file_path)
                receipt_details_list.append(receipt_details)
            else:
                continue

            trained_dir = os.path.join('training_data', 'trained')
            os.makedirs(trained_dir, exist_ok=True)
            shutil.move(file_path, os.path.join(trained_dir, os.path.basename(file_path)))

        return receipt_details_list