import re
import os
import pdfplumber
import pytesseract
from PIL import Image, ImageFilter, ImageOps  # Add ImageOps for preprocessing
import cv2  # Add OpenCV for advanced preprocessing
import numpy as np
from pdf2image import convert_from_path
from src.models.item import Item
from src.models.receipt_details import ReceiptDetails
import shutil
from collections import defaultdict

class CostcoReceiptParser:

    def _parse_item_line(self, line):
        """
        Parse a single line to extract item details (code, name, price).

        Args:
            line: A single line of text from the receipt.

        Returns:
            A tuple (item_code, item_name, price) if the line matches the item pattern, otherwise None.
        """
        item_pattern = r'^[A-Z]?\s*(\d+)\s+([A-Za-z0-9\s\-\'\",\.]+)\s+(\d+\.\d{2})\s*[A-Z]?$'
        match = re.search(item_pattern, line)
        if match:
            item_code = match.group(1).strip()
            item_name = match.group(2).strip()
            price = float(match.group(3).strip())
            return item_code, item_name, price
        return None

    def _parse_discount_line(self, line):
        """
        Parse a single line to extract discount details.

        Args:
            line: A single line of text from the receipt.

        Returns:
            The discount amount as a float if the line matches the discount pattern, otherwise None.
        """
        discount_pattern = r'^\s*\d+\s+/\d+\s+(\d+\.\d{2})-\s*$'
        match = re.search(discount_pattern, line)
        if match:
            return float(match.group(1).strip())
        return None

    def _should_skip_line(self, item_name):
        """
        Determine if a line should be skipped based on keywords.

        Args:
            item_name: The name of the item extracted from the line.

        Returns:
            True if the line should be skipped, otherwise False.
        """
        skip_keywords = ['total', 'tax', 'member', 'warehouse', 'date']
        return any(keyword in item_name.lower() for keyword in skip_keywords)

    def _parse_subtotal_or_tax_line(self, line):
        """
        Parse a line to extract subtotal or tax.

        Args:
            line: A single line of text from the receipt.

        Returns:
            A tuple (type, amount) where type is 'subtotal' or 'tax', and amount is the extracted value as a float.
            Returns None if the line does not match.
        """
        subtotal_pattern = r'\bsubtotal\b.*?(\d+\.\d{2})'
        tax_pattern = r'\bTAX\b.*?(\d+\.\d{2})'

        subtotal_match = re.search(subtotal_pattern, line, re.IGNORECASE)
        if subtotal_match:
            return 'subtotal', float(subtotal_match.group(1))

        tax_match = re.search(tax_pattern, line, re.IGNORECASE)
        if tax_match:
            return 'tax', float(tax_match.group(1))

        return None

    def _categorize_items(self, items):
        """
        Categorize items and calculate totals and tax percentages for each category.

        Args:
            items: List of Item objects.

        Returns:
            A tuple containing:
            - Dictionary of category totals.
            - Dictionary of category tax percentages.
        """
        category_totals = defaultdict(float)
        category_tax_percentages = {}

        for item in items:
            # Assume each item has a 'category' attribute (e.g., 'food', 'electronics', etc.)
            category = getattr(item, 'category', 'uncategorized')
            category_totals[category] += item.price

        # Calculate tax percentages for each category
        total_price = sum(category_totals.values())
        for category, total in category_totals.items():
            category_tax_percentages[category] = (total / total_price) * 100 if total_price > 0 else 0

        return dict(category_totals), category_tax_percentages

    def parse_receipt_text(self, text):
        """
        Parse raw text from a Costco receipt and extract items, subtotal, tax, and category details.

        Returns:
            A ReceiptDetails object containing parsed receipt details.
        """
        items = []
        subtotal = None
        tax = None
        lines = text.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i]
            item_details = self._parse_item_line(line)
            if item_details:
                item_code, item_name, price = item_details

                # Check for a discount on the next line
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    discount = self._parse_discount_line(next_line)
                    if discount:
                        price -= discount  # Apply the discount
                        i += 1  # Skip the discount line

                # Skip header lines, totals, tax lines, etc.
                if self._should_skip_line(item_name):
                    i += 1
                    continue

                items.append(Item(
                    code=item_code,
                    name=item_name,
                    price=price
                ))
            else:
                # Check for subtotal or tax in the current line
                subtotal_or_tax = self._parse_subtotal_or_tax_line(line)
                if subtotal_or_tax:
                    if subtotal_or_tax[0] == 'subtotal':
                        subtotal = subtotal_or_tax[1]
                    elif subtotal_or_tax[0] == 'tax':
                        tax = subtotal_or_tax[1]

            i += 1

        # Categorize items and calculate category totals and tax percentages
        category_totals, category_tax_percentages = self._categorize_items(items)

        # Return a ReceiptDetails object
        return ReceiptDetails(
            items=items,
            receipt_path=None,  # No file path available in this context
            subtotal=subtotal,
            tax=tax,
            category_totals=category_totals,
            category_tax_percentages=category_tax_percentages
        )

    def parse_receipt_pdf(self, pdf_path):
        """
        Extract text from a PDF receipt and parse it.
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ''.join(page.extract_text() for page in pdf.pages)
                print(f"Extracted text from {pdf_path}:\n{text}")
            # Directly use the ReceiptDetails object returned by parse_receipt_text
            receipt_details = self.parse_receipt_text(text)
            receipt_details.receipt_path = pdf_path  # Set the receipt path
            return receipt_details
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return ReceiptDetails(items=[], receipt_path=pdf_path, subtotal=None, tax=None)

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

        # Deskew the image
        # image = self._deskew_image(image)

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

    def parse_receipt_image(self, image_path):
        """
        Use OCR to extract text from an image receipt and parse it.
        """
        try:
            # Open the image
            image = Image.open(image_path)

            # Preprocess the image
            image = self._preprocess_image(image)

            # Extract text using Tesseract
            text = self._extract_text_with_ocr(image)
            print(f"Extracted text using OCR from {image_path}:\n{text}")

            # Directly use the ReceiptDetails object returned by parse_receipt_text
            receipt_details = self.parse_receipt_text(text)
            receipt_details.receipt_path = image_path  # Set the receipt path
            return receipt_details
        except Exception as e:
            print(f"Error reading image {image_path} with OCR: {e}")
            return ReceiptDetails(items=[], receipt_path=image_path, subtotal=None, tax=None)

    def parse_receipt_pdf_with_ocr(self, pdf_path):
        """
        Use OCR to extract text from a PDF receipt and parse it.
        """
        try:
            # Convert PDF pages to images
            images = convert_from_path(pdf_path)
            text = ''
            for image in images:
                # Preprocess each image
                image = self._preprocess_image(image)
                # Extract text from each image using Tesseract
                text += self._extract_text_with_ocr(image)
            print(f"Extracted text using OCR from {pdf_path}:\n{text}")
            # Directly use the ReceiptDetails object returned by parse_receipt_text
            receipt_details = self.parse_receipt_text(text)
            receipt_details.receipt_path = pdf_path  # Set the receipt path
            return receipt_details
        except Exception as e:
            print(f"Error reading PDF with OCR {pdf_path}: {e}")
            return ReceiptDetails(items=[], receipt_path=pdf_path, subtotal=None, tax=None)

    def parse_receipts_from_directory(self, directory_path, use_ocr=False):
        """
        Parse all PDF receipts in the specified directory.
        """
        receipt_details_list = []
        for file_name in os.listdir(directory_path):
            if file_name.lower().endswith('.pdf'):
                pdf_path = os.path.join(directory_path, file_name)
                if use_ocr:
                    receipt_details = self.parse_receipt_pdf_with_ocr(pdf_path)
                else:
                    receipt_details = self.parse_receipt_pdf(pdf_path)
                receipt_details_list.append(receipt_details)
        return receipt_details_list

    def parse_images_from_directory(self, directory_path):
        """
        Parse all image receipts in the specified directory and move processed files to 'trained' directory.
        """
        receipt_details_list = []
        trained_dir = os.path.join(directory_path, "trained")
        os.makedirs(trained_dir, exist_ok=True)

        for file_name in os.listdir(directory_path):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(directory_path, file_name)
                receipt_details = self.parse_receipt_image(image_path)
                receipt_details_list.append(receipt_details)
                
                # Move the processed file to the 'trained' directory
                shutil.move(image_path, os.path.join(trained_dir, file_name))
        
        return receipt_details_list

# Example usage
if __name__ == "__main__":
    # Sample receipt text (this would come from OCR or manual input)
    sample_receipt = """
    CosTco
=—=W/HOLESALE
LYNNWOOD #1190
18109 33RD AVE W
LYNNWOOD, WA 98037
211190208005 72503191309
Member 111953820648
E 1296507 ORG LENTILS 14.79 N
349802 /1296507 3.80-
E 177 ALB ORG FUJI 7.99 N
E 1550956 PANEER CHEES 8.99 N
E 1215097 KS ORG PEAS 8.99 N
E 9211 ORG YEL ONIO 4.99 N
E 1745718 ONLY BEAN 7.89 N
1701671 KS SCENT PCR 17.99 Y
1796136 CERT. 10PK 9.99 Y
E 1827786 CHPTLE RANCH 7.99 N
E 568915 ORG CUCUMBER 6.99 N
E 1015237 KS STIR FRY 9.59 N
970125 DELUXE BUNCH 9.99 Y
E 47825 GREEN GRAPES 7.99 N
SUBTOTAL 120.37
TAX 4.02
see TOTAL
XXXXXXXXXXXXX8190 CHIP read
APPROVED - PURCHASE
AMOUNT: $124.39

04/01/2025 20:16 1146330130

COSTCO VISA 70.23

CHANGE o

TOTAL TAX 0.00
TOTAL NUMBER OF ITEMS SOLD = 7
    """
    
    parser = CostcoReceiptParser()
    # parsed_items, subtotal, tax = parser.parse_receipt_text(sample_receipt)
    # print("Parsed Items:")
    # for item in parsed_items:
    #     print(f"Code: {item.code}, Name: {item.name}, Price: {item.price}")
    # print(f"Subtotal: {subtotal}")
    # print(f"Tax: {tax}")

    # Example usage for parsing PDFs
    training_data_dir = "./training_data"
    # parsed_receipts_from_pdfs = parser.parse_receipts_from_directory(training_data_dir)
    # print("Parsed Receipts from PDFs:")
    # for receipt in parsed_receipts_from_pdfs:
    #     print(f"Receipt Path: {receipt.receipt_path}")
    #     for item in receipt.items:
    #         print(f"Code: {item.code}, Name: {item.name}, Price: {item.price}")
    #     print(f"Subtotal: {receipt.subtotal}")
    #     print(f"Tax: {receipt.tax}")

    # Example usage for parsing PDFs with OCR
    # parsed_receipts_from_pdfs_with_ocr = parser.parse_receipts_from_directory(training_data_dir, use_ocr=True)
    # print("Parsed Receipts from PDFs (OCR):")
    # for receipt in parsed_receipts_from_pdfs_with_ocr:
    #     print(f"Receipt Path: {receipt.receipt_path}")
    #     for item in receipt.items:
    #         print(f"Code: {item.code}, Name: {item.name}, Price: {item.price}")
    #     print(f"Subtotal: {receipt.subtotal}")
    #     print(f"Tax: {receipt.tax}")

    # Example usage for parsing images
    parsed_receipts_from_images = parser.parse_images_from_directory(training_data_dir)
    print("Parsed Receipts from Images:")
    for receipt in parsed_receipts_from_images:
        print(f"Receipt Path: {receipt.receipt_path}")
        for item in receipt.items:
            print(f"Code: {item.code}, Name: {item.name}, Price: {item.price}")
        print(f"Subtotal: {receipt.subtotal}")
        print(f"Tax: {receipt.tax}")
        print(f"Category Totals: {receipt.category_totals}")
        print(f"Category Tax Percentages: {receipt.category_tax_percentages}")