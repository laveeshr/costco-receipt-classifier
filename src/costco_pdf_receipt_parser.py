from PIL import Image
from typing import List
import pdfplumber
from pdf2image import convert_from_path
import re
from src.models.item import Item
from src.models.receipt_details import ReceiptDetails

class CostcoPdfReceiptParser:
    def __init__(self):
        return

    def _parse_item_line(self, line):
        """Parse a single line to extract item details."""
        # Updated pattern to include Y/N at the end and optional leading character
        item_pattern = r'^([A-Z])?\s*(\d+)\s+(?:(.+?)\s+)?(\d+\.\d{2})\s*([YN])$'
        match = re.search(item_pattern, line)
        if match:
            item_code = match.group(2).strip() if match.group(2) else ""
            item_name = match.group(3).strip() if match.group(3) else ""
            price = float(match.group(4))
            is_taxable = match.group(5) == 'Y'
            return item_code, item_name, price, is_taxable
        return None

    def _parse_discount_line(self, line):
        """Parse a discount line."""
        # Updated pattern to handle more variations in discount format
        discount_pattern = r'^\s*\d+\s*/\s*(?:\d+|[A-Z]+(?:\s+[A-Z]+)*)\s+(\d+(?:\.\d{2})?)-\s*$'
        match = re.search(discount_pattern, line)
        if match:
            return float(match.group(1))
        return None

    def _parse_subtotal_or_tax_line(self, line):
        """Parse subtotal or tax line."""
        subtotal_pattern = r'\bSUBTOTAL\b.*?(\d+\.\d{2})'
        tax_pattern = r'\bTAX\b.*?(\d+\.\d{2})'

        if 'SUBTOTAL' in line:
            match = re.search(subtotal_pattern, line, re.IGNORECASE)
            return ('subtotal', float(match.group(1))) if match else None
        elif 'TAX' in line and 'TOTAL TAX' not in line:
            match = re.search(tax_pattern, line, re.IGNORECASE)
            return ('tax', float(match.group(1))) if match else None
        return None

    def _is_total_line(self, line):
        """Parse total line."""
        match = re.search(r'^\s*[*\s]*TOTAL\s+(\d+\.\d{2})', line, re.IGNORECASE)
        return float(match.group(1)) if match else None

    def _is_price_or_special_line(self, line: str) -> bool:
        """Check if a line contains price pattern or special keywords."""
        return (re.search(r'\d+\.\d{2}\s*[YN]$', line) or
                self._parse_subtotal_or_tax_line(line) or
                'TOTAL' in line or
                self._parse_discount_line(line))

    def parse_receipt_text(self, text: str) -> ReceiptDetails:
        """Parse the receipt text and return ReceiptDetails object."""
        items = []
        subtotal = None
        tax = None
        total = None
        pending_name_parts = []  # Store pending name parts before finding an item
        
        lines = text.split('\n')
        started_parsing = False
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            if not started_parsing and ('Member' in line or line.startswith('111953820648')):
                started_parsing = True
                i += 1
                continue
            
            if not started_parsing:
                i += 1
                continue
            
            # Check if this is a name-only line (no price pattern)
            if not self._is_price_or_special_line(line) and line and line != "Member" and line != "111953820648":
                pending_name_parts.append(line)
                i += 1
                continue
            
            item_details = self._parse_item_line(line)
            if item_details:
                # Process current item
                code, name, price, is_taxable = item_details
                current_item = Item(code=code, name=name, price=price, is_taxable=is_taxable)
                
                # Prepend any pending name parts
                if pending_name_parts or not current_item.name:
                    prefix = ' '.join(pending_name_parts)
                    current_item.name = f"{prefix} {current_item.name}".strip()
                    pending_name_parts = []  # Clear pending parts
                        
                    # Look ahead for suffix name parts only if we had prefix
                    j = i + 1
                    suffix_parts = []
                    while j < len(lines):
                        next_line = lines[j].strip()
                        # Stop if we find a line with price pattern or special keywords
                        if self._is_price_or_special_line(next_line):
                            break
                        if next_line:  # Add non-empty lines to suffix
                            suffix_parts.append(next_line)
                        j += 1
                    
                    # Append suffix parts if any
                    if suffix_parts:
                        current_item.name = f"{current_item.name} {' '.join(suffix_parts)}".strip()
                        i = j - 1  # Update loop counter to skip processed suffix lines
                
                # Check for discount on next line
                if i + 1 < len(lines):
                    discount = self._parse_discount_line(lines[i + 1])
                    if discount:
                        current_item.price -= discount
                        i += 1
                
                items.append(current_item)
                # Log the item for debugging
                print(f"Parsed item: {current_item}")

            else:
                subtotal_or_tax = self._parse_subtotal_or_tax_line(line)
                if subtotal_or_tax:
                    if subtotal_or_tax[0] == 'subtotal':
                        subtotal = subtotal_or_tax[1]
                    else:
                        tax = subtotal_or_tax[1]
                elif 'TOTAL' in line:
                    total = self._is_total_line(line)
                    if total:
                        total = float(total)
                        break
            
            i += 1
        
        return ReceiptDetails(
            items=items,
            subtotal=subtotal,
            tax=tax,
            total=total
        )

    def parse_receipt_pdf(self, pdf_path) -> ReceiptDetails:
        """Extract and parse text from a PDF receipt."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = '\n'.join(page.extract_text() for page in pdf.pages)
                print(f"Extracted text from {pdf_path}:\n{text}")
                
                # Log the raw text for debugging
                print(f"Raw text from {pdf_path}:\n{text}")
                
                receipt_details = self.parse_receipt_text(text)
                receipt_details.receipt_path = pdf_path

                # Log the parsed receipt details for debugging
                print(f"Parsed receipt details from {pdf_path}:\n{receipt_details}")
                
                return receipt_details
            
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return ReceiptDetails(items=[], receipt_path=pdf_path, subtotal=None, tax=None)

    def parse_receipt_pdf_with_ocr(self, pdf_path) -> List[Image]:
        """
        Use OCR to extract text from a PDF receipt and parse it.
        """
        try:
            # Convert PDF pages to high-quality images
            pages = convert_from_path(
                pdf_path,
                dpi=300,  # Higher DPI for better quality
                fmt='png',  # Use lossless PNG format
                grayscale=True,  # Keep color information
                transparent=False,  # Ensure white background
                output_file=None,  # Keep in memory
                thread_count=2,  # Use multiple threads for faster processing
                size=None,  # Keep original size
                paths_only=False,  # Return PIL Images
                use_cropbox=True,  # Use PDF cropbox for more accurate dimensions
                strict=False  # Be more permissive with PDF format
            )
            
            if not pages:
                raise ValueError(f"No pages found in PDF: {pdf_path}")
                
            # Process the cleaned text
            return pages
        except Exception as e:
            print(f"Error reading PDF with OCR {pdf_path}: {e}")
            return []
