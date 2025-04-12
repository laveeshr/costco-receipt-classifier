from typing import List, Dict
from src.models.item import Item

class ReceiptDetails:
    """
    Represents the details of a receipt.
    """
    def __init__(self, items: List[Item], receipt_path: str, subtotal: float, tax: float, category_totals: Dict[str, float] = None, category_tax_percentages: Dict[str, float] = None):
        self.items = items
        self.receipt_path = receipt_path
        self.subtotal = subtotal
        self.tax = tax
        self.category_totals = category_totals or {}  # Total sum for each category
        self.category_tax_percentages = category_tax_percentages or {}  # Tax percentage for each category

    def __repr__(self):
        return (f"ReceiptDetails(receipt_path={self.receipt_path}, subtotal={self.subtotal}, tax={self.tax}, "
                f"category_totals={self.category_totals}, category_tax_percentages={self.category_tax_percentages}, items={self.items})")
