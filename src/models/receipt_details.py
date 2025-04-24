from typing import List, Dict
from src.models.item import Item

class ReceiptDetails:
    """
    Represents the details of a receipt.
    """
    def __init__(self, items: List[Item], receipt_path: str = None, subtotal: float = 0.0, tax: float = 0.0, total: float = 0.0, category_totals: Dict[str, float] = None, category_tax_percents: Dict[str, float] = None, category_tax_amounts: Dict[str, float] = None):
        self.items = items
        self.receipt_path = receipt_path
        self.subtotal = subtotal
        self.tax = tax
        self.total = total  # Add new total field
        self.category_totals = category_totals or {}  # Total sum for each category
        self.category_tax_percentages = category_tax_percents or {}  # Tax totals for each category
        self.category_tax_amounts = category_tax_amounts or {}  # Tax amounts for each category

    def __repr__(self):
        return (f"ReceiptDetails(receipt_path={self.receipt_path}, subtotal={self.subtotal}, tax={self.tax}, total={self.total}, "
                f"category_totals={self.category_totals}, category_tax_percentages={self.category_tax_percentages}, "
                f"category_tax_amounts={self.category_tax_amounts}, items={self.items})")
