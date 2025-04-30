from dataclasses import dataclass
from typing import List
from datetime import datetime
from src.models.item import Item

@dataclass
class ReceiptDetails:
    items: List[Item]
    receipt_path: str
    subtotal: float
    tax: float
    total: float
    transaction_datetime: datetime = None
    warehouse_name: str = ""
    warehouse_location: str = ""
    warehouse_number: str = ""
    membership_number: str = ""
    category_totals: dict = None
    category_tax_percentages: dict = None
    category_tax_amounts: dict = None

    def __post_init__(self):
        if self.category_totals is None:
            self.category_totals = {}
        if self.category_tax_percentages is None:
            self.category_tax_percentages = {}
        if self.category_tax_amounts is None:
            self.category_tax_amounts = {}
