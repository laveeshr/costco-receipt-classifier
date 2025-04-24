class Item:
    """
    Represents an item on a receipt.
    """
    def __init__(self, code, name, price=0.0, is_taxable=False):
        self.code = code
        self.name = name
        self.price = price
        self.category = None  # Category is set later by the classifier
        self.is_taxable = is_taxable  # Whether the item is taxable (Y) or not (N)

    def __repr__(self):
        return f"Item(code={self.code}, name={self.name}, price={self.price}, category={self.category}, is_taxable={self.is_taxable})"
