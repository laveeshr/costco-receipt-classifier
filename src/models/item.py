class Item:
    """
    Represents an item on a receipt.
    """
    def __init__(self, code, name, price=0.0):
        self.code = code
        self.name = name
        self.price = price
        self.category = None  # Category is set later by the classifier

    def __repr__(self):
        return f"Item(code={self.code}, name={self.name}, price={self.price}, category={self.category})"
