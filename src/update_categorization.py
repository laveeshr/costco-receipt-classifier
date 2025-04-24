import pandas as pd
from typing import List, Dict

def update_item_categories(file_path: str, updates: List[Dict[str, str]]) -> None:
    """
    Update categories for multiple items in the CSV file.

    :param file_path: Path to the CSV file.
    :param updates: List of dictionaries with 'item_code' (int) and 'new_category' (str).
    """
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Update categories for each item in the updates list
    for update in updates:
        item_code: int = update['item_code']
        new_category: str = update['new_category']
        df.loc[df['item_code'] == item_code, 'category'] = new_category
    
    # Save the updated file
    df.to_csv(file_path, index=False)
    print(f"Updated categories for {len(updates)} items.")

# Example usage
file_path: str = "/home/laveeeshr/Projects/costco-receipt-classifier/receipt_data/training_data.csv"
updates: List[Dict[str, str]] = [
    {"item_code": 970125, "new_category": "Household"},
    {"item_code": 1796136, "new_category": "Household"},
    {"item_code": 1701671, "new_category": "Household"}
]
update_item_categories(file_path, updates)
