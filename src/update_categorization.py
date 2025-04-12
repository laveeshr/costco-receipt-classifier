import pandas as pd

def update_item_category(file_path, item_code, new_category):
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Locate the item by item_code and update its category
    df.loc[df['item_code'] == item_code, 'category'] = new_category
    
    # Save the updated file
    df.to_csv(file_path, index=False)
    print(f"Updated item with code {item_code} to category '{new_category}'.")

# Example usage
file_path = "/home/laveeeshr/Projects/costco-receipt-classifier/receipt_data/training_data.csv"
update_item_category(file_path, item_code=11223.0, new_category="Electronics")
