import pandas as pd
from typing import List, Dict

def update_item_categories(file_path: str, updates: List[Dict[str, str]]) -> bool:
    """
    Update categories for multiple items in the CSV file.

    :param file_path: Path to the CSV file.
    :param updates: List of dictionaries with 'item_code' (str) and 'new_category' (str).
    :return: Boolean indicating success of the operation
    """
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        # Convert item_code column to string
        if df['item_code'].dtype != 'object':
            df['item_code'] = df['item_code'].astype(str)

        print(f"First few item_codes in DataFrame: {df['item_code'].head()} (type: {type(df['item_code'].iloc[0])})")
        
        successful_updates = 0
        for update in updates:
            item_code = str(float(update['item_code']))
            new_category = update['new_category']
            
            # Debug logging
            print(f"Searching for item_code: {item_code}")            
            
            # Find matching rows
            mask = df['item_code'] == item_code
            if not any(mask):
                print(f"Warning: Item code {item_code} not found in the dataset")
                continue
            
            # Update category and mark as verified
            # Only update if the category is different
            if df.loc[mask, 'category'].iloc[0] != new_category:
                df.loc[mask, 'category'] = new_category
                df.loc[mask, 'verified'] = True
                df.loc[mask, 'confidence'] = 1.0
                df.loc[mask, 'timestamp'] = pd.Timestamp.now()
            else:
                print(f"Category for item {item_code} already set to {new_category}")
                
            successful_updates += 1
        
        # Save the updated file
        df.to_csv(file_path, index=False)
        
        print(f"Successfully updated {successful_updates} out of {len(updates)} items.")
        return successful_updates > 0
        
    except Exception as e:
        print(f"Error updating categories: {str(e)}")
        return False

# Example usage
if __name__ == "__main__":
    file_path = "/home/laveeeshr/Projects/costco-receipt-classifier/receipt_data/training_data.csv"
    updates = [
        {"item_code": '1796136', "new_category": "Household"},
        {"item_code": '1752052', "new_category": "Clothes"},
        {"item_code": '1479425', "new_category": "Household"},
        {"item_code": '2858210', "new_category": "Selfcare"},
        {"item_code": '970125', "new_category": "Household"},
        {"item_code": '713160', "new_category": "Household"}
    ]
    success = update_item_categories(file_path, updates)
    print(f"Update operation {'succeeded' if success else 'failed'}")
