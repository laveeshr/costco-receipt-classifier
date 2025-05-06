from typing import List
import json
from datetime import datetime
from src.models.item import Item
from src.models.receipt_details import ReceiptDetails
from src.costco_online_order_parser import CostcoOnlineOrderParser

class CostcoJsonReceiptParser:
    def __init__(self):
        pass

    def _extract_reference_item_number(self, item_data: dict) -> str:
        """Extract the referenced item number from a discount entry.
        Checks both itemDescription01 and frenchItemDescription1 fields.
        """
        # Check frenchItemDescription1 first as it's more likely to contain the item number
        french_desc = item_data.get('frenchItemDescription1', '')
        if french_desc and french_desc.startswith('/'):
            return french_desc.lstrip('/')
            
        # Fall back to itemDescription01
        eng_desc = item_data.get('itemDescription01', '')
        if eng_desc and eng_desc.startswith('/'):
            # Try to extract number from the description
            desc_parts = eng_desc.lstrip('/ ').split()
            if desc_parts and desc_parts[0].isdigit():
                return desc_parts[0]
        
        return ''

    def _is_discount_line(self, item_data: dict) -> bool:
        """Check if the given item data represents a discount line.
        
        Args:
            item_data: Dictionary containing item data
            
        Returns:
            bool: True if this is a discount line, False otherwise
        """
        return item_data.get('taxFlag') is None

    def _collect_discounts(self, item_array: List[dict]) -> dict:
        """Collect all discounts from the item array and return a mapping of item numbers to discount amounts."""
        discounts = {}
        for item_data in item_array:
            if self._is_discount_line(item_data):
                amount = float(item_data.get('amount', 0.0))
                ref_item_number = self._extract_reference_item_number(item_data)
                if ref_item_number:
                    discounts[ref_item_number] = discounts.get(ref_item_number, 0) + amount
        return discounts

    def _create_items(self, item_array: List[dict], discounts: dict) -> List[Item]:
        """Create Item objects from the item array, applying any relevant discounts."""
        items = []
        for item_data in item_array:
            if not self._is_discount_line(item_data):
                amount = float(item_data.get('amount', 0.0))
                item_number = item_data.get('itemNumber', '')
                final_price = amount
                if item_number in discounts:
                    final_price += discounts[item_number]
                
                item = Item(
                    code=item_number,
                    name=item_data.get('itemDescription01', ''),
                    price=final_price,
                    is_taxable=item_data.get('taxFlag', 'N') == 'Y'
                )
                items.append(item)
        return items

    def parse_json_receipt(self, json_file_path: str) -> List[ReceiptDetails]:
        """Parse a JSON file containing Costco receipts.
        
        Args:
            json_file_path: Path to the JSON file containing receipt data
            
        Returns:
            List of ReceiptDetails objects
        """
        try:
            with open(json_file_path, 'r') as f:
                receipts_data = json.load(f)

            # Check if the JSON data matches the online order format
            if isinstance(receipts_data, dict) and "data" in receipts_data and "getOnlineOrders" in receipts_data["data"]:
                # Use the CostcoOnlineOrderParser for online orders
                online_parser = CostcoOnlineOrderParser()
                return online_parser.parse_online_order_json(receipts_data, json_file_path)

            receipt_details_list = []
            
            # Handle both single receipt and array of receipts
            if not isinstance(receipts_data, list):
                receipts_data = [receipts_data]

            for receipt in receipts_data:
                item_array = receipt.get('itemArray', [])
                
                # Process items in two passes
                discounts = self._collect_discounts(item_array)
                items = self._create_items(item_array, discounts)

                # Create warehouse location string
                warehouse_location = f"{receipt.get('warehouseCity', '')}, {receipt.get('warehouseState', '')}"
                
                # Parse transaction datetime
                transaction_datetime = None
                if receipt.get('transactionDateTime'):
                    transaction_datetime = datetime.fromisoformat(receipt['transactionDateTime'])

                # Create ReceiptDetails object
                receipt_details = ReceiptDetails(
                    items=items,
                    receipt_path=json_file_path,
                    subtotal=float(receipt.get('subTotal', 0.0)),
                    tax=float(receipt.get('taxes', 0.0)),
                    total=float(receipt.get('total', 0.0)),
                    transaction_datetime=transaction_datetime,
                    warehouse_name=receipt.get('warehouseName', ''),
                    warehouse_location=warehouse_location,
                    membership_number=receipt.get('membershipNumber', '')
                )
                receipt_details_list.append(receipt_details)

            return receipt_details_list

        except Exception as e:
            print(f"Error parsing JSON receipt {json_file_path}: {e}")
            return [ReceiptDetails(items=[], receipt_path=json_file_path)]
