import json
from datetime import datetime
from src.models.receipt_details import ReceiptDetails
from src.models.item import Item
from src.costco_pdf_receipt_parser import CostcoPdfReceiptParser

class CostcoOnlineOrderParser(CostcoPdfReceiptParser):
    def parse_online_order_json(self, json_data: dict, file_path: str) -> list[ReceiptDetails]:
        """
        Parse the JSON data for online orders and return a list of ReceiptDetails objects.
        
        Args:
            json_data: The JSON data containing online order details.
            file_path: The file path of the JSON file (for reference in ReceiptDetails).
        """
        try:
            orders = json_data.get("data", {}).get("getOnlineOrders", [])[0].get("bcOrders", [])
            
            if not orders:
                raise ValueError("No orders found in the JSON data.")
            
            receipt_details_list = []
            
            for order in orders:
                items = []
                subtotal = 0.0
                tax = 0.0
                total = order.get("orderTotal", 0.0)
                warehouse_number = order.get("warehouseNumber", "")
                transaction_datetime = datetime.fromisoformat(order.get("orderPlacedDate"))
                
                for line_item in order.get("orderLineItems", []):
                    item = Item(
                        code=line_item.get("itemNumber", ""),
                        name=line_item.get("itemDescription", ""),
                        price=line_item.get("price", 0.0),
                        is_taxable=True
                    )
                    items.append(item)
                    subtotal += item.price
                
                # Calculate effective price for each item
                for item in items:
                    if subtotal > 0:
                        item_percentage = item.price / subtotal
                        item.price = item_percentage * total  # Effective price based on order total
                
                # Create and append the ReceiptDetails object
                receipt_details = ReceiptDetails(
                    items=items,
                    receipt_path=file_path,
                    subtotal=subtotal,
                    tax=0,
                    total=total,
                    transaction_datetime=transaction_datetime,
                    warehouse_number=str(warehouse_number),
                    )
                receipt_details_list.append(receipt_details)
            
            return receipt_details_list
        except Exception as e:
            print(f"Error parsing online order JSON data: {e}")
            return []
