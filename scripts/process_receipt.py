import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.item import Item
from src.receipt_agent import ReceiptAgent
from src.costco_receipt_parser import CostcoReceiptParser

# Initialize the agent and parser
agent = ReceiptAgent()
parser = CostcoReceiptParser()

# Parse receipt details from images in the directory
receipt_details_list = parser.parse_images_from_directory("./training_data")
# pdf_list = parser.parse_receipts_from_directory("./training_data")

for receipt_details in receipt_details_list:
    # Process the receipt
    updated_receipt_details = agent.process_receipt(receipt_details)

    # Print results
    print(f"\nReceipt Path: {updated_receipt_details.receipt_path}")
    print("Classification Results:")
    for item in updated_receipt_details.items:
        print(item)

    # Summarize cost by categories
    category_summary = agent.summarize_cost_by_category(updated_receipt_details.items)
    print("\nCost Summary by Category:")
    for category, total_cost in category_summary.items():
        print(f"{category}: ${total_cost:.2f}")

    # Verify the receipt total
    if updated_receipt_details.subtotal is not None:
        is_verified = agent.verify_receipt_total(updated_receipt_details.items, updated_receipt_details.subtotal)
        if is_verified:
            print("\nReceipt verification successful: Subtotal matches the total summary.")
        else:
            print("\nReceipt verification failed: Subtotal does not match the total summary.")
    else:
        print("\nReceipt subtotal could not be extracted.")

# Print data analysis
analysis = agent.analyze_data()
print("\nTraining Data Analysis:")
for key, value in analysis.items():
    if isinstance(value, dict):
        print(f"{key}:")
        for k, v in value.items():
            print(f"  {k}: {v}")
    else:
        print(f"{key}: {value}")