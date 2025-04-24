import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.receipt_details import ReceiptDetails
from src.receipt_agent import ReceiptAgent
from src.costco_receipt_parser import CostcoReceiptParser
# from src.utils.whatsapp_notifier import WhatsAppNotifier
from src.utils.email_notifier import EmailNotifier

# Initialize the agent, parser, and notifiers
agent = ReceiptAgent()
parser = CostcoReceiptParser()
# whatsapp_notifier = WhatsAppNotifier()
email_notifier = EmailNotifier()

# Parse receipt details from images in the directory
# receipt_details_list = parser.parse_images_from_directory("./training_data")
receipt_details_list: list[ReceiptDetails] = parser.parse_receipts_from_directory("./training_data", use_ocr=False)

for receipt_details in receipt_details_list:
    # Process the receipt
    updated_receipt_details = agent.process_receipt(receipt_details)

    # Verify the receipt total
    if updated_receipt_details.subtotal is not None:
        is_verified = agent.verify_receipt_total(updated_receipt_details)
        if is_verified:
            print("\nReceipt verification successful: Subtotal matches the total summary.")
        else:
            print("\nReceipt verification failed: Subtotal does not match the total summary.")
    else:
        print("\nReceipt subtotal could not be extracted.")

    # Send notifications
    
    # whatsapp_notifier.send_receipt_summary(
    #     category_totals=updated_receipt_details.category_totals,
    #     category_tax_totals=updated_receipt_details.category_tax_totals,
    #     subtotal=updated_receipt_details.subtotal,
    #     tax=updated_receipt_details.tax,
    #     total=updated_receipt_details.total
    # )
    email_notifier.send_receipt_summary(
        receipt_details=updated_receipt_details,
        is_verified=is_verified
    )

    # Print results
    print(f"\nReceipt Path: {updated_receipt_details.receipt_path}")
    print("Classification Results:")
    for item in updated_receipt_details.items:
        print(item)

    # Summarize cost by categories
    print("\nCost and Tax Summary by Category:")
    print(receipt_details.category_totals)
    print(receipt_details.category_tax_amounts)

# Print data analysis
# analysis = agent.analyze_data()
# print("\nTraining Data Analysis:")
# for key, value in analysis.items():
#     if isinstance(value, dict):
#         print(f"{key}:")
#         for k, v in value.items():
#             print(f"  {k}: {v}")
#     else:
#         print(f"{key}: {value}")