from whatsapp import GreenAPI
import os
from typing import Dict
from datetime import datetime

class WhatsAppNotifier:
    def __init__(self):
        # Get credentials from environment variables
        self.instance_id = os.getenv('GREENAPI_INSTANCE_ID')
        self.api_token = os.getenv('GREENAPI_API_TOKEN')
        self.to_number = os.getenv('WHATSAPP_TO_NUMBER')    # Your personal WhatsApp number
        
        if not all([self.instance_id, self.api_token, self.to_number]):
            raise ValueError("Missing required Green-API credentials in environment variables")
        
        self.client = GreenAPI(self.instance_id, self.api_token)

    def send_receipt_summary(self, category_totals: Dict[str, float], 
                           category_tax_totals: Dict[str, float], 
                           subtotal: float,
                           tax: float,
                           total: float) -> bool:
        """Send receipt summary via WhatsApp"""
        try:
            message = f"*Costco Receipt Summary* ({datetime.now().strftime('%Y-%m-%d')})\n\n"
            
            # Add category breakdown
            message += "*Category Breakdown:*\n"
            for category, amount in category_totals.items():
                message += f"• {category}: ${amount:.2f}\n"
            
            message += f"\n*Subtotal:* ${subtotal:.2f}"
            message += f"\n*Tax:* ${tax:.2f}"
            message += f"\n*Total:* ${total:.2f}"

            # Send message via Green-API
            response = self.client.sending.sendMessage(self.to_number, message)
            return response.code == 200
        except Exception as e:
            print(f"Error sending WhatsApp message: {e}")
            return False
