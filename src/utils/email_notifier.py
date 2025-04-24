import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from src.models.receipt_details import ReceiptDetails

class EmailNotifier:
    """Email notification handler for Costco receipts.
    
    Setup Instructions:
    1. Enable 2-Step Verification in Google Account (Security settings)
    2. Generate App Password:
        - Go to Google Account > Security
        - Under "2-Step Verification", click on "App passwords"
        - Select "Mail" and your device
        - Copy the 16-character password
    3. Set environment variables:
        export SENDER_EMAIL="your.email@gmail.com"
        export SENDER_EMAIL_PASSWORD="your-16-char-app-password"
        export RECIPIENT_EMAIL="recipient@email.com"
    """
    
    def __init__(self):
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.sender_email = os.getenv('SENDER_EMAIL')
        self.sender_password = os.getenv('SENDER_EMAIL_PASSWORD')
        self.recipient_email = os.getenv('RECIPIENT_EMAIL')

        if not all([self.sender_email, self.sender_password, self.recipient_email]):
            raise ValueError("Missing required email credentials in environment variables")

    def send_receipt_summary(self, receipt_details: ReceiptDetails, is_verified: bool) -> bool:
        """Send receipt summary via email"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = f"Costco Receipt Summary - {datetime.now().strftime('%Y-%m-%d')}"

            body = f"Costco Receipt Summary ({datetime.now().strftime('%Y-%m-%d')})\n\n"
            body += "Category Breakdown:\n"
            for category, amount in receipt_details.category_totals.items():
                tax_amount = receipt_details.category_tax_amounts.get(category, 0.0)
                tax_percent = receipt_details.category_tax_percentages.get(category, 0.0)
                body += f"\t• {category}:\n"
                body += f"\t\t  - Subtotal: ${amount:.2f}\n"
                body += f"\t\t  - Tax: ${tax_amount:.2f} ({tax_percent:.1f}%)\n"
                body += f"\t\t  - <b>Total: ${(amount + tax_amount):.2f}</b>\n"
                body += "\n"
            
            body += f"\nReceipt Summary:"
            body += f"\nSubtotal: ${receipt_details.subtotal:.2f}"
            body += f"\nTax: ${receipt_details.tax:.2f}"
            body += f"\nTotal: ${receipt_details.total:.2f}"
            body += f"\n\nReceipt Path: {receipt_details.receipt_path}"
            body += f"\n\nVerification Status: {'Verified' if is_verified else 'Not Verified'}"

            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                try:
                    server.starttls()
                    server.login(self.sender_email, self.sender_password)
                    server.send_message(msg)
                except smtplib.SMTPAuthenticationError:
                    print("SMTP Authentication failed. Make sure you're using an App Password if using Gmail.")
                    print("See class docstring for setup instructions.")
                    return False
                except smtplib.SMTPException as smtp_error:
                    print(f"SMTP error occurred: {smtp_error}")
                    return False
            return True
        except Exception as e:
            print(f"Error sending email: {e}")
            return False
