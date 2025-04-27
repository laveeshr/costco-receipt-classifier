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
            parsed_date = self._parse_date_from_path(receipt_details.receipt_path)
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = f"Costco Receipt Summary - {parsed_date}"            

            body = f"""
            <html>
            <head>
                <style>
                    details {
                        margin: 10px 0;
                        padding: 10px;
                        background-color: #f9f9f9;
                        border-radius: 4px;
                    }
                    summary {
                        cursor: pointer;
                        padding: 5px;
                    }
                </style>
            </head>
            <body>
            <h1>Costco Receipt Summary ({parsed_date})</h1>
            <h2>Category Breakdown:</h2>
            """
            
            for category, amount in receipt_details.category_totals.items():
                tax_amount = receipt_details.category_tax_amounts.get(category, 0.0)
                tax_percent = receipt_details.category_tax_percentages.get(category, 0.0)
                body += f"""
                <details>
                    <summary>
                        <strong>{category}</strong> - Total: ${(amount + tax_amount):.2f}
                    </summary>
                    <div style="margin-left: 20px;">
                        <p>Subtotal: ${amount:.2f}</p>
                        <p>Tax: ${tax_amount:.2f} ({tax_percent:.1f}%)</p>
                        <ul>
                """
                for item in receipt_details.items:
                    if item.category == category:
                        body += f"<li>{item.description} - ${item.price:.2f}</li>"
                body += """
                        </ul>
                    </div>
                </details>
                """

            body += f"""
            <h2>Receipt Summary:</h2>
            <p>Subtotal: ${receipt_details.subtotal:.2f}</p>
            <p>Tax: ${receipt_details.tax:.2f}</p>
            <p>Total: ${receipt_details.total:.2f}</p>
            <p>Receipt Path: {receipt_details.receipt_path}</p>
            <p>Verification Status: {'✅ Verified' if is_verified else '❌ Not Verified'}</p>
            </body>
            </html>
            """

            msg.attach(MIMEText(body, 'html'))

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

    def _parse_date_from_path(self, receipt_path: str) -> str:
        # Extract date from receipt path (assuming format like 01-03-2024.pdf)
            receipt_date = os.path.basename(receipt_path).split('.')[0]
            try:
                return datetime.strptime(receipt_date, '%m-%d-%Y').strftime('%Y-%m-%d')
                
            except ValueError:
                # Fallback to current date if parsing fails
                return "Unknown"