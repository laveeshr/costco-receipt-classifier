# Costco Receipt Classifier

## Overview
The Costco Receipt Classifier is a Python-based project designed to process and classify items from Costco receipts. It uses OCR (Optical Character Recognition) to extract text from receipt images or PDFs, preprocesses the data, and categorizes items into predefined categories. The project also includes functionality for training a machine learning model to improve classification accuracy over time.

## Features
- **OCR Integration**: Extract text from receipt images or PDFs using Tesseract OCR.
- **Item Categorization**: Automatically classify items into categories such as Groceries, Electronics, Clothes, etc.
- **Machine Learning**: Train and retrain a model using scikit-learn to improve classification accuracy.
- **Receipt Parsing**: Handle discounts, subtotals, and tax calculations.
- **Data Analysis**: Analyze training data to identify common items, verification rates, and more.
- **User Feedback**: Allow manual corrections to improve classification and retrain the model dynamically.

## Key Components
- **ReceiptAgent**: Core logic for training the model, predicting categories, and processing receipts.
- **CostcoReceiptParser**: Parses receipt text, extracts items, and handles OCR preprocessing.
- **Models**:
  - `Item`: Represents individual items on a receipt.
  - `ReceiptDetails`: Represents the details of a receipt, including items, subtotal, and tax.
- **Scripts**:
  - `process_receipt.py`: Script to process receipts from a directory and print results.
  - `update_categorization.py`: Utility to update item categories in the training data.

## Requirements
- Python 3.8+
- Dependencies listed in `requirements.txt`:
  - pandas, numpy, scikit-learn, joblib, python-dateutil, pytesseract, pillow

## Usage
1. **Setup**:
   - Install dependencies: `pip install -r requirements.txt`
   - Ensure Tesseract OCR is installed and configured.

2. **Train the Model**:
   - Add training data to `receipt_data/training_data.csv`.
   - Run the `ReceiptAgent` to train the model.

3. **Process Receipts**:
   - Place receipt images or PDFs in the `training_data` directory.
   - Use `process_receipt.py` to process and classify receipts.

4. **Update Categories**:
   - Use `update_categorization.py` to manually update item categories.

## Future Enhancements
- Add support for refund receipts.
- Develop a web or Android app for easier receipt uploads.
- Automate fetching and pushing receipt data to external services.

## Contributing
Contributions are welcome! Feel free to submit issues or pull requests to improve the project.

## License
This project is licensed under the MIT License.
