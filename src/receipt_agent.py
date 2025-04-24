import os
import pandas as pd
pd.options.mode.copy_on_write = True  # Enable copy-on-write mode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
import joblib
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from src.models.item import Item
from src.models.receipt_details import ReceiptDetails

class ReceiptAgent:
    def __init__(self, data_dir: str = "./receipt_data") -> None:
        self.data_dir = data_dir
        self.training_data_path = os.path.join(data_dir, "training_data.csv")
        self.model_path = os.path.join(data_dir, "model")
        self.confidence_threshold = 0.9  # Threshold for automatic classification
        self.vectorizer = None
        self.classifier = None
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize or load existing data
        required_columns = ['item_code', 'item_text', 'category', 'confidence', 'verified', 'timestamp']
        if os.path.exists(self.training_data_path):
            self.training_data = pd.read_csv(self.training_data_path)
            for col in required_columns:
                if col not in self.training_data.columns:
                    self.training_data[col] = None
        else:
            self.training_data = pd.DataFrame(columns=required_columns)
            self.training_data.to_csv(self.training_data_path, index=False)
        
        # Load model if it exists
        if os.path.exists(f"{self.model_path}_classifier.pkl"):
            self.load_model()
        
    def preprocess(self, item_name: str) -> str:
        """Preprocess the name of an Item object"""
        # Convert to lowercase
        text = str(item_name).lower()
        
        # Handle common Costco abbreviations
        abbreviations = {
            "ks": "kirkland signature",
            "org": "organic",
            "tp": "toilet paper",
            "rotis": "rotisserie",
            "chkn": "chicken",
            "pt": "paper towel",
            "det": "detergent",
            "chse": "cheese",
        }
        
        for abbr, full in abbreviations.items():
            text = text.replace(f" {abbr} ", f" {full} ")
            if text.startswith(f"{abbr} "):
                text = f"{full} {text[len(abbr)+1:]}"
        
        # Add synonym mapping
        synonyms = {
            "bath tissue": "toilet paper",
            "paper towels": "paper towel",
            "ipad": "tablet",
            "blk pepper": "black pepper",
        }
        for synonym, standard in synonyms.items():
            text = text.replace(synonym, standard)

        return text
    
    def train_model(self, items: List[Item], force: bool = False) -> bool:
        """
        Train or retrain the model using an array of Item objects.
        
        Args:
            items: List of Item objects to use for training.
            force: Whether to force training even with insufficient data.

        Returns:
            True if the model was successfully trained, False otherwise.
        """
        # Convert items to a DataFrame
        new_data = pd.DataFrame([{
            'item_code': item.code,
            'item_text': item.name,
            'category': item.category,
            'confidence': 1.0,  # Assume confidence is 1.0 for provided items
            'verified': True,  # Assume all provided items are verified
            'timestamp': datetime.now().isoformat()
        } for item in items])

        # Append new data to training data
        self.training_data = pd.concat([self.training_data, new_data], ignore_index=True)

        # Cleanup: Keep only unique data points based on 'item_code' and 'item_text'
        self.training_data.drop_duplicates(subset=['item_code', 'item_text'], keep='last', inplace=True)

        self.training_data.to_csv(self.training_data_path, index=False)

        # Only use verified data for training
        train_df = self.training_data[self.training_data['verified'] == True]

        # Check if we have enough data and at least 2 samples per category
        category_counts = train_df['category'].value_counts()
        min_samples_per_category = 2
        
        if not force and (
            len(train_df) < 10 or 
            any(count < min_samples_per_category for count in category_counts.values.tolist())
        ):
            print(f"Not enough data for training. Need at least {min_samples_per_category} samples per category.")
            print("Current category counts:", category_counts.to_dict())
            return False

        # Preprocess text
        train_df.loc[:, 'processed_text'] = train_df['item_text'].apply(self.preprocess)

        # Create vectorizer
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            max_features=500
        )

        # Create feature matrix
        X = self.vectorizer.fit_transform(train_df['processed_text'])
        y = train_df['category']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train classifier
        self.classifier = RandomForestClassifier(n_estimators=100)
        self.classifier.fit(X_train, y_train)

        # Evaluate
        y_pred = self.classifier.predict(X_test)
        print("Model performance:")
        print(classification_report(
            y_test, 
            y_pred,
            zero_division=0,  # Handle categories with no predictions
            digits=3  # Show 3 decimal places for better precision monitoring
        ))

        # Save model
        self.save_model()

        return True
    
    def predict(self, item: Item) -> Tuple[Optional[str], float]:
        """
        Predict category for an Item object using item.code or item.name.

        Args:
            item: The Item object to predict the category for.

        Returns:
            A tuple containing the predicted category and the confidence score.
        """
        if not self.classifier or not self.vectorizer:
            # If no model exists, return None
            return None, 0

        # Check if item.code exists in training data
        if item.code:
            exact_match = self.training_data[self.training_data['item_code'] == item.code]
            if not exact_match.empty and any(exact_match['verified']):
                # If we have a verified match, use that category
                category = exact_match[exact_match['verified']].iloc[0]['category']
                return category, 1.0

        # Preprocess item name
        processed_text = self.preprocess(item.name)

        # Transform to features
        features = self.vectorizer.transform([processed_text])

        # Predict
        category = self.classifier.predict(features)[0]

        # Get confidence
        probabilities = self.classifier.predict_proba(features)[0]
        confidence = max(probabilities)

        return category, confidence
    
    def process_receipt(self, receipt_details: ReceiptDetails, user_feedback: bool = True) -> ReceiptDetails:
        """
        Process a ReceiptDetails object.

        Args:
            receipt_details: A ReceiptDetails object containing items, receipt path, and subtotal.
            user_feedback: Whether to prompt for user feedback.

        Returns:
            The updated ReceiptDetails object with categorized items.
        """
        items = receipt_details.items
        results = []

        for item in items:
            # Check if we've seen this exact item code or name before
            exact_match = self.training_data[
                (self.training_data['item_code'] == item.code) |
                (self.training_data['item_text'] == item.name)
            ]
            
            if not exact_match.empty and any(exact_match['verified']):
                # If we have a verified match, use that category
                category = exact_match[exact_match['verified']].iloc[0]['category']
                confidence = 1.0
                verified = True
            elif self.classifier is not None:
                # Otherwise use the model to predict
                category, confidence = self.predict(item)
                verified = confidence >= self.confidence_threshold
            else:
                # No model yet, use default category
                category = "Uncategorized"
                confidence = 0
                verified = False

            # Get user feedback if needed
            if user_feedback and not verified:
                print(f"\nItem: {item.name} (Code: {item.code})")
                print(f"Predicted category: {category} (confidence: {confidence:.2f})")
                # Dynamic category suggestions
                suggested_categories = self.training_data['category'].value_counts().head(5).index.tolist()
                print(f"Suggested categories: {', '.join(suggested_categories)}")
                user_category = input("Enter correct category (or press Enter to accept prediction): ")
                
                if user_category:
                    category = user_category
                    verified = True
                    confidence = 1.0
            
            # Update item
            item.category = category
            
            # Add to results
            results.append(item)
            
            # Add to training data
            new_row = pd.DataFrame([{
                'item_code': item.code,
                'item_text': item.name,
                'category': category,
                'confidence': confidence,
                'verified': verified,
                'timestamp': datetime.now().isoformat()
            }])
            self.training_data = pd.concat([self.training_data, new_row], ignore_index=True)

        # Cleanup: Keep only unique data points based on 'item_code' and 'item_text'
        self.training_data.drop_duplicates(subset=['item_code', 'item_text'], keep='last', inplace=True)

        # Save updated training data
        self.training_data.to_csv(self.training_data_path, index=False)
        
        # Retrain model if we have new verified data
        if any(item.category != "Uncategorized" for item in results):
            self.train_model(results)
        
        # Recalculate category-wide totals and tax percentages
        category_totals, category_tax_percents = self._summarize_cost_by_category_and_tax(results)

        # Calculate tax amounts per category
        category_tax_amounts = {}
        if receipt_details.tax > 0:
            for category, tax_percent in category_tax_percents.items():
                category_tax_amounts[category] = (tax_percent / 100.0) * receipt_details.tax

        # Update the ReceiptDetails object
        receipt_details.items = results
        receipt_details.category_totals = category_totals
        receipt_details.category_tax_percentages = category_tax_percents
        receipt_details.category_tax_amounts = category_tax_amounts
        
        return receipt_details
    
    def save_model(self) -> None:
        """Save the trained model"""
        if not self.vectorizer or not self.classifier:
            return
        
        with open(f"{self.model_path}_vectorizer.pkl", 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        joblib.dump(self.classifier, f"{self.model_path}_classifier.pkl")
    
    def load_model(self) -> bool:
        """
        Load the trained model.

        Returns:
            True if the model was successfully loaded, False otherwise.
        """
        try:
            with open(f"{self.model_path}_vectorizer.pkl", 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            self.classifier = joblib.load(f"{self.model_path}_classifier.pkl")
            return True
        except:
            print("Could not load model. Will train a new one.")
            return False
    
    def analyze_data(self) -> Dict[str, Any]:
        """
        Analyze the training data.

        Returns:
            A dictionary containing analysis results.
        """
        if len(self.training_data) == 0:
            return "No data available for analysis."
        
        # Calculate statistics
        total_items = len(self.training_data)
        verified_items = sum(self.training_data['verified'])
        category_counts = self.training_data['category'].value_counts()
        
        # Get most common items
        common_items = self.training_data['item_text'].value_counts().head(10)
        
        # Get items with low confidence
        low_confidence = self.training_data[self.training_data['confidence'] < self.confidence_threshold]
        
        return {
            'total_items': total_items,
            'verified_items': verified_items,
            'verification_rate': verified_items / total_items if total_items > 0 else 0,
            'category_distribution': category_counts.to_dict(),
            'common_items': common_items.to_dict(),
            'items_needing_verification': len(low_confidence)
        }
    
    def _calculate_taxable_amounts(self, items: List[Item], tax_total: float) -> Dict[str, float]:
        """
        Calculate tax amount for each item based on their taxable status.

        Args:
            items: List of Item objects
            tax_total: Total tax amount from receipt

        Returns:
            Dictionary with item codes as keys and their tax amounts as values
        """
        total_taxable = sum(item.price for item in items if item.is_taxable)
        tax_amounts = {}
        
        if total_taxable > 0:
            tax_rate = tax_total / total_taxable
            for item in items:
                if item.is_taxable:
                    tax_amounts[item.code] = item.price * tax_rate
                else:
                    tax_amounts[item.code] = 0.0
        else:
            # If no taxable items, set all tax amounts to 0
            tax_amounts = {item.code: 0.0 for item in items}
            
        return tax_amounts

    def _summarize_cost_by_category_and_tax(self, items: List[Item]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Summarize the total cost and tax by categories.

        Args:
            items: List of Item objects.

        Returns:
            A tuple containing:
            - A dictionary with categories as keys and total costs as values.
            - A dictionary with categories as keys and tax percentages as values.
        """
        category_summary = {}
        category_taxable_amounts = {}
        
        # Calculate totals per category and track taxable amounts
        for item in items:
            if item.category:
                # Add to category total
                category_summary[item.category] = category_summary.get(item.category, 0) + item.price
                # Track taxable amounts per category
                if item.is_taxable:
                    category_taxable_amounts[item.category] = category_taxable_amounts.get(item.category, 0) + item.price

        # Calculate tax percentages based on taxable amounts
        total_taxable = sum(category_taxable_amounts.values())
        category_tax_percentages = {}
        
        for category in category_summary.keys():
            if total_taxable > 0:
                category_tax_percentages[category] = (category_taxable_amounts.get(category, 0) / total_taxable) * 100
            else:
                category_tax_percentages[category] = 0
        
        return category_summary, category_tax_percentages

    def verify_receipt_total(self, receipt_details: ReceiptDetails) -> bool:
        """
        Verify that the subtotal of the receipt matches the total summary per category.

        Args:
            receipt_details: The ReceiptDetails object containing the receipt data.

        Returns:
            A boolean indicating whether the subtotal matches the total summary.
        """
        category_summary = receipt_details.category_totals
        total_from_categories = sum(category_summary.values())
        return abs(total_from_categories - receipt_details.subtotal) < 0.01  # Allow small floating-point differences

# Example usage
if __name__ == "__main__":
    agent = ReceiptAgent()
    
    # Sample receipt items
    sample_items = [
        Item(name="KS ORG EGGS", code="12345"),
        Item(name="PAPER TOWEL", code="67890"),
        Item(name="ROTIS CHKN", code="54321"),
        Item(name="KS BATH TISSUE", code="98765"),
        Item(name="APPLE IPAD", code="11223"),
        Item(name="BLK PEPPER", code="44556")
    ]
    
    # Process receipt
    receipt_details = ReceiptDetails(items=sample_items, receipt_path="path/to/receipt", subtotal=100.0)
    updated_receipt_details = agent.process_receipt(receipt_details)
    
    # Print results
    for item in updated_receipt_details.items:
        print(f"{item.name} (Code: {item.code}): {item.category}")
    
    # Analyze data
    analysis = agent.analyze_data()
    print("\nData Analysis:")
    for key, value in analysis.items():
        print(f"{key}: {value}")