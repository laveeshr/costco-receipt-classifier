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
from src.models.item import Item
from src.models.receipt_details import ReceiptDetails

class ReceiptAgent:
    def __init__(self, data_dir="./receipt_data"):
        self.data_dir = data_dir
        self.training_data_path = os.path.join(data_dir, "training_data.csv")
        self.model_path = os.path.join(data_dir, "model")
        self.confidence_threshold = 0.7  # Threshold for automatic classification
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
        
    def preprocess(self, item_name):
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
    
    def train_model(self, items, force=False):
        """
        Train or retrain the model using an array of Item objects.
        
        Args:
            items: List of Item objects to use for training.
            force: Whether to force training even with insufficient data.
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

        # Check if we have enough data
        if len(train_df) < 10 and not force:
            print("Not enough verified data to train model.")
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
        print(classification_report(y_test, y_pred))

        # Save model
        self.save_model()

        return True
    
    def predict(self, item):
        """Predict category for an Item object using item.code or item.name."""
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
    
    def process_receipt(self, receipt_details: ReceiptDetails, user_feedback=True):
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
            
            # Dynamic category suggestions
            suggested_categories = self.training_data['category'].value_counts().head(5).index.tolist()
            print(f"Suggested categories: {', '.join(suggested_categories)}")

            # Get user feedback if needed
            if user_feedback and not verified:
                print(f"\nItem: {item.name} (Code: {item.code})")
                print(f"Predicted category: {category} (confidence: {confidence:.2f})")
                print("Suggested categories: Groceries, Household, Electronics, Clothing, Furniture/Decor, Health/Personal Care, Auto, Pet")
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
        
        # Update the ReceiptDetails object
        receipt_details.items = results
        return receipt_details
    
    def save_model(self):
        """Save the trained model"""
        if not self.vectorizer or not self.classifier:
            return
        
        with open(f"{self.model_path}_vectorizer.pkl", 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        joblib.dump(self.classifier, f"{self.model_path}_classifier.pkl")
    
    def load_model(self):
        """Load the trained model"""
        try:
            with open(f"{self.model_path}_vectorizer.pkl", 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            self.classifier = joblib.load(f"{self.model_path}_classifier.pkl")
            return True
        except:
            print("Could not load model. Will train a new one.")
            return False
    
    def analyze_data(self):
        """Analyze the training data"""
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
    
    def summarize_cost_by_category(self, items):
        """
        Summarize the total cost by categories from a list of Item objects.

        Args:
            items: List of Item objects.

        Returns:
            A dictionary with categories as keys and total costs as values.
        """
        category_summary = {}
        for item in items:
            if item.category:
                category_summary[item.category] = category_summary.get(item.category, 0) + item.price
        return category_summary

    def verify_receipt_total(self, items, receipt_subtotal):
        """
        Verify that the subtotal of the receipt matches the total summary per category.

        Args:
            items: List of Item objects.
            receipt_subtotal: The subtotal extracted from the receipt.

        Returns:
            A boolean indicating whether the subtotal matches the total summary.
        """
        category_summary = self.summarize_cost_by_category(items)
        total_from_categories = sum(category_summary.values())
        return abs(total_from_categories - receipt_subtotal) < 0.01  # Allow small floating-point differences

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