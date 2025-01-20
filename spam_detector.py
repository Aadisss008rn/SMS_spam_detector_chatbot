import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

class SMSSpamDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,          
            stop_words='english', 
            ngram_range=(1, 2) 
        )
        self.model = MultinomialNB(alpha=1.0)
        self.trained = False
    
    def load_data(self, csv_path):
        try:
            # Read the CSV file with proper encoding
            df = pd.read_csv(csv_path, encoding='latin-1')
            
            # If the CSV has different column names, adjust accordingly
            if 'v1' in df.columns and 'v2' in df.columns:
                df = df.rename(columns={'v1': 'label', 'v2': 'message'})
            
            # Convert labels to lowercase and ensure they're either 'spam' or 'ham'
            df['label'] = df['label'].str.lower()
            if not all(df['label'].isin(['spam', 'ham'])):
                raise ValueError("Labels must be either 'spam' or 'ham'")
            
            # Basic data cleaning
            df['message'] = df['message'].astype(str)
            df['message'] = df['message'].str.strip()
            
            # Remove empty messages
            df = df[df['message'].str.len() > 0]
            
            print(f"Dataset Statistics:")
            print(f"Total messages: {len(df)}")
            print(f"Spam messages: {len(df[df['label'] == 'spam'])}")
            print(f"Ham messages: {len(df[df['label'] == 'ham'])}")
            print(f"Average message length: {df['message'].str.len().mean():.1f} characters")
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def train_model(self, csv_path, test_size=0.2):
        """Train the spam detection model using CSV data"""
        # Load data from CSV
        df = self.load_data(csv_path)
        
        if df is None:
            raise Exception("Failed to load dataset")
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            df['message'],
            (df['label'] == 'spam').astype(int),
            test_size=test_size,
            random_state=42
        )
        
        # Transform the messages into vectors
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_vectorized, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test_vectorized)
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))
        
        # Store vocabulary size
        self.vocab_size = len(self.vectorizer.vocabulary_)
        self.trained = True
        
        return self
    
    def predict_message(self, message):
        """Predict whether a message is spam or not"""
        if not self.trained:
            raise Exception("Model needs to be trained first!")
        
        # Vectorize the message
        vectorized_message = self.vectorizer.transform([message])
        
        # Make prediction
        prediction = self.model.predict(vectorized_message)[0]
        probabilities = self.model.predict_proba(vectorized_message)[0]
        
        # Get feature importance for keywords
        feature_importance = self.get_important_features(message)
        
        return {
            'is_spam': bool(prediction),
            'confidence': float(probabilities[1] if prediction else probabilities[0]),
            'spam_probability': float(probabilities[1]),
            'top_keywords': feature_importance,
            'message_length': len(message)
        }
    
    def get_important_features(self, message):
        """Get the most important features (keywords) for the prediction"""
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Transform message
        message_vector = self.vectorizer.transform([message])
        
        # Get feature importance scores
        feature_importance = np.multiply(message_vector.toarray()[0], self.model.feature_log_prob_[1])
        
        # Get top features
        top_indices = feature_importance.argsort()[-5:][::-1]
        return [feature_names[i] for i in top_indices if feature_importance[i] > 0]

def run_example():
    # Initialize and train the detector
    detector = SMSSpamDetector()
    detector.train_model('data/spam_dataset.csv')
    
    # Test with sample messages
    test_messages = [
        "WINNER!! Claim your 90,000 prize reward now! Call 0909088840 for details.",
        "Hey, can we meet at 3pm for coffee?",
        "FREE UNLIMITED MOBILE CONTENT! Download now! Text YES to 80082 to get access!",
        "Don't forget to bring the documents for tomorrow's meeting.",
        "Congratulations! You've been selected for a free iPhone! Click here to claim: http://bit.ly/claim-prize/aditya"
    ]
    
    print("\nTesting Sample Messages:")
    for msg in test_messages:
        result = detector.predict_message(msg)
        print(f"\nMessage: {msg}")
        print(f"Is Spam: {result['is_spam']}")
        print(f"Spam Probability: {result['spam_probability']:.2%}")
        print(f"Key words: {', '.join(result['top_keywords'])}")

if __name__ == "__main__":
    run_example()