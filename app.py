import streamlit as st
from models.spam_detector import SMSSpamDetector

def load_model():
    detector = SMSSpamDetector()
    detector.train_model('data/spam_dataset.csv')
    return detector

def main():
    st.title("SMS Spam Detection System")
    
    # Initialize model
    if 'detector' not in st.session_state:
        with st.spinner('Loading spam detection model...'):
            st.session_state.detector = load_model()
    
    # Create input area
    message = st.text_area("Enter your message:", height=100)
    
    if st.button("Check for Spam"):
        if message:
            # Get prediction
            result = st.session_state.detector.predict_message(message)
            
            # Display result
            col1, col2 = st.columns(2)
            
            with col1:
                if result['is_spam']:
                    st.error("🚨 This message appears to be SPAM!")
                else:
                    st.success("✉️ This message appears to be legitimate")
                
                st.info(f"Confidence: {result['confidence']:.2%}")
            
            # Show keywords found
            with col2:
                st.subheader("Keywords Detected")
                if result['top_keywords']:
                    st.write("Key terms found in message:")
                    for word in result['top_keywords']:
                        st.write(f"- {word}")
                else:
                    st.write("No significant keywords found")
        else:
            st.warning("Please enter a message to analyze")

if __name__ == "__main__":
    main()