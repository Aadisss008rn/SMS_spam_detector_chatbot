# SMS Spam Detection System

A machine learning-powered spam detection system that uses Natural Language Processing (NLP) and the Multinomial Naive Bayes algorithm to identify spam messages in real-time.

## 🚀 Features

- **Real-time Spam Detection**: Instantly analyze messages for spam content
- **Machine Learning Based**: Uses TF-IDF vectorization and Naive Bayes classification
- **Keyword Analysis**: Identifies and displays key spam indicators
- **Confidence Scoring**: Provides probability scores for classifications
- **User-Friendly Interface**: Clean and intuitive Streamlit web interface

## 📋 Prerequisites

- Python 3.7+
- pip (Python package installer)

## 🛠️ Installation

1. Clone the repository,

2. Install required packages:
```bash
pip install -r requirements.txt
```

## 💻 Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to local host.

3. Enter a message in the text area and click "Check for Spam"

## 📁 Project Structure

```
sms-spam-detector/
├── data/
│   └── spam_dataset.csv      # Training dataset
├── models/
│   └── spam_detector.py      # Spam detection model implementation
├── app.py                    # Streamlit web application
└── requirements.txt          # Project dependencies
```

## 🤖 Model Details

- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Classification**: Multinomial Naive Bayes
- **Features**: 
  - Text preprocessing
  - Stop word removal
  - Feature importance analysis
  - Confidence scoring

## 📊 Performance

The model achieves:
- High accuracy in spam detection
- Low false positive rate
- Real-time processing capabilities
- Robust keyword identification

## 🔧 Configuration

The system can be configured by modifying the following parameters:
- Model hyperparameters in `models/spam_detector.py`
- UI elements in `app.py`
- Dataset selection in the training process

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch 
3. Commit your changes 
4. Push to the branch
5. Open a Pull Request

## 👥 Authors

Name -  ADITYA RAJ & [GitHub Profile](https://github.com/Aadisss008rn)

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the SMS Spam Collection Dataset
- Streamlit team for the amazing web framework
- scikit-learn developers for the machine learning tools

## 📧 Contact

Mail - aadirajput6951@gmail.com
