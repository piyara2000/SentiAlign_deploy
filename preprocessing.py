import pandas as pd
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK components (punkt_tab not required)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to clean and normalize text
def preprocess_text(text, remove_stopwords=True, do_lemmatize=True):
    if pd.isna(text):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|ftp\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove numbers and punctuation
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Simple tokenization
    tokens = text.split()

    # Remove stopwords
    if remove_stopwords:
        tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    if do_lemmatize:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)

# === Load dataset ===
df = pd.read_csv("bug_reports.csv", encoding='latin1')

# Check column names
print("Columns in dataset:", df.columns)

# Use the correct column name for bug descriptions
if 'issue_title' in df.columns:
    text_column = 'issue_title'
elif 'Short Description' in df.columns:
    text_column = 'Short Description'
else:
    raise KeyError("No column named 'issue_title' or 'Short Description' found.")

# Drop null or empty rows
df = df[df[text_column].notnull()]

# Apply preprocessing
df['cleaned_text'] = df[text_column].apply(preprocess_text)

# Save the cleaned dataset to a new CSV
output_path = "bug_reports_120k_cleaned.csv"
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"Preprocessed data saved to: {output_path}")

# Preview
print(df[[text_column, 'cleaned_text']].head())
