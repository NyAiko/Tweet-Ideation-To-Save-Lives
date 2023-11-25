import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

def remove_spec(text):
    #remove punctuations and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    
    # Remove @usernames
    text = re.sub(r'@\w+', '', text)
    
    # Remove HTTP links
    text = re.sub(r'http\S+', '', text)
    
    # Remove extra whitespace and trim the text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Convert text to lowercase (optional)
    text = text.lower()
    return text

def remove_stopwords(text):
    # Tokenize the text into words
    words = nltk.word_tokenize(text)
    # Get the list of English stopwords
    stop_words = set(stopwords.words('english'))
    # Remove stopwords from the list of words
    filtered_words = [word for word in words if word not in stop_words]
    # Reconstruct the text from the filtered words
    filtered_text = ' '.join(filtered_words)
    return filtered_text

def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    lemmatized_text = ' '.join(lemmatized_words)
    return lemmatized_text

def stem(text):
    stemmer = PorterStemmer()
    words = nltk.word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in words]
    stemmed_text = ' '.join(stemmed_words)
    return stemmed_text

def normalize_text(text):
    text = str(text)
    text = remove_spec(text)
    text = remove_stopwords(text)
    text = stem(text)
    text = lemmatize(text)
    return text


