#Text Processing
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize.treebank import TreebankWordDetokenizer

def remove_punct(tweet):
    """
    Remove punctuation and other unwanted characters from a tweet.

    Args:
        tweet (str): A string containing the text of the tweet.

    Returns:
        str: A modified string with punctuation, numbers, hashtags, and URLs removed.
    """
    #removes any leading or trailing white space from the tweet.
    #tweet = tweet.strip()
    
    #replaces any non-alphanumeric characters (excluding spaces) with a space
    tweet = re.sub('[^a-zA-Z0-9 ]', ' ', str(tweet))
    
    #removes any digits from the tweet. This is done to remove any numbers that might not be relevant to the analysis.
    tweet = re.sub('[0-9]+', ' ', tweet)
    
    #removes the hash symbol (#) from the tweet. This is done to remove any hashtags that might be used in social media platforms.
    tweet = re.sub(r'#', '', str(tweet))  
    
    #removes any hyperlinks from the tweet. This is done to remove any links that might be present in the tweet.
    tweet = re.sub(r'http\S+', ' ', tweet)
    
    #replaces any multiple spaces or newline characters with a single space. This is done to make sure that there are no unnecessary spaces in the tweet.
    tweet = re.sub(r'\s+|\\n', ' ', tweet)
    
    #removes the text "RT" (which stands for "retweet") and any whitespace characters that follow it. This is done to remove any retweets that might be present in the tweet.
    tweet = re.sub('RT[\s]+', '', tweet) # Removing RT
    
    #removes any hyperlinks from the tweet. This is done to remove any links that might be present in the tweet.
    tweet = re.sub('https?:\/\/\S+', '', tweet) # Removing hyperlink

    return tweet

def case_folding(tweet):
    """
    Convert all characters in a tweet to lowercase using case-folding.

    Args:
        tweet (str): A string containing the text of the tweet.

    Returns:
        str: A modified string with all characters converted to lowercase.
    """
    tweet = tweet.casefold()
    
    return tweet

def tokenization(tweet):
    """
    Split a tweet into individual words, removing non-alphanumeric characters.

    Args:
        tweet (str): A string containing the text of the tweet.

    Returns:
        list: A list of individual words in the tweet, with non-alphanumeric characters removed.
    """
    tweet = re.split('\W+', tweet)
    
    return tweet

def normalisasi(tweet):
    """
    Normalize a tweet by replacing slang words with their corresponding standard words.

    Args:
        tweet (list): A list of words representing the text of the tweet.

    Returns:
        list: A modified list of words with slang words replaced by standard words, and all words converted to lowercase.
    """
    # Change dataframe to dictionary
    kamusalay_dict = dict(df_uploaded_kamusalay.values)

    # Compile a regular expression pattern for matching slang words in the tweet
    pattern = re.compile(r'\b(' + '|'.join(kamusalay_dict.keys()) + r')\b')

    # Replace slang words with their corresponding standard words in the tweet
    content = []
    for kata in tweet:
        filteredSlang = pattern.sub(lambda x: kamusalay_dict[x.group()], kata)
        content.append(filteredSlang.lower())

    # Convert all words to lowercase and return the modified list of words
    tweet = content
    return tweet

def stemming(tweet):
    """
    Perform stemming on a tweet by reducing each word to its base or root form using Sastrawi stemmer.

    Args:
        tweet (list): A list of words representing the text of the tweet.

    Returns:
        list: A modified list of words with each word reduced to its base or root form.
    """
    # Create a stemmer object using Sastrawi
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
    # Perform stemming on each word in the tweet
    #tweet = [stemmer.stem(word) for word in tweet]
    #tweet = stemmer.stem(tweet)

    # Return the modified list of words with stemming applied
    return stemmer.stem(tweet)

def remove_stopwords(tweet):
    """
    Remove stop words from a tweet.

    Args:
        tweet (list): A list of words representing the text of the tweet.

    Returns:
        list: A modified list of words with stop words removed.
    """
    # stopword is a list of commonly occurring words that are typically not useful for analysis
    stopword = nltk.corpus.stopwords.words('indonesian')
    
    # Remove stop words from the tweet
    tweet = [word for word in tweet if word not in stopword]

    # Return the modified list of words with stop words removed
    return tweet

def detokenize_text(tweet):
    """
    Detokenizes the given text using the TreebankWordDetokenizer from the nltk package.

    Parameters:
    text (str): The text to detokenize.

    Returns:
    str: The detokenized text.
    """
    # Detokenize the tokens
    detokenizer = TreebankWordDetokenizer()
    tweet = detokenizer.detokenize(tweet)

    return tweet

# Creating a new column sentiment based on overall ratings
def relabel_rating(rating, threshold):
    if rating > threshold:
        return 1
    else:
        return 0
