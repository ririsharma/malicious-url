"""
----- Script File -----
Contains support functions for UI.
Feature extracting functions of the URL.
"""

# Importing ML libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Import additional libraries
import re
from urllib.parse import urlparse
from tld import get_tld
import pandas as pd


#Dump
import pickle


def lengthFeaturesExtract(inputData : pd.DataFrame) -> pd.DataFrame:
    # Extracting all the length features and saving them in the
    # dataset as new features.
    # Length of URL
    inputData['url_length'] = inputData['url'].apply(lambda i: len(str(i)))

    # Hostname Length
    inputData['hostname_length'] = inputData['url'].apply(lambda i: len(urlparse(i).netloc))

    # Path Length
    inputData['path_length'] = inputData['url'].apply(lambda i: len(urlparse(i).path))

    # First Directory Length
    def firstDirLength(url):
        urlpath= urlparse(url).path
        try:
            return len(urlpath.split('/')[1])
        except:
            return 0

    inputData['fd_length'] = inputData['url'].apply(lambda i: firstDirLength(i))

    # Length of Top Level Domain. Extracted using TLD library
    inputData['tld'] = inputData['url'].apply(lambda i: get_tld(i,fail_silently=True))
    def tld_length(tld):
        try:
            return len(tld)
        except:
            return -1

    inputData['tld_length'] = inputData['tld'].apply(lambda i: tld_length(i))
    inputData = inputData.drop('tld', axis=1)   # Removing the unwanted feature used for count

    return inputData


def countFeaturesExtract(inputData : pd.DataFrame) -> pd.DataFrame:
    # Extracting all the "count features" and saving them as new feautres
    # in the dataset.
    inputData['numberOf-'] = inputData['url'].apply(lambda i: i.count('-'))
    inputData['numberOf@'] = inputData['url'].apply(lambda i: i.count('@'))
    inputData['numberOf?'] = inputData['url'].apply(lambda i: i.count('?'))
    inputData['numberOf%'] = inputData['url'].apply(lambda i: i.count('%'))
    inputData['numberOf.'] = inputData['url'].apply(lambda i: i.count('.'))
    inputData['numberOf='] = inputData['url'].apply(lambda i: i.count('='))
    inputData['numberOfhttp'] = inputData['url'].apply(lambda i : i.count('http'))
    inputData['numberOfhttps'] = inputData['url'].apply(lambda i : i.count('https'))
    inputData['numberOfwww'] = inputData['url'].apply(lambda i: i.count('www'))

    def digit_count(url):
        digits = 0
        for i in url:
            if i.isnumeric():
                digits = digits + 1
        return digits
    inputData['numberOfdigits']= inputData['url'].apply(lambda i: digit_count(i))

    def letter_count(url):
        letters = 0
        for i in url:
            if i.isalpha():
                letters = letters + 1
        return letters
    inputData['numberOfletters']= inputData['url'].apply(lambda i: letter_count(i))

    def no_of_dir(url):
        urldir = urlparse(url).path
        return urldir.count('/')
    inputData['numberOfdir'] = inputData['url'].apply(lambda i: no_of_dir(i))

    return inputData


def binaryFeaturesExtract(inputData : pd.DataFrame) -> pd.DataFrame:
    # Checking the use of IP in domain
    def havingIP(url):
        match = re.search(
            '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
            '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
            '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
            '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
        if match:
            return -1
        else:
            return 1
        
    inputData['use_of_ip'] = inputData['url'].apply(lambda i: havingIP(i))

    # Cheking wether the URL used a shortening service or not
    def shorteningService(url):
        match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                        'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                        'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                        'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                        'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                        'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                        'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                        'tr\.im|link\.zip\.net',
                        url)
        if match:
            return -1
        else:
            return 1

    inputData['short_url'] = inputData['url'].apply(lambda i: shorteningService(i))
    return inputData


def featureExtract(inputData : pd.DataFrame) -> pd.DataFrame:
    
    lengthFeatures = lengthFeaturesExtract(inputData)
    countFeatures = countFeaturesExtract(lengthFeatures)
    binFeatures = binaryFeaturesExtract(countFeatures)
    finalData = binFeatures.drop(columns=['short_url', 'url_length'])
    return finalData

# if __name__ == "__main__":
    
#     inputURL = {"url" : ["https://www.google.com"]}

#     inputData = pd.DataFrame.from_dict(inputURL)
#     inputData = featureExtract(inputData)
#     inputData = inputData.iloc[:, 1:]
#     model = pickle.load(open("URLModel.pkl", 'rb'))

#     outu = model.predict(inputData)
#     print(outu[0])