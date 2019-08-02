import pandas as pd
import tweepy
import json
import datetime
import re
import pickle
import spacy
import string

from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels
from sklearn.externals import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, roc_auc_score

import warnings
warnings.filterwarnings("ignore")

# Enter Access Identification numbers
with open('./twitter_credentials.json') as cred_data:
    info = json.load(cred_data)
    consumer_key = info['CONSUMER_KEY']
    consumer_secret = info['CONSUMER_SECRET']
    access_token = info['ACCESS_TOKEN']
    access_secret = info['ACCESS_SECRET']

# Authenticate Twitter API using  account Keys
auth_details = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth_details.set_access_token(access_token, access_secret)
my_accounts = []

# create api instance
api = tweepy.API(auth_details, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

# define fucntion to get tweets and add class functionality
def get_tweets(username, list_name = None, num = 200):

    # check for list
    if not list_name:
        # get timeline of user if no list
        tweet = tweepy.Cursor(api.user_timeline, screen_name = username).items(num)
    else:
        # get list timeline if list
        tweet = tweepy.Cursor(api.list_timeline, owner_screen_name = username, slug = list_name).items(num)

    # Empty Arrays
    tmp=[]
    closure_list = []

    # create array of tweet information: username,
    # tweet id, date/time, text
    # create dictionary for each tweet
    for status in tweet:
        tweet_dict = {}
        tweet_dict['id'] = status.id
        tweet_dict['username'] = status.user.screen_name
        tweet_dict['date'] = status.created_at
        tweet_dict['text'] = status.text
        tweet_dict['hashtags'] = status.entities['hashtags']
        tweet_dict['geo'] = status.coordinates
        tweet_dict['type'] = 'official'
        tmp.append(status.text)
        closure_list.append(tweet_dict)

    # return tweet_dict
    return closure_list

# Driver code
if __name__ == '__main__':

    # Here goes the twitter handle for the user and optional list
    # whose tweets are to be extracted.
    twts = get_tweets('ClydeLazersex', 'Evac-Route-RT', num = 200)

# create dataframe from tweets
tweets = pd.DataFrame(twts)

# define function to run regex startments over a column labeled "text"
def twt_preprocess(twt):
    # run regex to remove urls
    twt['text'] = twt['text'].map(lambda x: re.sub(r"((http|ftp|https):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?", ' ', x))

    # run regex to remove non alphanumeric characters
    twt['text'] = twt['text'].map(lambda x: re.sub(r"[@\?\.$%_\!\[\]()+:;\/*\"]", ' ', x, flags=re.I))
    twt['text'] = twt['text'].map(lambda x: re.sub(r"[,']", '', x, flags=re.I))

    # run regex to remove common words
    twt['text'] = twt['text'].map(lambda x: re.sub('(twitter|tweet)[s]?', ' ', x,  flags=re.I))

# run regex cleaner on full dataframe
twt_preprocess(tweets)

# establish lists of words to search for in dataframe

# list of words having to do with roads
road_keywords = ['road', 'st', 'rd', 'hwy', 'highway', 'ave', 'avenue',
                 'intersection', 'bridge', 'sr-', 'cr-', 'us-', 'i-', 'blvd']

# list of words associated with road closures
closed_keywords = ['closed', 'remains closed', 'shut down', 'backed up',
                   'no travel', 'delay', 'blocked', 'delays',
                   'disabled', 'traffic', 'fire', 'flood', 'closures', 'closure']

# list of words that would generate false positives
to_drop = ["open", "opened", "lifted", "reopened", "clear", "cleared"]

# define function to filter the full dataframe for tweets that contain words from a keyword list
# modified code from arielle miro
def tweet_filter (df, col, keywords, roads, dropwords):

    # create a new column in the given dataframe
    # assign a value of 1 if any word in the text is in the keyword and road lists
    # assign a value of 0 if any word in the test is in the drop words list
    df['road_closure'] = df[col].map(lambda x: 1 if ((any(word in x for word in roads))
                                                     & (any(word in x for word in keywords))
                                                     & (not any(word in x for word in dropwords)))
                                     else 0)
    return df['road_closure']

# backup original tweet to new dataframe
tweets['tweet'] = tweets['text']

# make tweet text lowercase
tweets['text'] = tweets['text'].str.lower()

# run function on full dataframe
tweets['road_closure'] = tweet_filter(tweets, 'text', closed_keywords, road_keywords, to_drop)
tweets['state'] = 'Florida'

# run gradient pickled boosting classifier
# load the model from disk

filename = './data/Models/gb_tvec_07302019.sav'
gb = joblib.load(filename)

# Create the predictions from pickled model
tweets['gb_pred'] = gb.predict(tweets['text'])

# create and format new dataframe tweets
rt_closures = tweets[['date', 'text', 'type', 'username', 'tweet', 'state', 'road_closure', 'gb_pred']]
rt_closures['modified_text'] = ''
rt_closures['location'] = ''

format_dict = {"hwy": "highway ",
            "blvd": "boulevard",
            " st": "street",
           "CR ": "County Road ",
           "SR ": "State Road",
           "I-": "Interstate ",
           "EB ": "Eastbound ",
           "WB ": "Westbound ",
           "SB ": "Southbound",
           "NB ": "Northbound",
           " on ": " at ",
           " E ": " East ",
           " W ": " West ",
           " S ": " South",
           " N ": " North",
           "mi ": "mile ",
           "between ": "at ",
           "Between ": "at ",
           " In ": " in",
           " in ": " at "}

def spacy_cleaner(df, col, word_dict):
    modified_text = "At " + df[col].replace(word_dict, regex=True)
    modified_text = modified_text.str.title()
    return modified_text

# run spacy formatter
rt_closures['modified_text'] = spacy_cleaner(rt_closures, 'tweet', format_dict)
rt_closures['date'] = pd.to_datetime(rt_closures['date'])
rt_loc_df = rt_closures[(rt_closures['road_closure'] == 1)]

def get_loc(df, text_column, location_column):

    # Use Spacy to extract location names from `text` column
    for i in range(len(df)):

        #instantiate spacy model
        nlp = spacy.load("en_core_web_sm")

        # create documewnt from modified text column
        doc = nlp(df[text_column].iloc[i])

        locations = set()

        # loop through every entity in the doc
        for ent in doc.ents:

            # find entities labelled as places
            if (ent.label_=='GPE') or (ent.label_=='FAC') or (ent.label_ == 'LOC'):

                # put locations in a set
                locations.add(ent.text)
                df[location_column].iloc[i] = locations

    return df[location_column]

rt_loc = get_loc(rt_loc_df, 'modified_text', 'location')
rt_loc_df['location'] = rt_loc

rt_loc_df.to_csv("./data/Loc_Extracted/rt_locations_sample_08022019.csv", index = False)
