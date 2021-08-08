import sys
import csv
import tweepy
import datetime
import pandas as pd
import numpy as np
import tweepy
import datetime
import json
import nltk  # for sentiment analysis
import openpyxl
# Sentiment analysis packages
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
import re, string, random


# Twitter API credentials
consumer_key = "23KwbMhQH3fYpsQAVWI1msnGW"
consumer_secret = "ztQkdubvoEmBdZDpGoYDiy3pE91QSBYvYVxIUUQjaKw5zSd1Yq"
access_key = "914576029783609344-pFEnUPNHaL9qfa9aPsk1SVlzN6CGqwv"
access_secret = "trZyHc6YgJWGfRcLttBQdw4Ibwfiwx5Ahfl2uZRfo4G0L"

# http://tweepy.readthedocs.org/en/v3.1.0/getting_started.html#api
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)

today = datetime.date.today()
yesterday= today - datetime.timedelta(days=1)

twitter_handle = input("Provide a twitter handle : ")
tweets_list = tweepy.Cursor(api.user_timeline, id=twitter_handle, tweet_mode='extended', lang='en').items()
user = api.get_user(twitter_handle)


tweet_count = 0
output = []
output_tweets_tokens = []
output_hashtag = []
output_mentions = []

print("Running...")


def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token


def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


if __name__ == "__main__":

    print("building model for sentiment analysis...")
    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')
    text = twitter_samples.strings('tweets.20150430-223406.json')
    tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]

    stop_words = stopwords.words('english')

    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    all_pos_words = get_all_words(positive_cleaned_tokens_list)

    freq_dist_pos = FreqDist(all_pos_words)

    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

    positive_dataset = [(tweet_dict, "Positive") for tweet_dict in positive_tokens_for_model]

    negative_dataset = [(tweet_dict, "Negative") for tweet_dict in negative_tokens_for_model]

    dataset = positive_dataset + negative_dataset

    random.shuffle(dataset)

    train_data = dataset[:7000]
    test_data = dataset[7000:]

    classifier = NaiveBayesClassifier.train(train_data)

    print("Accuracy is:", classify.accuracy(classifier, test_data))


def get_sentiment(text):
    custom_tokens = remove_noise(word_tokenize(text))
    sentiment = classifier.classify(dict([token, True] for token in custom_tokens))
    return sentiment


try:
    print("Pulling tweet data...")
    for tweet in tweets_list:
        tweet_count += 1
        hashtag_tweet_id_list = []
        mention_tweet_id_list = []
        tweet_hashtags_str = ""
        tweet_mentions_str = ""

        # for more tweet attributes, visit https://developer.twitter.com/en/docs/twitter-api/v1/data-dictionary/object-model/tweet
        # Tweet data
        id = tweet.id_str  # The string representation of the unique identifier for this Tweet.
        text = tweet._json["full_text"]  # The actual UTF-8 text of the status update.
        favourite_count = tweet.favorite_count
        retweet_count = tweet.retweet_count
        # reply_count = tweet.reply_count
        # quote_count = tweet.quote_count
        created_at = tweet.created_at
        tweet_sentiment = get_sentiment(text)

        is_retweet = False

        if 'RT' in text:
            is_retweet = True

        source = tweet.source  # Utility used to post the Tweet, as an HTML-formatted string. Tweets from the Twitter website have a source value of web.
        truncated = tweet.truncated  # Indicates whether the value of the text parameter was truncated, for example, as a result of a retweet exceeding the original Tweet text length limit of 140 characters. Truncated text will end in ellipsis, like this ... Since Twitter now rejects long Tweets vs truncating them, the large majority of Tweets will have this set to false . Note that while native retweets may have their toplevel text property shortened, the original text will be available under the retweeted_status object and the truncated parameter will be set to the value of the original status (in most cases, false ).
        in_reply_to_status_id = tweet.in_reply_to_status_id_str  # Nullable. If the represented Tweet is a reply, this field will contain the string representation of the original Tweet’s ID.
        in_reply_to_user_id = tweet.in_reply_to_user_id_str  # Nullable. If the represented Tweet is a reply, this field will contain the string representation of the original Tweet’s author ID. This will not necessarily always be the user directly mentioned in the Tweet.

        # User data
        user_id = user.id_str
        user_name = user.name
        user_screen_name = user.screen_name
        user_location = user.location
        user_description = user.description
        user_protected = user.protected
        user_verified = user.verified
        user_followers_count = user.followers_count
        user_friends_count = user.friends_count
        user_listed_count = user.listed_count
        user_favourites_count = user.favourites_count
        user_statuses_count = user.statuses_count
        user_created_at = user.created_at
        user_profile_banner_url = user.profile_banner_url
        user_profile_image_url_https = user.profile_image_url_https
        user_withheld_in_countries = user.withheld_in_countries

        hashtags = tweet._json["entities"]["hashtags"]
        mentions = tweet._json["entities"]["user_mentions"]
        place = tweet._json["place"]
        coordinates = tweet._json["coordinates"]

        # Nulled by default in case nothing comes back from endpoint
        tweet_place_id = None
        tweet_place_url = None
        tweet_place_type = None
        tweet_place_name = None
        tweet_place_full_name = None
        tweet_place_country_code = None
        tweet_place_country = None
        tweet_place_contained_within = None
        tweet_coordinates_latitude = None
        tweet_coordinates_longitude = None
        tweet_coordinates_type = None

        # Loops through every hashtag per tweet
        if len(hashtags) > 0:
            for hashtag in hashtags:
                h = hashtag['text']
                hashtag_tweet_id_list.append(id)
                hashtag_list = {'Tweet ID': id, 'Hashtag': h}
                output_hashtag.append(hashtag_list)

                # For Tweet Dataset
                tweet_hashtags_str = tweet_hashtags_str + h + ' '

        # Loops through every mention per tweet
        if place is not None:
            tweet_place_id = place['id']
            tweet_place_url = place['url']
            tweet_place_type = place['place_type']
            tweet_place_name = place['name']
            tweet_place_full_name = place['full_name']
            tweet_place_country_code = place['country_code']
            tweet_place_country = place['country']
            tweet_place_contained_within = place['contained_within']

        if coordinates is not None:
            tweet_coordinates_latitude = coordinates['coordinates'][0]
            tweet_coordinates_longitude = coordinates['coordinates'][1]
            tweet_coordinates_type = coordinates['type']

        # Loops through every mention per tweet
        if len(mentions) > 0:
            for mention in mentions:
                m = mention['screen_name']
                mention_tweet_id_list.append(id)
                mention_list = {'Tweet ID': id, 'Mention': m}
                output_mentions.append(mention_list)

                # For Tweet Dataset
                tweet_mentions_str = tweet_mentions_str + m + ' '

        line = {
            # Tweet Data
            'Tweet ID': id,
            'Text': text,
            'Hashtags': tweet_hashtags_str,
            'Users Mentioned': tweet_mentions_str,
            'Favourite Count': favourite_count,
            'Retweet Count': retweet_count,
            'Created At': created_at,
            'Text Sentiment': tweet_sentiment,
            'Is a retweet?': is_retweet,
            'Device': source,
            'Truncated': truncated,
            'In Reply To Status ID': in_reply_to_status_id,
            'In Reply To User ID': in_reply_to_user_id,
            'Coordinates Latitude': tweet_coordinates_latitude,
            'Coordinates Longitude': tweet_coordinates_longitude,
            'Coordinates Type': tweet_coordinates_type,
            'Place ID': tweet_place_id,
            'Place URl': tweet_place_url,
            'Place Type': tweet_place_type,
            'Place Name': tweet_place_name,
            'Place Full Name': tweet_place_full_name,
            'Place Country Code': tweet_place_country_code,
            'Place Country': tweet_place_country,
            'Place Contained Within': tweet_place_contained_within,
            'User ID': user_id,
            'User Name': user_name,
            'User Screen Name': user_screen_name,
            'User Account Location': user_location,
            'User Description': user_description,
            'User Protected': user_protected,
            'User Verified': user_verified,
            'User Followers Count': user_followers_count,
            'User Friend Count': user_friends_count,
            'User Listed Count': user_listed_count,
            'User Favourites Count': user_favourites_count,
            'User Statuses Count': user_statuses_count,
            'User Creation Date': user_created_at,
            'User Profile Banner URL': user_profile_banner_url,
            'User Profile Image URL': user_profile_image_url_https,
            'User Banned in Countries': user_withheld_in_countries,
            'Date Data Was Pulled': today
        }

        tweet_tokens = remove_noise(word_tokenize(text))
        if is_retweet is False:
            for token in tweet_tokens:
                word = {'Word': token}
                output_tweets_tokens.append(word)

        output.append(line)
        tweet_hashtags_str = ""
        tweet_mentioned_str = ""

    print("Tweets Pulled : " + str(tweet_count))

except exception as e:
    print(e)
    close()


# Populating Dataframes
df = pd.DataFrame(output)
df_hashtags = pd.DataFrame(output_hashtag)
df_mentions = pd.DataFrame(output_mentions)
df_tokens = pd.DataFrame(output_tweets_tokens)

print("Saving to file...")
writer = pd.ExcelWriter(twitter_handle+'_tweet_data.xlsx', engine = 'xlsxwriter')
df.to_excel(writer, sheet_name='By Tweet', index=False)
df_hashtags.to_excel(writer, sheet_name='By Hashtag', index=False)
df_mentions.to_excel(writer, sheet_name='By Mention', index=False)
df_tokens.to_excel(writer, sheet_name='Used Words (Cleaned)', index=False)
writer.save()
print("Saved")


