# Road Closure Location Extraction from Twitter
## Detecting & Mapping Live Road Closures Using NLP, Named Entity Recognition, and Supervised Modeling Techniques

- Temple Moore
- Nathan Jacques
- David Trichter

## Problem Statement
Leveraging social media such as Twitter, we want to identify real time road closures, damaged roads, traffic congestion, flooding, and other blocked routes that may affect travel time, travel safety, and accessibility to emergency response crews.


## Overview
During a disaster scenario, GPS Information Systems (GIS) can be a useful tool to navigating around road closures and traffic. However, GIS systems can be slow to update and may require proprietary authorization to plot road closures. Fortunately, social media can provide real-time and accurate updates on road closures.  We aimed to create a reliable platform that verifies, consolidates, and _maps_ road closures as they happen through text scraped from social media platforms. Through acquiring credible data about road closures from blue-checkmarked 511 Twitter accounts, this project implements supervised modeling techniques to map live closures. We wanted to use real Twitter data from a real disaster, so we collected historical data from five states' 511 twitter accounts, specifically during Hurricane Matthew in 2016. We then focused on generating reliable results for Tweets pertaining to the Jacksonville, FL metro region, which was severely impacted by the storm.

## Methodology
The following is how we achieved our goal of collecting, modeling, and mapping road closures from social media.
- **Step 1: Data Acquisition**: We use the Tweepy and GetOldTweets modules to access the Twitter API. We used the Twitter API to get Tweets from blue-checkmarked 511 Twitter accounts from five different states: Virginia, North Carolina, South Carolina, Georgia, and Florida. Since our focus was looking at Twitter during Hurricane Matthew, we historical tweets from October 4, 2016 to October 14, 2016. We also used a curated list of traffic and weather accounts, including Florida 511, to draw from to get real-time tweets to use as unseen data and for demonstration purposes.
- **Step 2: Exploratory Data Analysis**: Using a Count Vectorizer, we observed the frequency of words in the collected tweets. We then used a keyword classification function to classify Tweets as `1` for road closed  if any of the selected keywords were found in the text, and `0` if the tweet did meet the classification criteria. The compiled list of keywords was refined over time after the assessing the performance of the each iteration of the filter.
- **Step 3: Pre-Processing and Modeling**: After the Tweets have been cleaned with RegEx and classified by keywords, we trained two models on the set of 24,084 Tweets and evaluated the performance. Each model was tuned on a grid of hyperparameters, and the the best models were pickled for use on unseen data.
  - **Logistic Regression Classifier**: Using Count Vectorized words, this model scored an accuracy score of 99.92% on the training set, and 99.90% on the testing set. The ROC AUC of this model was 0.9872, the Sensitivity on the testing set was 98.95%, the Specificity was 98.49%.
  - **Gradient Boosting Classifier**: Using TF-IDF Vectorized words, this model scored an accuracy score of 99.92% on the training set, and 99.90% on the testing set. The ROC AUC of this model was 0.9872, the Sensitivity on the testing set was 99.81%, the Specificity was 98.94%.
  - **Modeling On Unseen Data**: Using the models trained on historical tweets, we evaluated their performance on tweets taken real time. The Logistic Regression Model had an accuracy of 93.33% on testing data, a Sensitivity of 95.65%, and a Specificity of 91.89%. The Gradient Boosted Model had an accuracy of 96.67% on testing data, a Sensitivity of 100%, and a Specificity of 94.59%.
- **Step 4: SpaCy Location Extraction**: Using the SpaCy module, we used Named Entity Recognition (NER) Functionality to pull important features form the Tweet text. We first had to format the Tweets to be friendly to SpaCy's algorithm. Then, using the information extracted, we compared the tweets and the extracted locations to a dataset of interstate exits and cross streets to extract the GPS coordinates from known entities. This information was added to the Tweet dataset to help build queries for mapping purposes.
- **Step 5: Mapping**: Using the locations extracted from the Tweets, we plot the known coordinates using the Google Maps API. If we do not have the coordinates, we build a search query from information extracted during the NER process using the Here.com API. We build the query from information like: Interstate, Exit Number, Cross Street, and the location entities. The coordinates are plotted alongside the text of the Tweet and the time it was posted.

## Results
We were successfully able to detect road closures based on the content of the Tweet, and plot them using the extracted features.  To detect tweets, we implemented a keyword filter and as supervised model. All supervised models scored very well on the testing sets, and were able to minimize negatives in each model, showing Sensitivity scores of greater than 90%. Then, after searching the collected interstate exit data, we attempt to match Tweets to known coordinates. SpaCy was able to give us effective queries in the absence of known GPS coordinates, so we use the entity-tagged words to build a query for the Here.com API to geolocate. Then, we were successfully able to plot the tweets with coordinates on a map through the Google Maps API.

## End User Information
- Any user of the code in this repository will need a valid Twitter API key, HERE.com API key, and Google Maps API key. Included in this repository are sample credential files to use for your own API keys. Simply remove the "sample" from each file name to ensure compatibility with the notebooks.
- We are still working on deploying a python script that runs all the code and outputs a Google Maps object as an HTML file. This file is store in the "scripts" directory, but it currently not functional.
- All notebooks can be run and will be compatible with each other. You may need to change the names of the output files for better results, especially if you take in real-time tweets. The notebooks are numbered in order that they should be run for reproducibility.
