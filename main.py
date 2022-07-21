from transformers import pipeline
import pandas as pd

from get_sentiment import get_sentiment

# read the survey response
# Change the name in the function pd.read_csv() to change the file to read
survey = pd.read_csv("Heidelberg_response.csv")
survey = survey[2:].reset_index()

# sentiment analysis pipeline
classifier = pipeline("sentiment-analysis")

# define which questions need to do sentiment analysis
# In this example question 5 and question 20 are used to do sentiment analysis
ids = ["Q5", "Q20"]

# do sentiment analysis and bind results with the original dataset
survey_sentiments = get_sentiment(survey, question_ids=ids, classifier=classifier)

# export to csv file. Can further edit the file and conduct analysis in excel
survey_sentiments.to_csv("sentiments.csv")
