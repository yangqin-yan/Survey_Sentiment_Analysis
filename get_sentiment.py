import pandas as pd

# Input: 
#       dataset: the survey responses
#       question_ids: the set of IDs of the question. For example, the
#                     set ["Q5","Q20"] contains 2 questions that are
#                     question 5 and question 20. We can see in this case
#                     question 20 has the ID "Q20". Question IDs can be seen
#                     in the column names of the dataset. 
#       classifier: the sentiment analysis model.
# Output:
#       new dataset with columns of sentiments and confidence scores.
# Usage:
#       get_sentiment(survey, "Q20", classifier)

def get_sentiment(dataset, question_ids, classifier):
    temp = dataset
    for ids in question_ids:
        sentiment_col = ids + "_sentiment"
        sentiment_score = ids + "_sscore"
        temp[sentiment_col] = 'UNKNOWN'
        temp[sentiment_score] = 0

        for i in range(len(dataset)):
            if pd.isna(dataset[ids].loc[i]):
                continue
            res = classifier(dataset[ids].loc[i])[0]
            temp[sentiment_col].loc[i] = res['label']
            temp[sentiment_score].loc[i] = res['score']

    return temp
    


