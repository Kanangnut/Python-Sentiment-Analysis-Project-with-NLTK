# Python-Sentiment-Analysis-Project-with-NLTK and classify Amazon Reviews

In case you run NLTK for the first time must be install:
  1.nltk.download('punkt') install
  2.nltk.download('averaged_perceptron_tagger')
  3.nltk.download('maxent_ne_chunker')
  4.nltk.download('words')
  5.nltk.download('vader_lexicon')

Note: This project working on Lenovo 2in1 ideapad miix 520, window 10. 

Start!!
For sentiment Analysis in Python of this project by using two different techniques:
1.VADER (Valence Aware Dictionary and Sentiment Reasoner)
2.Roberta Pretrained Model
3.Huggingface Pipeline

Step 0. Read in Data and NLTK (Natural Language Toolkit)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

plt.style.use('ggplot')

Link with the data CSV file:
df = pd.read_csv('../20230709-Python-Sentiment-Analysis-Project-with-NLTK/Reviews.csv')
df = df.head(500)
print(df.shape)
This step should run for check the result
Result show:
![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/d5db9774-3ee2-44d3-833d-7b5fcda13345)

Then check Table result by:
df.head()
![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/729a3695-1be5-44cd-bc59-569a6b8f8378)

Quick EDA
df['Score'].value_counts().sort_index().plot(kind='bar', title='Count of Reviews by Starts', figsize=(8, 5))
![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/b2785148-b983-4146-896e-4da5d78ee3f2)

![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/180794db-8757-47b4-8697-c2c7e70c11e6)

![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/f4340184-af8a-4e9b-b697-36e7af9caa5a)


NLTK NLTK for generate postag:
![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/e5423197-79c0-427a-a5a7-c4c90c5b9454)


NLTK for generate entitle:
![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/0b0d5f05-1467-4e4c-8376-f6893666e8a3)


Step 1. VADER Seniment Score
Use NLTK's SentimentIntensityAnalyzer to get the neg/neu/pos scores of the reviews.

VADER_lexicon for NLTK sentiment: <br>
from nltk.sentiment import SentimentIntensityAnalyzer<br>
from tqdm.notebook import tqdm<br>
sia = SentimentIntensityAnalyzer()<br>

![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/9fd105e1-828d-41da-803b-de15107d76a8)

#Run the polarity score on the entire dataset<br>
res = {}<br>
for i, row in tqdm(df.iterrows(), total=len(df)):<br>
    text = row['Text']<br>
    myid = row['Id']<br>
    res[myid] = sia.polarity_scores(text)<br>


























































 

































