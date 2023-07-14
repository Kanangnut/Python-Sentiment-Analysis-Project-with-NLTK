# Python-Sentiment-Analysis-Project-with-NLTK and classify Amazon Reviews

In case you run NLTK for the first time must be install:<br>
  1.nltk.download('punkt') install<br>
  2.nltk.download('averaged_perceptron_tagger')<br>
  3.nltk.download('maxent_ne_chunker')<br>
  4.nltk.download('words')<br>
  5.nltk.download('vader_lexicon')<br>

Note: This project working on Lenovo 2in1 ideapad miix 520, window 10. <br>

Start!!
For sentiment Analysis in Python of this project by using two different techniques:<br>
1.VADER (Valence Aware Dictionary and Sentiment Reasoner)<br>
2.Roberta Pretrained Model<br>
3.Huggingface Pipeline<br>

Step 0. Read in Data and NLTK (Natural Language Toolkit)<br>
import pandas as pd<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
import seaborn as sns<br>
import nltk<br>

plt.style.use('ggplot')<br>

Link with the data CSV file:<br>
df = pd.read_csv('../20230709-Python-Sentiment-Analysis-Project-with-NLTK/Reviews.csv')<br>
df = df.head(500)<br>
print(df.shape)<br>
This step should run for check the result<br>
Result show:<br>
![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/d5db9774-3ee2-44d3-833d-7b5fcda13345)

Then check Table result by:<br>
df.head()<br>
![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/729a3695-1be5-44cd-bc59-569a6b8f8378)

Quick EDA<br>
df['Score'].value_counts().sort_index().plot(kind='bar', title='Count of Reviews by Starts', figsize=(8, 5))<br>
![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/b2785148-b983-4146-896e-4da5d78ee3f2)

![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/180794db-8757-47b4-8697-c2c7e70c11e6)

![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/f4340184-af8a-4e9b-b697-36e7af9caa5a)


NLTK NLTK for generate postag:<br>
![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/e5423197-79c0-427a-a5a7-c4c90c5b9454)


NLTK for generate entitle:<br>
![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/0b0d5f05-1467-4e4c-8376-f6893666e8a3)


Step 1. VADER Seniment Score<br>
Use NLTK's SentimentIntensityAnalyzer to get the neg/neu/pos scores of the reviews.<br>

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

Run for check neg/neu/pos DATAFRAME:
![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/7446a9ad-1063-4e85-b037-59ca41a71c0f)

vaders = pd.DataFrame(res).T<br>
vaders = vaders.reset_index().rename(columns={'index': 'Id'})<br>
vaders = vaders.merge(df, how='left')<br>

























































 

































