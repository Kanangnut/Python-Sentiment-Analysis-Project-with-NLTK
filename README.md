# Python Sentiment Analysis Project with Natural Language Toolkit (NLTK) for Classify Amazon Reviews

Open source models for NLP: https://huggingface.co/models. Import data from CSV file.<br>

Start!!<br>
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

Quick EDA <br>
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

Run for check neg/neu/pos DATAFRAME:<br>
![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/7446a9ad-1063-4e85-b037-59ca41a71c0f)

vaders = pd.DataFrame(res).T<br>
vaders = vaders.reset_index().rename(columns={'index': 'Id'})<br>
vaders = vaders.merge(df, how='left')<br>

Run for check table of neg/neu/pos <br>
![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/cff342e3-3fbe-47a4-ab0b-e8e75152cf13)

Plot VADER results:<br>
*I try to change axis y=Compound but found some error so i use axis y=compound instead. (can someone advise to me pls?)<br>
![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/f4aea661-b6dc-46bb-b4b9-711c74bb8e46)
![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/b2c88862-47e3-4775-a1b3-49bdf3a8d5fe)

Step 3.Roberta Pretrained Model

![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/30ea7628-2c0f-4692-994f-116876dc79e1)

![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/1e86e288-6620-4c7f-b9a9-82d7d74bea00)

![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/e4fc52a6-f59c-4060-a37e-c8b3f7b6641e)

Step 3. Run for Roberta Model

Check input_ids first:
![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/1e0ad98e-3891-48c4-a241-cea709ef530b)

![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/c1c47c74-6ace-4735-afa5-05ce53c09bc0)

![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/a7d8a949-9a44-47f5-942d-e4cda89cd8c4)

![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/ac75466c-8913-474d-9a8c-6a20d280c88f)

![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/da11437c-b9f2-4781-a270-46197eb28d0e)

![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/5492f016-bd82-4682-a293-660e50799d59)

![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/237f3a2c-84a6-429a-94ad-12802c23512d)

Compare Scores between model
![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/84672747-a850-4d11-a51b-f885b81fa9e0)

![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/7cb9b911-028c-4aa4-9171-efcfa3e092c3)

Step 4. Review Examples: Identify Positive 1-Star and Negative 5-Star Reviews and look at some examples where the model scoring and review score differ the most.
Start with check table of result

![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/c24b72fe-90be-4284-8b5b-899bd701ea46)

![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/86f71af3-69cb-4144-97a2-722b829ce929)

![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/ed19dbac-0fad-4b23-add5-4fcce56940a0)

Extra: The Transformers Pipeline
![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/460ba10b-0b9d-4502-9366-4766ee36390f)

![image](https://github.com/Kanangnut/Python-Sentiment-Analysis-Project-with-NLTK/assets/130201193/242c1213-fe47-4cb7-9a15-7490f96cf3d7)

Requirment: <br>
punkt <br>
averaged_perceptron_tagger<br>
maxent_ne_chunker <br>
words <br>
vader_lexicon <br>
Xformers <br>
<br>
Note: This project working on Lenovo 2in1 ideapad miix 520, window 10, Azure machine learning workspace.

