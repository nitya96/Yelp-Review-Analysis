#!/usr/bin/env python
# coding: utf-8

# #### Read json file into a dataframe

# In[32]:


import pandas as pd
import json
data_file = open("E:/My projects/Python/Yelp Review Analysis/yelp_academic_dataset_business.json", encoding = 'utf8')
data = []
for line in data_file.readlines():
    data.append(json.loads(line))
data_df = pd.DataFrame(data)
data_file.close()


# #### Convert Dataframe to CSV

# In[33]:


data_df.to_csv("E:/My projects/Python/Yelp Review Analysis/business.csv", index=False)


# #### Read converted CSV file

# In[34]:


business_df = pd.read_csv("E:/My projects/Python/Yelp Review Analysis/business.csv")


# #### Filter all restaurants of USA by creating a list of USA states

# In[35]:


states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

usa_states = business_df.loc[business_df['state'].isin(states)]


# In[36]:


usa_states.shape


# #### Select all restaurants in USA states if the category column contains restaurant

# In[37]:


usa_restaurants = usa_states[usa_states["categories"].str.contains("Restaurants") == True]


# #### Select out 16 cuisine types of restaurants, create a list of cuisine type and rename the "categories" column to "category"

# In[38]:


usa_restaurants.is_copy = False #### returns the original data frame

usa_restaurants["category"] = pd.Series() 

cuisine_type=["American","Mexican","Italian","Japenese","Chinese","Thai","Mediterranean","French","Vietnamese","Greek","Indian"
             ,"Korean","Hawaiian","African","Spanish","Middle_eastern"]

for cuisine in cuisine_type:
    usa_restaurants.loc[usa_restaurants.categories.str.contains(cuisine), "category"] = cuisine


# #### Drop null values from Category coloumn, drop categories column and reset index

# In[39]:


usa_restaurants=usa_restaurants.dropna( axis= 0 , subset= ["category"])

del usa_restaurants["categories"]

usa_restaurants = usa_restaurants.reset_index(drop = True)


# #### Check whether the new dataframe usa_restaurants has duplicate business id

# In[40]:


usa_restaurants.business_id.duplicated().sum()


# #### Check the null values

# In[41]:


usa_restaurants.isnull().sum()


# #### Drop the column "hours"

# In[42]:


del usa_restaurants["hours"]


# ## Load Review Table

# In[43]:


with open("E:/My projects/Python/Yelp Review Analysis/yelp_academic_dataset_review.json", "rb") as f:
    all_data = []
    count = 0
    for line in f:
        all_data.append(json.loads(line))
        count += 1
        if (count > 25000):
            break
        
        


# In[44]:


pd.DataFrame(all_data).shape


# #### Convert the list into Dataframe

# In[45]:


all_data = pd.DataFrame(all_data)


# #### Convert the dataframe to csv

# In[46]:


all_data.to_csv("E:/My projects/Python/Yelp Review Analysis/reviews1.csv", index=False)


# #### Read converted CSV file

# In[47]:


reviews_df = pd.read_csv("E:/My projects/Python/Yelp Review Analysis/reviews1.csv")


# #### Check missing values

# In[48]:


reviews_df.isnull().sum()


# #### Check if reviewid is duplicated

# In[49]:


reviews_df.review_id.duplicated().sum()


# ## Merge the two dataframes and get new dataframe usa_restaurants_reviews

# #### Merge both dataframes on common attribute "business_id" 

# In[50]:


usa_restaurants_reviews = pd.merge(usa_restaurants, reviews_df, on = "business_id")


# #### Update column names "stars_x" to "avg_star" and "stars_y" to "review_star"

# In[51]:


usa_restaurants_reviews.rename(columns={"stars_x": "avg_star" , "stars_y":"review_star"} , inplace=True)


# #### Add column "num_words_review" of number of words in review

# In[52]:


usa_restaurants_reviews['num_words_review'] = usa_restaurants_reviews.text.str.replace('\n','').                                           str.replace('[!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~]','').map(lambda x: len(x.split()))


# #### Label reviews as positive or negative

# In[53]:


usa_restaurants_reviews["labels"] = ""
usa_restaurants_reviews.loc[usa_restaurants_reviews.review_star >= 4, "labels"] = "positive"
usa_restaurants_reviews.loc[usa_restaurants_reviews.review_star == 3, "labels"] = "neutral"
usa_restaurants_reviews.loc[usa_restaurants_reviews.review_star < 3, "labels"] = "negative"


# #### Drop rows with label = neutral for easy analysis

# In[54]:


usa_restaurants_reviews.drop(usa_restaurants_reviews[usa_restaurants_reviews["labels"] == "neutral"].index, axis= 0, inplace= True)
usa_restaurants_reviews.reset_index(drop=True, inplace=True)


# ## Exploratory Data Analysis

# ### Restaurants Distribution 

# #### Visualization 1:  Distribution of restaurants in each category

# In[55]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(11,4))

category = usa_restaurants["category"].unique()

count = usa_restaurants["category"].value_counts()

sns.set_style(style="darkgrid")
g=sns.barplot(data= usa_restaurants, x=category ,y= count).set(title="Count of Restaurants by Category")
plt.xticks(rotation=60)
plt.xlabel('category', fontsize=14, labelpad=10)
plt.ylabel('Number of restaurants', fontsize=14)




for  i, v in enumerate(count):
    plt.text(i, v*1.02, str(v), horizontalalignment ='center',fontweight='bold', fontsize=10)

    


# #### Visualization 2 : Top 10 cities with most restaurants

# In[56]:


plt.figure(figsize=(11,5))
city= usa_restaurants["city"].value_counts()[:10]
sns.countplot(y="city", data=usa_restaurants, order=city.index, palette= "inferno").set(title="Count of Restaurants by City (Top 10)")
plt.ylabel('City', fontsize=14, labelpad=10)
plt.xlabel('Number of restaurant', fontsize=14, labelpad=10)
plt.tick_params(labelsize=12)


for  i, v in enumerate(city):
    plt.text(v, i+0.15, str(v), fontweight='bold', fontsize=12)


# #### Visualization 3 : Distribution of restaurants in each state

# In[57]:


plt.figure(figsize=(11,5))
states= usa_restaurants["state"].value_counts()
ustates= usa_restaurants["state"].unique()

sns.barplot(x=ustates, y=states, data=usa_restaurants).set(title = "Distribution of restaurants in each state")
plt.xticks(rotation=0)
plt.xlabel('States', fontsize=14, labelpad=10)
plt.ylabel('Number of restaurants', fontsize=14)

for  i, v in enumerate(states):
    plt.text(i, v*1.02, str(v), horizontalalignment ='center',fontweight='bold', fontsize=10)


# ## Reviews Distribution

# #### Visualization 4 : Distribution of reviews by cuisine type

# In[58]:


plt.figure(figsize=(15,5))
grouped = usa_restaurants.groupby("category")["review_count"].sum().sort_values(ascending = False)


palette_color = sns.color_palette('colorblind')

# # plotting data on chart
plt.pie(grouped, labels=grouped.index,colors=palette_color,autopct='%.0f%%')

plt.title("Distribution of reviews by cuisine type", fontsize=17)
plt.ylabel('category', fontsize=14)
plt.xlabel('Distribution of reviews', fontsize=14)

  
# # displaying chart
plt.show()


# #### Visualization 5 : Top 10 cities with most reviews

# In[59]:


plt.figure(figsize=(11,4))

groupedcity = usa_restaurants.groupby("city")["review_count"].sum().sort_values(ascending = False)[:10]
sns.barplot(x=groupedcity.index, y=groupedcity.values, data= usa_restaurants, palette= "pastel")
plt.xlabel('Number of restaurant', fontsize=14, labelpad=10)
plt.tick_params(labelsize=14)
plt.xticks(rotation=60)

for  i, v in enumerate(groupedcity):
    plt.text(i, v*1.02, str(v), horizontalalignment ='center',fontweight='bold', fontsize=10)


# #### Visualization 6 :Top 9 restaurants with most reviews

# In[60]:


plt.figure(figsize=(11,4))
restaurants = usa_restaurants[["name","review_count"]].sort_values(by= "review_count",ascending = False)[:9]
restaurants

sns.barplot(x=restaurants.review_count, y = restaurants.name, palette= "Paired")
plt.xlabel('Count of Review', labelpad=10, fontsize=14)
plt.ylabel('Restaurants', fontsize=14)
plt.title('TOP 9 Restaurants with Most Reviews', fontsize=15)
plt.tick_params(labelsize=14)
plt.xticks(rotation=15)
for  i, v in enumerate(grouped.review_count):
    plt.text(v, i, str(v), fontweight='bold', fontsize=12)


# #### Distribution of positive and negative reviews in each category

# In[61]:


table = pd.pivot_table(usa_restaurants_reviews, values=["review_id"], index=["category"],columns=["labels"], 
                       aggfunc=len, margins=True, dropna=True,fill_value=0)
table


# In[62]:


table_percentage = table.div( table.iloc[:,-1], axis=0).iloc[:-1,-2].sort_values(ascending=False)
table_percentage


# In[63]:


plt.figure(figsize=(11,8))
plt.subplot(211)
sns.pointplot(x=table_percentage.index, y= table_percentage.values)
plt.xlabel('Category', labelpad=7, fontsize=14)
plt.ylabel('Percentage of positive reviews', fontsize=14)
plt.title('Percentage of Positive Reviews', fontsize=15)
plt.tick_params(labelsize=14)
plt.xticks(rotation=40)
for  i, v in enumerate(table_percentage.round(2)):
    plt.text(i, v*1.001, str(v), horizontalalignment ='center',fontweight='bold', fontsize=14)
    


# In[64]:


plt.figure(figsize=(11,8))
plt.subplot(212)
grouped = usa_restaurants_reviews.groupby('category')['review_star'].mean().round(2).sort_values(ascending=False)
sns.pointplot(grouped.index, grouped.values)
plt.ylim(3)
plt.xlabel('Catagory', labelpad=10, fontsize=14)
plt.ylabel('Average Rating', fontsize=14)
plt.title('Average Rating of each Category', fontsize=15)
plt.tick_params(labelsize=14)
plt.xticks(rotation=40)
for  i, v in enumerate(grouped):
    plt.text(i, v, str(v), horizontalalignment ='center',fontweight='bold', fontsize=14)
    
plt.subplots_adjust(hspace=1)


# ### Average length of reviews

# #### Average length of words in each category

# In[65]:


table = usa_restaurants_reviews.groupby(['category','labels'])['num_words_review'].mean().round().unstack()
table
plt.figure(figsize=(11,8))
sns.heatmap(table, cmap='YlGnBu', fmt='g',annot=True, linewidths=1)
plt.tick_params(labelsize=15)


# ### Ratings Distribution

# #### Distribution of ratings by restaurants

# In[66]:


plt.figure(figsize=(11,6))
grouped= usa_restaurants.stars.value_counts().sort_index()
sns.barplot(x=grouped.index, y= grouped.values)
plt.xlabel('Average Rating', labelpad=10, fontsize=14)
plt.ylabel('Count of restaurants', fontsize=14)
plt.title('Count of Restaurants against Ratings', fontsize=15)
for i,v in enumerate(grouped):
    plt.text(i, v*1.02,str(v),horizontalalignment ='center',fontweight='bold', fontsize=14)


# #### Distribution of ratings by reviews

# In[67]:


plt.figure(figsize=(11,6))
grouped= usa_restaurants_reviews.review_star.value_counts().sort_index()
sns.barplot(x=grouped.index, y= grouped.values)
plt.xlabel("Review Rating",  labelpad=10, fontsize=14)
plt.ylabel('Count of reviews', fontsize=14)
plt.title('Count of Reviews against Rating', fontsize=15)
plt.tick_params(labelsize=14)
for  i, v in enumerate(grouped):
    plt.text(i, v*1.02, str(v), horizontalalignment ='center',fontweight='bold', fontsize=14)


# ## Review Analysis

# ### Positive and negative words

# In[68]:


import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
pd.set_option('display.float_format', lambda x: '%.4f' % x)


# In[69]:


import nltk
import string, itertools
from collections import Counter, defaultdict
from nltk.text import Text
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from sklearn.cluster import KMeans
from wordcloud import WordCloud


# #### Convert "text" to lower case

# In[70]:


usa_restaurants_reviews.text = usa_restaurants_reviews.text.str.lower()


# In[71]:


usa_restaurants_reviews.category


# #### Remove unnecessary punctuation

# In[72]:


usa_restaurants_reviews['removed_punct_text']= usa_restaurants_reviews.text.str.replace('\n','').                                           str.replace('[!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~]','')


# #### Import positive file which contains common meaningless positive words such as "good"

# In[73]:


positive_file = open('E:/My projects/Python/Yelp Review Analysis/positive.txt')
read_file =csv.reader(positive_file)
words_positive = [word[0] for word in read_file]


# #### Import negative file which contains common meaningless positive words such as "bad"

# In[74]:


negative_file = open('E:/My projects/Python/Yelp Review Analysis/negative.txt')
read_file =csv.reader(negative_file)
words_negative = [word[0] for word in read_file]


# #### Get dataset by category

# In[75]:


def get_dataset(category):
    new_df = usa_restaurants_reviews[['removed_punct_text','labels']][usa_restaurants_reviews.category==category]
    new_df.reset_index(drop=True, inplace =True)
    new_df.rename(columns={'removed_punct_text':'text'}, inplace=True)
    return new_df


# #### Only keep positive and negative words

# In[76]:


def filter_words(review):
    words = [word for word in review.split() if word in words_positive + words_negative]
    words = ' '.join(words)
    return words


# #### For category "Korean"

# In[77]:


Korean_reviews = get_dataset("Korean")


# #### Split the "Korean_reviews" into train and test data

# In[78]:


Korean_train, Korean_test = train_test_split(Korean_reviews[["text","labels"]],test_size= 0.5)


# In[79]:


def split_data(dataset, test_size):
    df_train, df_test = train_test_split(dataset[['text','labels']],test_size=test_size)
    return df_train


# In[80]:


Korean_train.text = Korean_train.text.apply(filter_words)


# In[81]:


train_terms = list(Korean_train["text"])
train_class = list(Korean_train["labels"])


# In[82]:


test_terms = list(Korean_test["text"])
test_class = list(Korean_test["labels"])


# #### Use CountVectorizer to get bag of words : the frequencies of various words appeared in each review

# In[83]:


countvectorizer = CountVectorizer()
train_feature_counts = countvectorizer.fit_transform(train_terms)
train_feature_counts.shape


# #### Run the SVM Model

# In[84]:


svm = LinearSVC()
svm.fit(train_feature_counts, train_class)


# #### Calculate polarity score of each word in Korean category

# In[85]:


coeff = svm.coef_[0]


# #### Create dataframe for score of each word in a review calculated by SVM model

# In[86]:


Korean_words_score = pd.DataFrame({"score": coeff , "words" : countvectorizer.get_feature_names()})


# #### Get frequency of each words in all reviews in Korean category

# In[136]:


Korean_reviews = pd.DataFrame(train_feature_counts.toarray(), columns = countvectorizer.get_feature_names())
Korean_reviews['labels'] = train_class
Korean_frequency = Korean_reviews[Korean_reviews['labels'] =='positive'].sum()[:-1]


# In[88]:


Korean_words_score.set_index('words', inplace=True)


# In[135]:


Korean_polarity_score = Korean_words_score
Korean_polarity_score['frequency'] = Korean_frequency


# #### Calculate polarity score 

# In[90]:


Korean_polarity_score['polarity'] = Korean_polarity_score.score * Korean_polarity_score.frequency / Korean_reviews.shape[0]


# In[91]:


## drop unnecessary words
list1= ['great','amazing','love','best','awesome','excellent','good','favorite','loved','perfect','gem','perfectly','wonderful','happy','enjoyed','nice','well','super','like',
        'better','decent','fpretty','enough','excited','impressed','ready','fantastic','glad','right','fabulous']

unuseful_positive_words = Korean_polarity_score.reindex(columns = list1)

list2 = ['bad','disappointed','unfortunately','disappointing','horrible','lacking','terrible','sorry', 'disappoint']

unuseful_negative_words =  Korean_polarity_score.reindex(columns = list2)

Korean_polarity_score.drop(unuseful_positive_words.index, axis=0, inplace = False)
Korean_polarity_score.drop(unuseful_negative_words.index, axis=0, inplace = False)


# In[92]:


Korean_polarity_score.polarity = Korean_polarity_score.polarity.astype(float)
Korean_polarity_score.frequency = Korean_polarity_score.frequency.astype(float)


# In[137]:


Korean_polarity_score[Korean_polarity_score.polarity>0].sort_values('polarity', ascending=False)[:10]


# #### Get top most informative positive and negative words for "Korean"

# In[94]:


Korean_top_positive_words = ['delicious','friendly','attentive','recommend','fresh','variety','reasonable','tender','clean','authentic']
Korean_top_negative_words = ['bland','slow','expensive','overpriced', 'cold', 'greasy','sweet','fatty','rude','dirty']
Korean_top_words = Korean_polarity_score.reindex(Korean_top_positive_words + Korean_top_negative_words).polarity.dropna()


# In[98]:


plt.figure(figsize=(11,4))
colors = ['red' if c < 0 else 'blue' for c in Korean_top_words.values]
sns.barplot(y=Korean_top_words.index, x=Korean_top_words.values, palette=colors)
plt.xlabel('Polarity Score', labelpad=10, fontsize=14)
plt.ylabel('Words', fontsize=14)
plt.title('TOP Positive and Negative Words in Korean Restaurants', fontsize=15)
plt.tick_params(labelsize=14)
plt.xticks(rotation=15)


# #### Create function to get polarity score of category column

# In[138]:


def get_polarity_score(dataset):
    dataset.text = dataset.text.apply(filter_words)
    
    terms_train=list(dataset['text'])
    class_train=list(dataset['labels'])
    
    ## get bag of words
    vectorizer = CountVectorizer()
    feature_train_counts=vectorizer.fit_transform(terms_train)
    
    ## run model
    svm = LinearSVC()
    svm.fit(feature_train_counts, class_train)
    
    ## create dataframe for score of each word in a review calculated by svm model
    coeff = svm.coef_[0]
    cuisine_words_score = pd.DataFrame({'score': coeff, 'word': vectorizer.get_feature_names()})
    
    ## get frequency of each word in all reviews in specific category
    cuisine_reviews = pd.DataFrame(feature_train_counts.toarray(), columns=vectorizer.get_feature_names())
    cuisine_reviews['labels'] = class_train
    cuisine_frequency = cuisine_reviews[cuisine_reviews['labels'] =='positive'].sum()[:-1]
    
    cuisine_words_score.set_index('word', inplace=True)
    cuisine_polarity_score = cuisine_words_score
    cuisine_polarity_score['frequency'] = cuisine_frequency
    
    cuisine_polarity_score.score = cuisine_polarity_score.score.astype(float)
    cuisine_polarity_score.frequency = cuisine_polarity_score.frequency.astype(int)
    
    ## calculate polarity score 
    cuisine_polarity_score['polarity'] = cuisine_polarity_score.score * cuisine_polarity_score.frequency / cuisine_reviews.shape[0]
    
    cuisine_polarity_score.polarity = cuisine_polarity_score.polarity.astype(float)
    ## drop unnecessary words
    unuseful_positive_words = ['great','amazing','love','best','awesome','excellent','good',
                                                   'favorite','loved','perfect','gem','perfectly','wonderful',
                                                    'happy','enjoyed','nice','well','super','like','better','decent','fine',
                                                    'pretty','enough','excited','impressed','ready','fantastic','glad','right',
                                                    'fabulous']
    unuseful_positive_words = cuisine_polarity_score.reindex(columns =  unuseful_positive_words)
    
    unuseful_negative_words =  ['bad','disappointed','unfortunately','disappointing','horrible',
                                                    'lacking','terrible','sorry']
    unuseful_negative_words = cuisine_polarity_score.reindex(columns =  unuseful_negative_words)
    
    unuseful_words = unuseful_positive_words + unuseful_negative_words
    cuisine_polarity_score.drop(unuseful_words.index, axis=0, inplace=False)
    
    return cuisine_polarity_score


# #### Create function to plot top positive and negative words

# In[142]:


def plot_top_words(top_words, category):
    plt.figure(figsize=(11,4))
    colors = ['red' if c < 0 else 'blue' for c in top_words.values]
    sns.barplot(y=top_words.index, x=top_words.values, palette=colors)
    plt.xlabel('Polarity Score', labelpad=10, fontsize=14)
    plt.ylabel('Words', fontsize=14)
    plt.title('TOP 10 Positive and Negative Words in %s Restaurants ' % category, fontsize=15)
    plt.tick_params(labelsize=14)
    plt.xticks(rotation=15)


# #### Create function to get top words in reviews

# In[143]:


def get_top_words(dataset, label, number=20):
    if label == 'positive':
        df = dataset[dataset.polarity>0].sort_values('polarity',ascending = False)[:number]
    else:
        df = dataset[dataset.polarity<0].sort_values('polarity')[:number]
    return df


# #### For "Greek" cuisine 

# In[144]:


Greek_reviews = get_dataset('Greek')
Greek_train = split_data(Greek_reviews, 0.5)
print('Total %d number of reviews' % Greek_train.shape[0])


# In[145]:


Greek_polarity_score = get_polarity_score(Greek_train)
get_top_words(Greek_polarity_score, 'positive',10)


# In[146]:


get_top_words(Greek_polarity_score,'negative',10)


# In[147]:


Greek_top_positive_words = ['great','amazing','delicious','love','awesome','excellent',
                               'loved','best','enjoyed','enough']
Greek_top_negative_words = ['impressed','fried','right','well','clean','bad','friendly','disappointed','cold','worst']
Greek_top_words = Greek_polarity_score.loc[Greek_top_positive_words+Greek_top_negative_words,'polarity']
plot_top_words(Greek_top_words,'Greek')


# #### For "Thai" cuisine

# #### For "Chinese" cuisine

# In[117]:


Chinese_reviews = get_dataset('Chinese')
Chinese_train = split_data(Chinese_reviews, 0.85)
print('Total %d number of reviews' % Chinese_train.shape[0])


# In[118]:


Chinese_polarity_score = get_polarity_score(Chinese_train)
get_top_words(Chinese_polarity_score,'positive',10)


# In[133]:


Chinese_polarity_score = get_polarity_score(Chinese_train)
get_top_words(Chinese_polarity_score,'negative',10)


# In[134]:


Chinese_top_positive_words = ['great','favorite','delicious','good','best','hot','nice',
                           'authentic','recommend','better']
Chinese_top_negative_words = ['like','slow','love','enough','amazing','right','fine','sweet','sour','wrong']
Chinese_top_words = Chinese_polarity_score.loc[Chinese_top_positive_words+Chinese_top_negative_words,'polarity']
plot_top_words(Chinese_top_words, 'Chinese')


# #### For "Vietnamese" cuisine

# In[120]:


Vietnamese_reviews = get_dataset('Vietnamese')
Vietnamese_train = split_data(Vietnamese_reviews, 0.5)
print('Total %d number of reviews' % Vietnamese_train.shape[0])


# In[122]:


Vietnamese_polarity_score = get_polarity_score(Vietnamese_train)
get_top_words(Vietnamese_polarity_score,'positive',10)


# In[129]:


get_top_words(Vietnamese_polarity_score,'negative',10)


# In[130]:


Viet_top_positive_words = ['great','delicious','fresh','favorite','best','clean','amazing',
                           'love','cheap','recommend']
Viet_top_negative_words = ['friendly','good','better','enjoy','hard','hot','disappointing','impressed','wrong','bland']
Viet_top_words = Vietnamese_polarity_score.loc[Viet_top_positive_words+Viet_top_negative_words,'polarity']
plot_top_words(Viet_top_words,'Viet')


# In[168]:


Thai_reviews = get_dataset('Thai')
Thai_train = split_data(Thai_reviews, 0.8)
print('Total %d number of reviews' % Thai_train.shape[0])


# In[169]:


Thai_polarity_score = get_polarity_score(Thai_train)


# In[171]:


get_top_words(Thai_polarity_score,'positive',10)


# In[173]:


get_top_words(Thai_polarity_score,'negative',10)


# In[174]:


Thai_top_positive_words = ['great','delicious','good','best','excellent','enjoy','favorite',
                           'clean','love','nice']
Thai_top_negative_words = ['sweet','enough','fried','like','top','bad','work','fresh','worth','bland']
Thai_top_words = Thai_polarity_score.loc[Thai_top_positive_words+Thai_top_negative_words,'polarity']
plot_top_words(Thai_top_words, 'Thai')

