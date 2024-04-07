#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd 
import numpy as np


# In[4]:


df = pd.read_csv('spam.csv')


# In[5]:


df.sample(5)


# In[6]:


df.shape


# In[7]:


# Data Cleaning
# EDA
# Text Preprocessing
# Model Building
# Evaluation
# Improvements
# Website
# deploy


# ## 1. Data Cleaning

# In[8]:


df.info()


# In[9]:


# drop last 3 columns
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[10]:


df.sample(5)


# In[11]:


#renaming the columns
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
df.sample(5)


# In[12]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[13]:


df['target'] = encoder.fit_transform(df['target'])


# In[14]:


df.head()


# In[15]:


#missing values
df.isnull().sum()


# In[16]:


# check for duplicate values
df.duplicated().sum()


# In[17]:


#remove duplicates
df = df.drop_duplicates(keep='first')


# In[18]:


df.duplicated().sum()


# In[19]:


df.shape


# ## 2. EDA

# In[20]:


df.head()


# In[21]:


df['target'].value_counts()


# In[22]:


import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(),labels = ['ham','spam'],autopct="%0.2f")
plt.show()


# In[23]:


# Data is inbalanced


# In[24]:


import nltk


# In[25]:


get_ipython().system('pip install nltk')


# In[26]:


nltk.download('punkt')


# In[27]:


df['num_characters'] = df['text'].apply(len)


# In[28]:


df.head()


# In[29]:


# num of words
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[30]:


df.head()


# In[31]:


# num of words
df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[32]:


df.head()


# In[33]:


df[['num_characters','num_words','num_sentences']].describe()


# In[34]:


#ham messages
df[df['target'] == 0][['num_characters','num_words','num_sentences']].describe()


# In[35]:


# spam messages
df[df['target'] == 1][['num_characters','num_words','num_sentences']].describe()


# In[36]:


import seaborn as sns


# In[37]:


sns.histplot(df[df['target'] == 0]['num_characters'])
sns.histplot(df[df['target'] == 1]['num_characters'],color='red')


# In[38]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'],color='red')


# In[39]:


sns.pairplot(df,hue='target')


# In[40]:


sns.heatmap(df.corr(),annot=True)


# ## 3. Data Preprocesssing

# In[41]:


# lower case and tokenization
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()

    
    for i in text:
        y.append(ps.stem(i))
        
        
    return " ".join(y)


# In[42]:


from nltk.corpus import stopwords
stopwords.words('english')


# In[43]:


import string 
string.punctuation


# In[68]:


transform_text("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.")


# In[67]:


df['text'][10]


# In[65]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('playing')


# In[66]:


df['transformed_text'] = df['text'].apply(transform_text)


# In[ ]:


df.head()


# In[69]:


get_ipython().system('pip install wordcloud')


# In[70]:


from wordcloud import WordCloud
WC = WordCloud(width=500,height=500,min_font_size=10,background_color='white')


# In[71]:


spam_WC = WC.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))


# In[72]:


plt.figure(figsize=(15,6))
plt.imshow(spam_WC)


# In[73]:


ham_WC = WC.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))


# In[74]:


plt.figure(figsize=(15,6))
plt.imshow(spam_WC)


# In[75]:


df.head()


# In[76]:


spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[77]:


len(spam_corpus)


# In[78]:


from collections import Counter
sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0],pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation ='vertical')
plt.show()


# In[79]:


ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[80]:


len(ham_corpus)


# In[81]:


from collections import Counter
sns.barplot(pd.DataFrame(Counter(ham_corpus).most_common(30))[0],pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation ='vertical')
plt.show()


# ## 4. Model Building

# In[306]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)


# In[307]:


X = tfidf.fit_transform(df['transformed_text']).toarray()


# In[308]:


X.shape


# In[309]:


y = df['target'].values


# In[310]:


y


# In[311]:


from sklearn.model_selection import train_test_split


# In[312]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[313]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[314]:


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[315]:


gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[316]:


mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[317]:


bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(precision_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))


# In[269]:


# tfidf ----> MNB


# In[270]:


get_ipython().system('pip install xgboost')


# In[271]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[272]:


svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50, random_state=2)
xgb = XGBClassifier(n_estimators=50, random_state=2)


# In[273]:


clfs = {
    'SVC' : svc,
    'KN' : knc,
    'NB' : mnb,
    'DT' : dtc,
    'LR' : lrc,
    'RF' : rfc,
    'AdaBoost' : abc,
    'BgC' : bc,
    'ETC' : etc,
    'GBDT' : gbdt,
    'xgb' : xgb
}


# In[274]:


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision


# In[275]:


train_classifier(svc,X_train,y_train,X_test,y_test)


# In[276]:


accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ", name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[277]:


performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)


# In[278]:


performance_df


# In[279]:


performance_df1 = pd.melt(performance_df, id_vars = "Algorithm")


# In[280]:


performance_df1


# In[281]:


sns.catplot(x = 'Algorithm', y='value',
               hue= 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()


# In[282]:


# Model Improvement
# 1. Change the max_features parameter of TfIdf 


# In[283]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000',ascending=False)


# In[284]:


performance_df.merge(temp_df,on='Algorithm')


# In[285]:


# voting classifier
svc = SVC(kernel='sigmoid',gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier


# In[286]:


voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting = 'soft')


# In[287]:


voting.fit(X_train,y_train)


# In[288]:


y_pred = voting.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[289]:


#Applying stacking
estimators=[('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator = RandomForestClassifier()


# In[290]:


from sklearn.ensemble import StackingClassifier


# In[291]:


clf = StackingClassifier(estimators=estimators, final_estimator= final_estimator)


# In[292]:


clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[318]:


import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))


# In[ ]:




