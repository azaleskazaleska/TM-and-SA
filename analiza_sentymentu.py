#!/usr/bin/env python
# coding: utf-8

# # Modelowanie tematyczne i analiza sentymentu pojazdów elektrycznych na podstawie danych z Reddit

# Aleksandra Załęska

# Styczeń, 2024

# # Instalacja bibliotek

# Niezbędne biblioteki:

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import re
import nltk
import spacy

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')


# # Zebranie danych 

# Dane wykorzystane w tym projekcie pochodzą ze strony: https://the-eye.eu/redarcs/. Wykorzystano dwa subreddity: r/Cars oraz r/cartalk i pobrano pliki dla postów oraz komentarzy.

# Wczytanie plików JSON dla postów. Wczytywany plik jest jedynie plikiem przykładowym i nie stanowi całości pliku, który został wykorzystany w dalszej analizie. Służy przedstawieniu działania poniższych funkcji:

# Funkcja wczytująca pliki JSON:

# In[2]:


def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return data


# Podanie ścieżki do pliku JSON:

# In[3]:


input_file_path = r'C:\Users\sciezka_do_pliku'


# Załadowanie pliku JSON:

# In[4]:


cartalk_submissions = load_json_file(input_file_path)


# Wyświetlenie dwóch pierwszych elementów:

# In[5]:


for i, element in enumerate(cartalk_submissions[:2], 1):
    print(f"Element {i}: {element}")


# Odfiltrowanie określonych kluczy, które bedą dla nas przydatne:

# Funkcja do odlflitrowania:

# In[6]:


def filter_json(json_data, keys):
    filtered_data = []
    for item in json_data:
        filtered_item = {key: item[key] for key in keys if key in item}
        filtered_data.append(filtered_item)
    return filtered_data


# Wybrane klucze:

# In[7]:


selected_keys = ["author", "score", "selftext", "subreddit", "created_utc", "name", "url", "num_comments"]


# Filtrowanie danych korzystając z funkcji:

# In[8]:


cartalk_submissions_filtered = filter_json(cartalk_submissions, selected_keys)


# Wyświetlenie dwóch pierwszych elementów:

# In[9]:


for i, element in enumerate(cartalk_submissions_filtered[:2], 1):
    print(f"Element {i}: {element}")


# Przekształcenie danych na DataFrame:

# In[10]:


cartalk_submissions_df = pd.DataFrame(cartalk_submissions_filtered)


# Usunięcie wierszy, które w kolumnie 'name' mają wartość 'NaN', ponieważ w kolumnie tej znajdują się unikalne wartości stanowiące później kolumnę łączącą posty z komentarzami:

# In[11]:


cartalk_submissions_df.dropna(subset=['name'], inplace=True)


# Odfiltrowanie wierszy, które w kolumnie 'selftext', czyli w kolumnie, która przechowuje treść postów występują frazy takie jak: "electric car", "electric cars", "electric vehicle", "electric vehicles", "evs". W tym celu na kolumnie 'selftext' należy przeprowadzić kilka operacji:

# Usunięcie znaków interpunkcyjnych, wykrzyknień, pytajników itp.:

# In[12]:


cartalk_submissions_df['selftext'] = cartalk_submissions_df['selftext'].str.replace(r'[^\w\s]', '', regex=True)


# Usunięcie znaków nowej linii:

# In[13]:


cartalk_submissions_df['selftext'] = cartalk_submissions_df['selftext'].str.replace('\n', ' ')


# Zamiana wszystkich liter na małe:

# In[14]:


cartalk_submissions_df['selftext'] = cartalk_submissions_df['selftext'].str.lower()


# Lista słów kluczowych:

# In[15]:


keywords = ["electric car", "electric cars", "electric vehicle", "electric vehicles", "evs"]


# Warunek logiczny sprawdzający, czy któryś z kluczowych słów występuje w kolumnie 'selftext'/:

# In[16]:


condition = cartalk_submissions_df['selftext'].str.contains('|'.join(keywords), case=False)


# Wybieranie wierszy spełniających warunek:

# In[17]:


cartalk_submissions_df_with_keywords = cartalk_submissions_df[condition]


# Zapisanie DataFrame do pliku Excel:

# In[18]:


cartalk_submissions_df_with_keywords.to_excel('cartalk_posts.xlsx', index=False)


# Wczytanie plików JSON dla komentarzy:

# Korzystamy z wczesniej stworzonej funckji load_json_file:

# Podanie ścieżki do pliku:

# In[19]:


input_file_path_2 = r'C:\Users\sciezka_do_pliku'


# Załadowanie pliku JSON z komentarzami:

# In[20]:


cartalk_comments = load_json_file(input_file_path_2)


# Wyświetlenie dwóch pierwszych elementów:

# In[21]:


for i, element in enumerate(cartalk_comments[:2], 1):
    print(f"Element {i}: {element}")


# Odfiltrowanie tylko kluczy: "author", "score", "body", "subreddit", "created_utc", "link_id". W tym celu wykorzystamy ponownie wczesniejszą funkcję filter_json: 

# In[22]:


selected_keys = ["author", "score", "body", "subreddit", "created_utc", "link_id"]


# In[23]:


cartalk_comments_filtered = filter_json(cartalk_comments, selected_keys)


# Wyświetlenie dwóch pierwszych elementów:

# In[24]:


for i, element in enumerate(cartalk_comments_filtered[:2], 1):
    print(f"Element {i}: {element}")


# Przekształcenie na DataFrame:

# In[25]:


cartalk_comments = pd.DataFrame(cartalk_comments_filtered)


# Odfiltrowanie ramki danych cars_comments, tak aby pozostawić tylko to wiersze, które w kolumnie 'link_id' posiadają wartośći, które występują także w kolumnie 'name' w tabeli cartalk_submissions_df_with_keywords:

# In[26]:


cartalk_comments = cartalk_comments[cartalk_comments['link_id'].isin(cartalk_submissions_df_with_keywords['name'])]


# Zapisanie DataFrame do pliku Excel:

# In[27]:


cartalk_comments.to_excel('cartalk_comments.xlsx', index=False, encoding='utf-8')


# # Przetwarzanie danych

# Załadowanie tabel z plików Excel:

# In[28]:


cars_comm = pd.read_excel('https://docs.google.com/spreadsheets/d/e/2PACX-1vQbudhhu7rixkAetFc7Bj4Hgh-47mCiYv40dUYgccX8kwrlDWdGtc7yb3nTfu_MMQ/pub?output=xlsx')
cartalk_comm = pd.read_excel('https://docs.google.com/spreadsheets/d/e/2PACX-1vSWtTNocelTZ-89zGEzawpUeE0PnVitvEuWPRq2Z074wGRRx_2B0VHSBxD5yHyPoQ/pub?output=xlsx')


# In[29]:


cars_posts = pd.read_excel('https://docs.google.com/spreadsheets/d/e/2PACX-1vRNTTCM4HQOgHrZxCM-RlO2Uz-5VOkkwO4dEo6su4iSraUjRSL24hVrXq9nQIdUag/pub?output=xlsx')
cartalk_posts = pd.read_excel('https://docs.google.com/spreadsheets/d/e/2PACX-1vRT0BPUALq_swCALsinT9zMCQhpmQayYab2_SNDZUkE2_i8qZxUrQ9N24GbnWZLqQ/pub?output=xlsx')


# Złączenie plików z danymi z komentarzy w jeden zbiór:

# In[30]:


ev_comments = pd.concat([cars_comm, cartalk_comm])


# Złączenie plików z danymi z postów w jeden zbiór:

# In[31]:


ev_posts = pd.concat([cars_posts, cartalk_posts])


# Przekształcenie formatu kolumny ev_posts['created_utc']:

# In[32]:


ev_posts['created_utc'] = pd.to_datetime(ev_posts['created_utc'], unit='s')


# In[33]:


ev_posts['created_utc'] = ev_posts['created_utc'].dt.strftime('%Y-%m-%d')


# Filtrowanie postów od od 01-01-2012 do 31-12-2022:

# In[34]:


ev_posts = ev_posts[(ev_posts['created_utc'] >= '2012-01-01') & (ev_posts['created_utc'] <= '2022-12-31')]


# Pobranie listy stop words z NLTK:

# In[35]:


stop_words = set(stopwords.words('english'))


# Dodanie niestandardowych stop words

# In[36]:


custom_stop_words = set(['lol', 'itâs', 'iâm', 'donât', 'youre', 'iam', 'yes', 'guy', 'like', 'dont', 'doesnt',
                         'model', 'thats', 'yeah', 'le', 'adhd', 'ive', 'ill', 'even', 'one', 'though', 'think', 
                         'thing', 'weve', 'kinda'])
stop_words.update(custom_stop_words)


# Pobranie lematyzatora z NLTK:

# In[37]:


lemmatizer = WordNetLemmatizer()


# Przed analizą sentymentu należy usunąć frazy: "electric car", "electric cars", "electric vehicle", "electric vehicles", "evs"

# In[39]:


phrases_to_remove = ["electric car", "electric cars", "electric vehicle", "electric vehicles", 
                     "evs", "[deleted]", "car", "cars"]


# Funkcja do przetworzenia danych:

# In[40]:


def preprocess_text(text, phrases_to_remove, remove_numbers=True):
    if pd.isna(text):
        return ''

    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()

    for phrase in phrases_to_remove:
        text = text.replace(phrase, '')

    if remove_numbers:
        text = re.sub(r'\d+', '', text)

    tokens = word_tokenize(text)

    # usuwanie słów krótszych niż 3 znaki
    tokens = [word for word in tokens if len(word) > 2]

    # usuwanie słów spoza angielskiego
    tokens = [word for word in tokens if word.isalpha()]

    # usuwanie stop words
    tokens = [word for word in tokens if word not in stop_words]

    # POS tagging
    tagged_tokens = pos_tag(tokens)

    # lematyzacja
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


# Przetwarzanie kolumny ev_posts['selftext']

# In[41]:


ev_posts['selftext_processed'] = ev_posts['selftext'].apply(lambda x: preprocess_text(x, phrases_to_remove, remove_numbers=True))


# Przetwarzanie kolumny ev_comments['body']

# In[42]:


ev_comments['body_processed'] = ev_comments['body'].apply(lambda x: preprocess_text(x, phrases_to_remove, remove_numbers=True))


# # Analiza sentymentu

# In[43]:


sia = SentimentIntensityAnalyzer()


# Sentyment dla postów:

# In[44]:


ev_posts['posts_sentiment_score'] = ev_posts['selftext_processed'].apply(lambda x: sia.polarity_scores(x)['compound'])


# Sentyment dla komentarzy:

# In[45]:


ev_comments['comm_sentiment_score'] = ev_comments['body_processed'].apply(lambda x: sia.polarity_scores(x)['compound'])


# ## Kategoryzacja sentymentu

# Dla postów:

# In[46]:


ev_posts['posts_sentiment_label'] = ev_posts['posts_sentiment_score'].apply(lambda x: 'positive' if x > 0.33 else ('negative' if x < -0.33 else 'neutral'))


# In[47]:


ev_posts


# Dla komentarzy:

# In[48]:


ev_comments['comm_sentiment_label'] = ev_comments['comm_sentiment_score'].apply(lambda x: 'positive' if x > 0.33 else ('negative' if x < -0.33 else 'neutral'))


# In[49]:


ev_comments


# ## Porównanie rozkładu sentymentu w postach i komentarzach

# Zliczenie sentymentów w różnych kategoriach dla postów:

# In[50]:


sentiment_distribution_posts = ev_posts['posts_sentiment_label'].value_counts()
sentiment_distribution_posts


# Zliczenie sentymentów w różnych kategoriach dla komentarzy:

# In[51]:


sentiment_distribution_comm = ev_comments['comm_sentiment_label'].value_counts()
sentiment_distribution_comm


# ## Wykres słupkowy porównujący rozkład sentymentu w komentarzach i postach

# Grupowanie i zliczanie ev_posts:

# In[52]:


posts_sentiment_distribution = ev_posts['posts_sentiment_label'].value_counts(normalize=True) * 100


# Grupowanie i zliczanie ev_comments:

# In[53]:


comments_sentiment_distribution = ev_comments['comm_sentiment_label'].value_counts(normalize=True) * 100


# Wykres:

# In[54]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

sns.barplot(x=posts_sentiment_distribution.index, y=posts_sentiment_distribution.values, ax=axes[0], palette="viridis")
axes[0].set_title('Rozkład Sentymentu dla postów')
axes[0].set_xlabel('Sentyment')
axes[0].set_ylabel('Procent')

for i, v in enumerate(posts_sentiment_distribution.values):
    axes[0].text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)

sns.barplot(x=comments_sentiment_distribution.index, y=comments_sentiment_distribution.values, ax=axes[1], palette="viridis")
axes[1].set_title('Rozkład sentymentu dla komentarzy')
axes[1].set_xlabel('Sentyment')
axes[1].set_ylabel('Procent')

for i, v in enumerate(comments_sentiment_distribution.values):
    axes[1].text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()


# ## Porównanie rozkładu sentymetu postów w zależności od subreddit

# Grupowanie i zliczanie dla Cartalk:

# In[55]:


cartalk_sentiment_distribution = ev_posts[ev_posts['subreddit'] == 'Cartalk']['posts_sentiment_label'].value_counts(normalize=True) * 100


# In[56]:


ev_posts


# Grupowanie i zliczanie dla cars:

# In[57]:


cars_sentiment_distribution = ev_posts[ev_posts['subreddit'] == 'cars']['posts_sentiment_label'].value_counts(normalize=True) * 100


# Wykres:

# In[58]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

sns.barplot(x=cartalk_sentiment_distribution.index, y=cartalk_sentiment_distribution.values, ax=axes[0], palette="viridis")
axes[0].set_title('Rozkład sentymentu postów w Cartalk')
axes[0].set_xlabel('Sentyment')
axes[0].set_ylabel('Procent')

for i, v in enumerate(cartalk_sentiment_distribution.values):
    axes[0].text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)

sns.barplot(x=cars_sentiment_distribution.index, y=cars_sentiment_distribution.values, ax=axes[1], palette="viridis")
axes[1].set_title('Rozkład sentymentu postów w cars')
axes[1].set_xlabel('Sentyment')
axes[1].set_ylabel('Procent')

for i, v in enumerate(cars_sentiment_distribution.values):
    axes[1].text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()


# ## Porównanie rozkładu sentymetu komentarzy w zależności od subreddit

# Grupowanie i zliczanie dla Cartalk:

# In[59]:


cartalk_comm_sentiment_distribution = ev_comments[ev_comments['subreddit'] == 'Cartalk']['comm_sentiment_label'].value_counts(normalize=True) * 100


# Grupowanie i zliczanie dla cars:

# In[60]:


cars_comm_sentiment_distribution = ev_comments[ev_comments['subreddit'] == 'cars']['comm_sentiment_label'].value_counts(normalize=True) * 100


# Wykres:

# In[61]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

sns.barplot(x=cartalk_comm_sentiment_distribution.index, y=cartalk_comm_sentiment_distribution.values, ax=axes[0], palette="viridis")
axes[0].set_title('Rozkład sentymentu komentarzy w Cartalk')
axes[0].set_xlabel('Sentyment')
axes[0].set_ylabel('Procent')

for i, v in enumerate(cartalk_comm_sentiment_distribution.values):
    axes[0].text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)

sns.barplot(x=cars_comm_sentiment_distribution.index, y=cars_comm_sentiment_distribution.values, ax=axes[1], palette="viridis")
axes[1].set_title('Rozkład sentymentu komentarzy w cars')
axes[1].set_xlabel('Sentyment')
axes[1].set_ylabel('Procent')

for i, v in enumerate(cars_comm_sentiment_distribution.values):
    axes[1].text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()


# ## Ogólny rozkład sentymentu w całym zbiorze

# Połączenie zbiorów ev_comments i ev_posts:

# In[62]:


combined_data = pd.concat([ev_comments['comm_sentiment_label'], ev_posts['posts_sentiment_label']])


# Grupowanie i zliczanie dla ogólnego zbioru:

# In[63]:


total_sentiment_distribution = combined_data.value_counts(normalize=True) * 100


# Wykres:

# In[64]:


plt.figure(figsize=(8, 5))
sns.barplot(x=total_sentiment_distribution.index, y=total_sentiment_distribution.values, palette="viridis")
plt.title('Ogólny rozkład sentymentu w całym zbiorze')
plt.xlabel('Sentyment')
plt.ylabel('Procent')

for i, v in enumerate(total_sentiment_distribution.values):
    plt.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)

plt.show()


# ## Rozkład sentymentu w czasie dla komentarzy i postów

# Konwertowanie kolumny 'created_utc' na obiekt datetime:

# In[65]:


ev_comments['created_utc'] = pd.to_datetime(ev_comments['created_utc'])
ev_posts['created_utc'] = pd.to_datetime(ev_posts['created_utc'])


# Utworzenie kolumny 'year' do grupowania co rok:

# In[66]:


ev_comments['year'] = ev_comments['created_utc'].dt.to_period("Y")
ev_posts['year'] = ev_posts['created_utc'].dt.to_period("Y")


# Średni sentyment dla każdego interwału w ev_comments i ev_posts:

# In[67]:


avg_sentiment_comments = ev_comments.groupby('year')['comm_sentiment_score'].mean()
avg_sentiment_posts = ev_posts.groupby('year')['posts_sentiment_score'].mean()


# Wykresy:

# In[68]:


plt.figure(figsize=(12, 6))

# ev_comments
plt.subplot(1, 2, 1)
sns.lineplot(x=avg_sentiment_comments.index.astype(str), y=avg_sentiment_comments.values, marker='o', color='b')
plt.title('Średni sentyment w czasie dla komentarzy')
plt.xlabel('Rok')
plt.ylabel('Średni sentyment')
plt.xticks(rotation=45)
plt.ylim(0, 0.5) 

# ev_posts
plt.subplot(1, 2, 2)
sns.lineplot(x=avg_sentiment_posts.index.astype(str), y=avg_sentiment_posts.values, marker='o', color='g')
plt.title('Średni sentyment w czasie dla postów')
plt.xlabel('Rok')
plt.ylabel('Średni sentyment')
plt.xticks(rotation=45)
plt.ylim(0, 0.5)  

plt.tight_layout()
plt.show()


# ## Rozkład sentymentu w czasie w całym zbiorze

# In[69]:


comments_subset = ev_comments[['created_utc', 'comm_sentiment_score']]
posts_subset = ev_posts[['created_utc', 'posts_sentiment_score']]


# In[70]:


comments_subset.columns = ['created_utc', 'sentiment_score']
posts_subset.columns = ['created_utc', 'sentiment_score']


# In[71]:


ev_all = pd.concat([comments_subset, posts_subset], ignore_index=True)


# Konwertowanie kolumny 'created_utc' na obiekt datetime:

# In[72]:


ev_all['created_utc'] = pd.to_datetime(ev_all['created_utc'])


# Utworzenie kolumny 'year' do grupowania co rok:

# In[73]:


ev_all['year'] = ev_all['created_utc'].dt.to_period("Y")


# Średni sentyment dla każdego interwału czasowego (co rok):

# In[74]:


avg_sentiment = ev_all.groupby('year')['sentiment_score'].mean()


# Wykres:

# In[75]:


plt.figure(figsize=(12, 6))
sns.lineplot(x=avg_sentiment.index.astype(str), y=avg_sentiment.values, marker='o', color='purple')
plt.title('Średni sentyment w czasie w całym zbiorze')
plt.xlabel('Rok')
plt.ylabel('Średni sentyment')
plt.xticks(rotation=45)
plt.ylim(0, 0.5)
plt.show()


# ## Zależność sentymentu od liczby reakcji pod komentarzami

# Wybór kolumn:

# In[76]:


subset_comments = ev_comments[['score', 'comm_sentiment_score']]


# Korelację Pearsona:

# In[77]:


correlation_matrix_comments = subset_comments.corr()


# Macierz korelacji:

# In[78]:


print("Macierz korelacji:")
print(correlation_matrix_comments)


# ## Zależność sentymentu od liczby reakcji pod postem

# Wybór kolumn:

# In[79]:


subset_posts = ev_posts[['score', 'posts_sentiment_score']]


# Korelację Pearsona:

# In[80]:


correlation_matrix_posts = subset_posts.corr()


# Macierz korelacji:

# In[81]:


print("Macierz korelacji:")
print(correlation_matrix_posts)


# ## Złączenie obu tabel

# Wybranie kolumn z tabeli ev_comments:

# In[82]:


ev_comments = ev_comments[['link_id', 'subreddit', 'score', 'created_utc', 'comm_sentiment_score',
                         'comm_sentiment_label','body_processed']]


# Ustalenie nowych nazw kolumnw tabeli z komentarzami:

# In[83]:


new_column_names = {
    'link_id': 'post_id',
    'score': 'comm_score',
    'created_utc': 'comm_date',
    'body_processed': 'comm_text_content'
}


# Zmiana nazwy kolumn w tabeli z komentarzami:

# In[84]:


ev_comments.rename(columns=new_column_names, inplace=True)


# Wybranie kolumn z tabeli ev_posts:

# In[85]:


ev_posts = ev_posts[['name', 'score', 'num_comments','created_utc', 'posts_sentiment_score',
                         'posts_sentiment_label', 'selftext_processed', ]]


# Ustalenie nowych nazw kolumnw tabeli z postami:

# In[86]:


new_column_names_2 = {
    'name': 'post_id',
    'score': 'post_score',
    'num_comments' : 'comments_number',
    'created_utc': 'post_date',
    'selftext_processed': 'post_text_content'
}


# Zmiana nazwy kolumn w tabeli z komentarzami:

# In[87]:


ev_posts.rename(columns=new_column_names_2, inplace=True)


# Połączenie tabel:

# In[88]:


ev = pd.merge(ev_posts, ev_comments, on='post_id')


# In[89]:


ev


# ## Zależność sentymentu komentarzy od sentymentu postów

# In[90]:


correlation_matrix = ev[['posts_sentiment_score', 'comm_sentiment_score']].corr()


# In[91]:


print("Macierz korelacji:")
print(correlation_matrix)


# Grupowanie i zliczanie:

# In[92]:


subset_positive_posts = ev[ev['posts_sentiment_label'] == 'positive']['comm_sentiment_label'].value_counts(normalize=True) * 100
subset_negative_posts = ev[ev['posts_sentiment_label'] == 'negative']['comm_sentiment_label'].value_counts(normalize=True) * 100
subset_neutral_posts = ev[ev['posts_sentiment_label'] == 'neutral']['comm_sentiment_label'].value_counts(normalize=True) * 100


# Wykresy:

# In[93]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
plt.subplots_adjust(wspace=0.3)

# positive
axes[0].pie(subset_positive_posts, labels=subset_positive_posts.index, autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral', 'lightskyblue'])
axes[0].set_title('Rozkład sentymentu komentarzy w postach z pozytywnym sentymentem', fontsize=9)

# negative
axes[1].pie(subset_negative_posts, labels=subset_negative_posts.index, autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral', 'lightskyblue'])
axes[1].set_title('Rozkład sentymentu komentarzy w postach z negatywnym sentymentem', fontsize=9)

# neutral
axes[2].pie(subset_neutral_posts, labels=subset_neutral_posts.index, autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral', 'lightskyblue'])
axes[2].set_title('Rozkład sentymentu komentarzy w postach z neutralnym sentymentem', fontsize=9)

plt.show()


# # Modelowanie tematyczne

# ## Chmury słów postów i komentarzy

# Łączenie tekstu w jedną długą linię:

# In[94]:


text_post = ' '.join(ev_posts['post_text_content'])


# Inicjalizacja WordCloud z ograniczeniem do 10 słów:

# In[95]:


wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=10).generate(text_post)


# Wyświetlenie chmury słów:

# In[96]:


plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('10 najczęściej występujących słów w postach')
plt.show()


# Łączenie tekstu w jedną długą linię:

# In[97]:


text_comm = ' '.join(ev_comments['comm_text_content'])


# Inicjalizacja WordCloud z ograniczeniem do 10 słów:

# In[98]:


wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=10).generate(text_comm)


# Wyświetlenie chmury słów:

# In[99]:


plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('10 najczęściej występujących słów w komentarzach')
plt.show()


# ## Grupy tematyczne

# Przygotowanie danych dla postów:

# In[100]:


corpus_posts = ev_posts['post_text_content'].tolist()


# Wektoryzacja tekstu dla postów:

# In[101]:


vectorizer_posts = TfidfVectorizer(max_features=1000, stop_words='english')
X_posts = vectorizer_posts.fit_transform(corpus_posts)


# Modelowanie tematyczne (LDA) dla postów:

# In[102]:


lda_posts = LatentDirichletAllocation(n_components=3, random_state=42)
lda_posts.fit(X_posts)


# Przygotowanie danych dla komentarzy:

# In[103]:


corpus_comments = ev_comments['comm_text_content'].tolist()


# Wektoryzacja tekstu dla komentarzy:

# In[104]:


vectorizer_comments = TfidfVectorizer(max_features=1000, stop_words='english')
X_comments = vectorizer_comments.fit_transform(corpus_comments)


# Modelowanie tematyczne (LDA) dla komentarzy:

# In[105]:


lda_comments = LatentDirichletAllocation(n_components=3, random_state=42)
lda_comments.fit(X_comments)


# Wizualizacja tematów dla postów i komentarzy:

# In[106]:


feature_names_posts = vectorizer_posts.get_feature_names_out()
feature_names_comments = vectorizer_comments.get_feature_names_out()


# In[107]:


for topic_idx, topic in enumerate(lda_posts.components_):
    top_words_idx = topic.argsort()[:-7:-1]
    top_words = [feature_names_posts[i] for i in top_words_idx]
    print(f'Temat dla postów {topic_idx + 1}: {", ".join(top_words)}')


# In[108]:


for topic_idx, topic in enumerate(lda_comments.components_):
    top_words_idx = topic.argsort()[:-7:-1]
    top_words = [feature_names_comments[i] for i in top_words_idx]
    print(f'Temat dla komentarzy {topic_idx + 1}: {", ".join(top_words)}')


# ## Grupy tematyczne pozytywnych postów

# In[109]:


subset_positive_posts = ev_posts[ev_posts['posts_sentiment_label'] == 'positive']


# In[110]:


corpus_positive_posts = subset_positive_posts['post_text_content'].tolist()


# In[111]:


vectorizer_positive_posts = TfidfVectorizer(max_features=1000, stop_words='english')
X_positive_posts = vectorizer_positive_posts.fit_transform(corpus_positive_posts)


# In[112]:


lda_positive_posts = LatentDirichletAllocation(n_components=3, random_state=42)
lda_positive_posts.fit(X_positive_posts)


# In[113]:


feature_positive_posts = vectorizer_positive_posts.get_feature_names_out()


# In[114]:


for topic_idx, topic in enumerate(lda_positive_posts.components_):
    top_words_idx = topic.argsort()[:-7:-1]
    top_words = [feature_positive_posts[i] for i in top_words_idx]
    print(f'Temat dla postów {topic_idx + 1}: {", ".join(top_words)}')


# ### Chmura słów

# In[115]:


text_post_positive = ' '.join(subset_positive_posts['post_text_content'])


# In[116]:


wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=10).generate(text_post_positive)


# In[117]:


plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('10 najczęściej występujących słów w pozytywnych postach')
plt.show()


# ## Grupy tematyczne negatywnych postów

# In[118]:


subset_negative_posts = ev_posts[ev_posts['posts_sentiment_label'] == 'negative']


# In[119]:


corpus_negative_posts = subset_negative_posts['post_text_content'].tolist()


# In[120]:


vectorizer_negative_posts = TfidfVectorizer(max_features=1000, stop_words='english')
X_negative_posts = vectorizer_negative_posts.fit_transform(corpus_negative_posts)


# In[121]:


lda_negative_posts = LatentDirichletAllocation(n_components=3, random_state=42)
lda_negative_posts.fit(X_negative_posts)


# In[122]:


feature_negative_posts = vectorizer_negative_posts.get_feature_names_out()


# In[123]:


for topic_idx, topic in enumerate(lda_negative_posts.components_):
    top_words_idx = topic.argsort()[:-7:-1]
    top_words = [feature_negative_posts[i] for i in top_words_idx]
    print(f'Temat dla negatywnych postów {topic_idx + 1}: {", ".join(top_words)}')


# ### Chmura słów

# In[124]:


text_post_negative = ' '.join(subset_negative_posts['post_text_content'])


# In[125]:


wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=10).generate(text_post_negative)


# In[126]:


plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('10 najczęściej występujących słów w negatywnych postach')
plt.show()

