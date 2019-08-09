import datetime
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
import sklearn
from konlpy.tag import Mecab

categories = {"모바일":"731", "인터넷/SNS":"226", #subcategories of IT news: mobile, internet, etc.
              "통신/뉴미디어":"227", "IT 일반":"230", "보안/해킹":"732", 
              "컴퓨터":"283", "게임/리뷰":"229", "과학 일반":"228"} 
main_data = pd.DataFrame([])
main_url = 'https://news.naver.com/main/list.nhn?mode=LS2D&mid=shm&sid1=105&sid2=' #main page of IT news
link = ""
dates = []
today = datetime.datetime.now()
for date_iteration in range(10): #crawls through articles from last 10 days of each subcategory
    date = today.date() - date_iteration * datetime.timedelta(days=1)
    date_string = re.sub('\-', '',str(date))
    dates.append(date_string) #list of last 10 dates
for category in categories:
    category_url = main_url + categories[category]
    for news_day in dates:
        date_url = category_url + '&date=' + news_day
        page_number_response = requests.get(date_url)
        page_number_html = page_number_response.content
        max_pages_finding_soup = BeautifulSoup(page_number_html,'lxml')
        pages_list = max_pages_finding_soup.find_all("div", class_="paging")
        num_list = pages_list[0].get_text().strip().split('\n') #maximum 10 pages per date
        if "다음" in num_list: 
            num_list.remove("다음") #removes "next" from list of page number strings
        integer_pages = []
        for string_page in num_list:
            integer_pages.append(int(string_page)) #turns all pages numbers to integers to find max
        max_page = max(integer_pages)
        for page_num in range (1, max_page + 1): #crawls through page 1 to maximum page number
            page_url = date_url + '&page=' + str(page_num) #turns page number back to string to access URL
            page_response = requests.get(page_url)
            page_html = page_response.content
            page_soup = BeautifulSoup(page_html,'lxml')
            articles = page_soup.find_all('a')
            for article in articles:
                text_category_data = {}
                if article.parent.name == 'dt':
                    next_link = article.get('href')
                    if next_link != link:
                        link = next_link
                        article_response = requests.get(link)
                        article_html = article_response.content
                        article_soup = BeautifulSoup(article_html,'lxml')
                        raw_body = article_soup.find(id = ['articleBodyContents', 'articeBody'])
                        unwanted_element = raw_body.find('script') 
                        if unwanted_element != None:
                            unwanted_element.extract() #removes script text from article body text
                        body_text = raw_body.get_text().strip()
                        body_text = re.sub(r'\s+', ' ', re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', body_text))
                        text_category_data["text"] = body_text
                        text_category_data["category"] = category
                        article_data = pd.DataFrame([text_category_data])
                        main_data = pd.concat([main_data, article_data])

final_data = pd.DataFrame([])
for main_index in range(len(main_data)): #run through each item in main_data
    cleaned_text = ""
    tokenized_data = pd.DataFrame([])
    category = main_data.iloc[main_index]["category"] #gets category name
    article_text = main_data.iloc[main_index]["text"] #gets article text
    processed_text = (" ").join(mecab.nouns(article_text)) #gets list of nouns from text and joins into single string
    article_data = {}
    article_data["text"] = processed_text
    article_data["category"] = category
    text_category_data = pd.DataFrame([article_data])
    final_data = pd.concat([final_data, text_category_data])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    final_data['text'], 
    final_data['category'], 
    train_size = .8
)
print("Train set: ", X_train.shape[0])
print("Test set: ", X_test.shape[0])

# libraries for TF-IDF with SVM and Naive-Bayes
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

#SVM
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)
tfidf_vect = TfidfVectorizer(max_features=5000)
tfidf_vect.fit(final_data['text'])
X_train_tfidf = tfidf_vect.transform(X_train)
X_test_tfidf = tfidf_vect.transform(X_test)
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(X_train_tfidf,y_train) # predict the labels on validation dataset
predictions_SVM = SVM.predict(X_test_tfidf) # Use accuracy_score function to get the accuracy
print("SVM Accuracy Score: ",accuracy_score(predictions_SVM, y_test)*100)

#Naive-Bayes
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_tfidf,y_train)
predictions = naive_bayes.predict(X_test_tfidf)
print("\nNaive-Bayes Scores:")
print("Accuracy: ", accuracy_score(y_test, predictions))
print("Recall: ", recall_score(y_test, predictions, average = 'weighted'))
print("Precision: ", precision_score(y_test, predictions, average = 'weighted'))
print("F1 score: ", f1_score(y_test, predictions, average = 'weighted'))

