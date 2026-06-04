from tkinter import _test
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
df = pd.read_json('yelp_academic_dataset_business.json', lines=True)
print(df.head())
print(df.columns)
print(df.shape)
Restaurants = df[df['categories'].str.contains('Restaurants', na=False)]
print(Restaurants.shape)
Restaurants_null = Restaurants.isnull()
Restaurants_null = Restaurants_null.sum()
print(Restaurants_null)
df_need = Restaurants[[ 'name', 'city', 'stars', 'review_count', 'categories', 'latitude', 'longitude', 'is_open']]
print(df_need)
df_open = df_need[df_need['is_open'] == 1]
print(df_open.shape)
df_average = df_open['stars'].mean()
print(df_average)
df_cities = df_open['city'].value_counts()
print(df_cities)
df_philly = df_open[df_open['city'] == 'Philadelphia']
print(df_philly.shape)
plt.figure()
df_philly['stars'].hist()
plt.title('restaurant rating')
plt.xlabel('rating')
plt.ylabel('restaurant')
#plt.show()
dfphilly_best = df_philly.nlargest(10, ['stars', 'review_count'])
print(dfphilly_best)
def recommend_restaurants(min_stars):
    filtered = df_philly[df_philly['stars'] >= min_stars]
    return filtered.sort_values('review_count', ascending = False).head(5)
print(recommend_restaurants(4.0))
df_philly['success'] = (df_philly['stars']>= 3.5) & (df_philly['review_count'] >= 100)
success_counts = (df_philly['success'].value_counts())
print(success_counts)
df_philly['review_velocity'] = (df_philly['review_count']) / (df_philly['review_count'].max())
print(df_philly['review_velocity'].head())
df_philly['star_weight'] = (df_philly['stars']) * (df_philly['review_velocity'])
print(df_philly['star_weight'].head())
dfphilly_cuisine = df_philly['categories'].str.split(' , ').str[0].fillna('Other')
df_philly['cuisine'] = dfphilly_cuisine 
ddf = pd.get_dummies(df_philly['cuisine'])
print(ddf)
df_philly['is_pizza'] = df_philly['categories'].str.contains('pizza', na = False)
df_philly['is_chinese'] = df_philly['categories'].str.contains('chinese', na = False)
df_philly['is_mexican'] = df_philly['categories'].str.contains('mexican', na = False)
df_philly['is_italian'] = df_philly['categories'].str.contains('italian', na = False)
x = df_philly[['latitude', 'longitude', 'is_pizza', 'is_italian', 'is_mexican', 'is_chinese']]
y = df_philly['success']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2 , random_state = 42)
print(x_test.shape)
forest_model = RandomForestClassifier(random_state= 42, class_weight = 'balanced').fit(x_train, y_train)
y_pred = forest_model.predict(x_test)
print(y_pred)
y_report = classification_report(y_test, y_pred)
print(y_report)
df_tampa = df_open[df_open['city'] == 'Tampa']
print(df_tampa)
df_tampa['success'] = (df_tampa['stars'] >= 3.5) & (df_tampa['review_count'] >= 25)
df_tampa['star_weight'] = (df_tampa['stars']) * (df_tampa['review_count'])
df_tampa['review_velocity'] = (df_tampa['review_count']) / (df_tampa['review_count'].max())
tampa_cuisine = df_tampa['categories'].str.split(' , ').str[0].fillna('Other')
df_tampa['cuisine'] = tampa_cuisine
df_tampa['is_pizza'] = df_tampa['categories'].str.contains('pizza', na = False)
df_tampa['is_chinese'] = df_tampa['categories'].str.contains('chinese', na = False)
df_tampa['is_mexican'] = df_tampa['categories'].str.contains('mexican', na = False)
df_tampa['is_italian'] = df_tampa['categories'].str.contains('italian', na = False)
x_tampa = df_tampa[['latitude', 'longitude', 'is_pizza', 'is_italian', 'is_mexican', 'is_chinese']]
y_tampa = df_tampa['success']
y_tampapred = forest_model.predict(x_tampa)
print(y_tampapred)
ytampa_report = classification_report(y_tampa, y_tampapred, zero_division = 0)
print(ytampa_report)
tampa_successrate = ytampa_report.mean()
philly_successrate = y_report.mean() 
