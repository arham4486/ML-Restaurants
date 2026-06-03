import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
df_philly['success'] = (df_philly['stars']>= 4.0) & (df_philly['review_count'] >= 50)
success_counts = (df_philly['success'].value_counts())
print(success_counts)
df_philly['review_velocity'] = (df_philly['review_count']) / (df_philly['review_count'].max())
print(df_philly['review_velocity'].head())
df_philly['star_weight'] = (df_philly['stars']) * (df_philly['review_velocity'])
print(df_philly['star_weight'].head())