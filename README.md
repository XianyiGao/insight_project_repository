# Etsymator: Competitive Pricing for Etsy Merchants

## About the Project

"Etsymator" is a web application that provides price estimation for handmade items for Etsy merchants. The working web app is hosted in http://www.etsypriceestimator.com

This project provides a data driven solution to estimate handmade item price using a machine learning model trained with the historical data from Etsy.com. The model achieves error (RMSE) as small as $19 for items within $1000.

Applied Machine Learning Topics:

* Natural Language Process (e.g. TF-IDF)
* Random Forest Regression

### Project Motivation and Challenge

With the increase of e-commerce market share, it's important to provide a great user experience for both buyers and sellers. However, with a little search online, it's not hard to find seller's frustration on pricing their items especially handmade ones. While there are many qualitative instructions online about how to price handmade goods, there is no decent price estimator available.

Estimating the price for handmade items is very difficult because the item is highly customized. The price of a handmade item can depend on many aspects of the item such as material used and size of the item. It can also depend on how popular you are as a seller and how much effort you (or someone else) spent making the item. This is very different from estimating the price of a common goods such as a particular smartphone where you can find price reference quickly online.

## About the Data

* 44771 items were extracted from Etsy.com with all item prices less than $1000
* PostgreSQL was used for data management

### The Distribution of Items over Different Categories

<img src="/images/git6.png" width="600">

### Scatter Plot of Price vs. Who Made the Item ("none" is when the feature is missing)

<img src="/images/git1.png" width="600">

### Scatter Plot of Price vs. When the Item was Made ("none" is when the feature is missing)

<img src="/images/git3.png" width="600">

### Scatter Plot of Price vs. Seller Rating ("0" is when the seller do not have rating yet)

<img src="/images/git2.png" width="600">

## Implementation Details and Model Performance

Since the problem is to estimate the item price, I first considered using a simple linear regression model because it is easy interpretable. However, after exploratory data analysis (as some of the data distribution figures shown in above section), I quickly realized that linear regression is not a good option for the problem for the following reasons:

* Data is much more complicated than linear (e.g. price distribution changes non-linearly with the increase of seller scores)
* There are a lot of categorical data. (Linear regression does not work well with categorical data)
* There are many features available about the item and they can correlate with each other (e.g. 3 features on categories: category layer 1, category layer 2, and category later 3 that can relate to each other)

Although correlation amoung features can be solved by removing some of the features, this simple approach is not ideal because different features can add in additional information even if it is a bit related to some other features.

Therefore, I applied Random Forest Regression instead. It is a tree based regression that is robust against different data types and data scale. Tree based model also align with some of structures of feature data (e.g. 3 layers of item category that is meant to split hierarchically).



