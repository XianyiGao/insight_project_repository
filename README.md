# Etsymator: Competitive Pricing for Etsy Merchants

## About the Project

"Etsymator" is a web application that provides price estimation for handmade items for Etsy merchants. This is an insight data science project that was built in two weeks. The working web app is hosted on http://www.etsypriceestimator.com

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

### This figure shows predicted item price vs. true item price on the testing data (20% of total data). It has a good RMSE of $34 over $1000 range of item price. However, there are still quite some outliers in the upper left conner of the figure.

<img src="/images/git8.png" width="600">

After revisiting available data, one key part that I can still use is the item title, but it’s in the text form. Therefore, natural language processing is needed to process. There are several commonly used NLP approaches:

* **word2vec**: A popular NLP algorithm to estimate the semantic similarity of words or sentences. This algorithm requires to train on the whole collection of documents to learn which words are similar to each other based on the context. While this is a good approach for semantic analysis, I would need exact word matching instead for my project.
* **Latent Dirichlet Allocation (LDA)**: A popular NLP algorithm for document classification. This algorithm compares words in different documents (item title in this case) and classifies them based on the similarity. Since I need a vector representation of the item title but not for classification, it does not match with this project.
* **TF-IDF**: This algorithm computes the frequency of words occurenting in the item title and it can convert the item title into a list of word features. It also considers the most common words and weights them less in the word vector. Therefore, **I applied this algorithm** for item title analysis.

### This figure shows predicted item price vs. true item price on the testing data (20% of total data) with improved prediction accuracy after applying TF-IDF. Now the RMSE drops to $19 and there are much fewer outliers in the figure.

<img src="/images/git9.png" width="600">

### The feature importance based on the trained machine learning model

<img src="/images/git7.png" width="800">

## Web Application

The final product of this project is a web app, currently hosted on http://www.etsypriceestimator.com

### The screenshot of the web app

<img src="/images/git10.png" width="1000">

### Example input to the web app: model can handle missing data and random string in "number only" fields

<img src="/images/git11.png" width="500">
