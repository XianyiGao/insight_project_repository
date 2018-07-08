# Etsymator: Competitive Pricing for Etsy Merchants

## About the Project

"Etsymator" is a web application that provides price estimation for handmade items for Etsy merchants. The working web app is hosted in http://www.etsypriceestimator.com

This project provides a data driven solution to estimate handmade item price using a machine learning model trained with the historical data from Etsy.com. The model achieves error (RMSE) as small as $19 for items within $1000.

Applied Machine Learning Topics:

* Natural Language Process (e.g. TF-IDF)
* Random Forest Regression

#### Project Motivation and Challenge

With the increase of e-commerce market share, it's important to provide a great user experience for both buyers and sellers. However, with a little search online, it's not hard to find seller's frustration on pricing their items especially handmade ones. While there are many qualitative instructions online about how to price handmade goods, there is no decent price estimator available.

Estimating the price for handmade items is very difficult because the item is highly customized. The price of a handmade item can depend on many aspects of the item such as material used and size of the item. It can also depend on how popular you are as a seller and how much effort you (or someone else) spent making the item. This is very different from estimating the price of a common goods such as a particular smartphone where you can find price reference quickly online.

## About the Data

* 44771 items were extracted from Etsy.com with all item prices less than $1000
* PostgreSQL was used for data management


## Implementation Details and Model Performance

