# Web Application Source Code Directory

Simply run `./run.py` to host the web app locally using flask.

## Things you will need
1. Since the web app use the pre-trained machine learning model to estimate price, you need to obtain the model by using "etsy_item_price_estimation" package.

Alternatively, you can download a pre-trained machine learning model from the link:
http://bit.ly/machine_learning_model_Gao

Make sure you save the model as filename: "machine_learning_model". Then put the trained model under the "flask_etsy_app/static" folder.

2. PostgreSQL: make sure you have PostgreSQL installed and started. You extracted data should be pushed to PostgreSQL using modules from "etsy_item_price_estimation" package either locally or in the server.

## Host on AWS

Please refer to the link for formal AWS hosting:
https://aws.amazon.com/getting-started/projects/host-static-website/
