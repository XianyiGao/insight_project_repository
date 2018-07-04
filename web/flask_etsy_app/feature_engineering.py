"""
This module focus on elements related to features and feature engineering
"""


import pickle


class FeatureLabels:
    """
    This class enables pulling out feature labels easily. As there are one-hot
    features, text features, and numerical features, handling all these can be
    not straight forward. These features are used in the web app.
    """

    def __init__(self, data_filename):
        """
        Loading pre-stored features from a file.
        """
        with open(data_filename, 'rb') as data_file:
            loaded_features = pickle.load(data_file)
        self.title_nlp_tfidf_features = loaded_features['title_NLP_TFIDF_features']
        self.other_features = loaded_features['other_features']
        self.category1_features = loaded_features['category1_features']
        self.category2_features = loaded_features['category2_features']
        self.category3_features = loaded_features['category3_features']
        self.material_features = loaded_features['material_features']
        self.who_made_features = loaded_features['whoMade_features']
        self.when_made_features = loaded_features['whenMade_features']
        self.style1_features = loaded_features['style1_features']
        self.style2_features = loaded_features['style2_features']
        self.feature_labels = loaded_features['feature_labels']

    def get_title(self):
        """
        Get labels related to tfidf features, 500 features
        """
        return self.title_nlp_tfidf_features

    def get_other_features(self):
        """
        Get labels related to other features such as whether it is digital
        """
        return self.other_features

    def get_cat1_features(self):
        """
        Get labels related to category layer 1 as one-hot encoded features
        """
        return self.category1_features

    def get_cat2_features(self):
        """
        Get labels related to category layer 2 as one-hot encoded features
        """
        return self.category2_features

    def get_cat3_features(self):
        """
        Get labels related to category layer 3 as one-hot enconded features
        """
        return self.category3_features

    def get_material_features(self):
        """
        Get labels related to item material
        """
        return self.material_features

    def get_who_features(self):
        """
        Get labels related to who made the item
        """
        return self.who_made_features

    def get_when_features(self):
        """
        Get labels related to when the item was made
        """
        return self.when_made_features

    def get_style1_features(self):
        """
        Get labels related to the first style of this item, one-hot encoded
        """
        return self.style1_features

    def get_style2_features(self):
        """
        Get labels related to the second style of this item (if there is any),
        one-hot encoded.
        """
        return self.style2_features

    def get_feature_labels(self):
        """
        Get features labels (all combined to a list)
        """
        return self.feature_labels
