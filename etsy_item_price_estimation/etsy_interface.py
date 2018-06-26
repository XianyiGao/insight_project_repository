
# coding: utf-8

# In[ ]:


import urllib
from urllib.parse import unquote, parse_qs
import json
import logging
import requests
#from oauth_hook import OAuthHook
from requests_oauthlib import OAuth1

log = logging.getLogger(__name__)

class Etsy(object):
    """
    Represents the etsy API
    """
    url_base = "https://openapi.etsy.com/v2"
    
    class EtsyError(Exception):
        pass
    
    def __init__(self, consumer_key, consumer_secret, oauth_token=None, oauth_token_secret=None, sandbox=False):
        self.params = {'api_key': consumer_key}
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        
        if sandbox:
            self.url_base = "http://sandbox.openapi.etsy.com/v2"
        
        # generic authenticated oauth hook
        self.simple_oauth = OAuth1(client_key=consumer_key, client_secret=consumer_secret)
        
        if oauth_token and oauth_token_secret:
            # full oauth hook for an authenticated user
            self.full_oauth = OAuth1(oauth_token, oauth_token_secret, consumer_key, consumer_secret)
    
    def show_listings(self, limit, offset, color=None, color_wiggle=5):
        """
        Show all listings on the site.
        color should be a RGB ('#00FF00') or a HSV ('360;100;100')
        """
        endpoint = '/listings/active' # + 'limit=' + limit + '&' + 'offset=' + offset
        if color:
            self.params['color'] = color
            self.params['color_accuracy'] = color_wiggle
        self.params['limit'] = limit
        self.params['offset'] = offset
        print(endpoint)
        print(self.simple_oauth)
        response = self.execute(endpoint)
        return json.loads(response.text)
    
    def get_user_info(self, user):
        """
        Get basic info about a user, pass in a username or a user_id
        """
        endpoint = '/users/%s' % user
        
        auth = {}
        if user == '__SELF__':
            auth = {'oauth': self.full_oauth}
            self.params = {} # etsy api ignores oauth if api_key is present in get params 
            
        response = self.execute(endpoint, **auth)
        return json.loads(response.text)
    
    def find_user(self, keywords):
        """
        Search for a user given the 
        """
        endpoint = '/users'
        self.params['keywords'] = keywords
        response = self.execute(endpoint)
        return json.loads(response.text)
    
    def get_auth_url(self, permissions=[]):
        """
        Returns a url that a user is redirected to in order to authenticate with
        the etsy API. This is step one in the authentication process.
        oauth_token and oauth_token_secret need to be saved for step two.
        """
        endpoint = '/oauth/request_token'
        self.params = {}
        if permissions:
            self.params = {'scope': " ".join(permissions)}
        response = self.execute(endpoint, oauth=self.oauth)
        parsed = parse_qs(response.text)
        url = parsed['login_url'][0]
        token = parsed['oauth_token'][0]
        secret = parsed['oauth_token_secret'][0]
        return {'oauth_token': token, 'url': url, 'oauth_token_secret': secret}
    
    def get_auth_token(self, verifier, oauth_token, oauth_token_secret):
        """
        Step two in the authentication process. oauth_token and oauth_token_secret
        are the same that came from the get_auth_url function call. Returned is
        the permanent oauth_token and oauth_token_secret that will be used in
        every subsiquent api request that requires authentication.
        """
        endpoint = '/oauth/access_token'
        self.params = {'oauth_verifier': verifier}
        oauth = OAuth1(oauth_token, oauth_token_secret, self.consumer_key, self.consumer_secret)
        response = self.execute(endpoint, method='post', oauth=oauth)
        parsed = parse_qs(response.text)
        return {'oauth_token': parsed['oauth_token'][0], 'oauth_token_secret': parsed['oauth_token_secret'][0]}
        
    
    def execute(self, endpoint, method='get', oauth=None):
        """
        Actually do the request, and raise exception if an error comes back.
        """
        querystring = urllib.parse.urlencode(self.params)
        url = "%s%s" % (self.url_base, endpoint)
        if querystring:
            url = "%s?%s" % (url, querystring)
        print(url)
        hooks = {}
        if oauth:
            # making an authenticated request, add the oauth hook to the request
            hooks = {'hooks': {'pre_request': oauth}}
        
        response = getattr(requests, method)(url, **hooks)
        
        if response.status_code > 201:
            e = response.text
            code = response.status_code
            raise self.EtsyError('API returned %s response: %s' % (code, e))
        return response

