import re

def clean_tweet(text):
    """
    Cleans tweet text by removing mentions, and special characters.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    # Remove Mentions (@user)
    text = re.sub(r'@\w+', '', text)
    
    # Remove basic special characters but keep hashtags content
    # Replace non-alphanumeric (except spaces) with nothing
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text