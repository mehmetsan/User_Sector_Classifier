from langdetect import detect

# Common content of the to be filtered sites
filters = ["godaddy","access denied","accessdenied","please sign", "please enable", "sign into","lasik"]

def check_english(text):
    try:
        return detect(text) == "en"
    except:    
        return False

def apply_filters(text):
    flag = True
    for each in filters:
        if each in text:
            flag = False
            break
    return flag