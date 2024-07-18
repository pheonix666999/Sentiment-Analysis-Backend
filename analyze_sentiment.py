import sys
import json
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
analyzer = SentimentIntensityAnalyzer()

# Positive and negative words
positive_words = [
    "beautiful", "well-maintained", "knowledgeable", "supportive", "wide range",
    "innovation", "vibrant", "welcoming", "excellent", "resources", "extracurricular",
    "well-equipped", "commendable", "helpful", "efficient", "convenient", "opportunities",
    "top-notch", "network", "alumni", "very helpful", "high-quality", "informative", 
    "technology", "excellent", "safe", "secure", "reputation", "well-deserved", "fantastic",
    "amazing", "awesome", "brilliant", "superb", "wonderful", "positive", "happy", "satisfied",
    "successful", "prosperous", "beneficial", "pleasant", "enjoyable", "grateful", "appreciative",
    "comfortable", "cozy", "delightful", "elegant", "fabulous", "friendly", "fun", "generous",
    "great", "harmonious", "incredible", "lovely", "magnificent", "marvelous", "outstanding",
    "peaceful", "perfect", "phenomenal", "remarkable", "rewarding", "satisfactory", "spectacular",
    "splendid", "stellar", "super", "terrific", "trustworthy", "uplifting", "valuable", "worthy",
    "accomplished", "admirable", "affectionate", "agreeable", "blissful", "breathtaking", "charming",
    "cheerful", "clean", "commendable", "confident", "conscientious", "considerate", "cool", "courteous",
    "creative", "dazzling", "decisive", "dedicated", "dependable", "diligent", "distinguished", "dynamic",
    "earnest", "eco-friendly", "effective", "efficient", "empathetic", "energetic", "engaging", "enthusiastic",
    "ethical", "exceptional", "exquisite", "fascinating", "flawless", "flexible", "focused", "forgiving", "funny",
    "gentle", "glorious", "graceful", "hardworking", "honest", "humble", "imaginative", "impressive", "inspirational",
    "intelligent", "joyful", "kind", "logical", "lovable", "loyal", "mindful", "motivated", "neat", "optimistic",
    "passionate", "patient", "perceptive", "persistent", "philanthropic", "polite", "powerful", "proactive",
    "productive", "radiant", "rational", "reliable", "resilient", "resourceful", "respectful", "responsible",
    "selfless", "sincere", "skilled", "sophisticated", "spectacular", "spirited", "successful", "talented", "thoughtful",
    "trustworthy", "understanding", "unique", "versatile", "vibrant", "wise", "youthful"
]


negative_words = [
    "outdated", "poorly maintained", "don't seem to care", "too high", "not enough support",
    "not safe", "severe lack", "too limited", "overpriced", "poor condition", "too many bureaucratic hurdles",
    "ineffective", "unhelpful", "nightmare", "expensive", "insufficient", "unhealthy", "inadequate", "hard to access",
    "awful", "bad", "boring", "confusing", "cruel", "dangerous", "deficient", "difficult", "dirty", "disappointing",
    "disastrous", "disgusting", "dreadful", "dull", "embarrassing", "exhausting", "frustrating", "horrible", "incompetent",
    "inconvenient", "inconsistent", "inefficient", "infuriating", "insensitive", "lacking", "lame", "lousy", "mediocre",
    "messy", "miserable", "neglectful", "negative", "offensive", "overcrowded", "overrated", "painful", "pathetic",
    "poor", "regretful", "ridiculous", "rude", "scary", "shabby", "shameful", "shoddy", "slow", "stressful", "stupid",
    "terrible", "ugly", "unacceptable", "unbearable", "uncomfortable", "undependable", "unfair", "unfriendly",
    "unhelpful", "unpleasant", "unprofessional", "unsatisfactory", "unsupportive", "untrustworthy", "upsetting",
    "useless", "weak", "worthless", "worse", "worst", "abysmal", "apathetic", "appalling", "atrocious", "careless",
    "clumsy", "depressing", "disgraceful", "haphazard", "heinous", "horrific", "hostile", "inadequate", "infuriating",
    "insecure", "irresponsible", "lazy", "manipulative", "monotonous", "nasty", "nauseating", "nonsense", "obnoxious",
    "off-putting", "oppressive", "overbearing", "pathetic", "perplexing", "petty", "pitiful", "precarious", "risky",
    "silly", "spiteful", "substandard", "tedious", "threatening", "troublesome", "unbearable", "uncaring", "uncontrolled",
    "undermined", "unethical", "unforgiving", "ungrateful", "unimpressive", "unjust", "unnecessary", "unreliable",
    "unsightly", "untenable", "unwanted", "unworthy", "vague", "vicious", "vindictive", "volatile", "wasteful", "woeful"
]


def analyze_sentiment(text):
    doc = nlp(text)
    sentiment_scores = analyzer.polarity_scores(doc.text)
    
    positive_score = sum([1 for word in positive_words if word in doc.text.lower()])
    negative_score = sum([1 for word in negative_words if word in doc.text.lower()])
    
    sentiment_scores['compound'] += (positive_score - negative_score) * 0.1  # Adjust compound score
    
    return sentiment_scores

if __name__ == "__main__":
    input_text = sys.stdin.read()
    sentiment = analyze_sentiment(input_text)
    print(json.dumps(sentiment))
