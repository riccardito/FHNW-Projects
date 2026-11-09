import spacy
import intentmodel as model

class intentclassifier:
    def __init__(self):
        self.nlp = spacy.load("de_core_news_sm")

    def intent_preprocessing(self,text):
        # lowercase
        text = text.lower()
        # remove punctuation
        text = text.translate(str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))
        # remove stopwords
        doc = self.nlp(text)
        text = " ".join([token.text for token in doc if not token.is_stop])
        # lemmatize
        text = " ".join([token.lemma_ for token in self.nlp(text)])
        return text

    def predict(self, text):
        # model gives back a list of tuples (intent, probability)
        all_pr = model(self.intent_preprocessing(text))
        # sort the list by probability
        sorted_pr = sorted(all_pr, key=lambda x: x[1], reverse=True)
        # return the first element of the list
        return sorted_pr[0][0]



