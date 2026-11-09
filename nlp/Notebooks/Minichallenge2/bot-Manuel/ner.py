import spacy

class ner:
    def __init__(self,):
        self.entities = []
        self.nlp = spacy.load('de_core_news_sm')

   # ner function with spacy
    def ner_spacy(self, text):
        doc = self.nlp(text)
        for ent in doc.ents:
            self.entities.append((ent.text, ent.label_))
        return self.entities
