import spacy
import unicodedata
import regex as re
import string


class PreProcessor(object):
    def __init__(self, text):
        self.text = text
        regular_punct = list(string.punctuation)
        puncts_extra = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$',
                        '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',
                        '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`', '<',
                        '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â',
                        '█', '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢',
                        '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥',
                        '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’',
                        '▀', '¨', '▄', '♫', '☆', 'é', '¯',
                        '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
                        '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³',
                        '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']
        self.puncts = list(set(regular_punct + puncts_extra))

    def get_text(self):
        return self.text

    def spacy_tokenize_words(self):
        raise NotImplementedError

    def normalize_unicode(self):
        self.text = unicodedata.normalize('NFKD', self.text)
        return self

    def remove_newline(self):
        """
        remove \n and  \t
        """
        self.text = ' '.join(self.text.split())
        return self

    def decontracted(self):
        # specific
        text = re.sub(r"(W|w)on(\'|\’)t", "will not", self.text)
        text = re.sub(r"(C|c)an(\'|\’)t", "can not", text)
        text = re.sub(r"(Y|y)(\'|\’)all", "you all", text)
        text = re.sub(r"(Y|y)a(\'|\’)ll", "you all", text)

        # general
        text = re.sub(r"(I|i)(\'|\’)m", "i am", text)
        text = re.sub(r"(A|a)in(\'|\’)t", "aint", text)
        text = re.sub(r"n(\'|\’)t", " not", text)
        text = re.sub(r"(\'|\’)re", " are", text)
        text = re.sub(r"(\'|\’)s", " is", text)
        text = re.sub(r"(\'|\’)d", " would", text)
        text = re.sub(r"(\'|\’)ll", " will", text)
        text = re.sub(r"(\'|\’)t", " not", text)
        self.text = re.sub(r"(\'|\’)ve", " have", text)

        return self

    def clean_punctuation(self):
        """
        add space before and after punctuation and symbols
        """
        for punct in self.puncts:
            if punct in self.text:
                self.text = self.text.replace(punct, f' {punct} ')

        return self

    def clean_numbers(self):
        text = self.text
        if bool(re.search(r'\d', text)):
            text = re.sub('[0-9]{5,}', '#####', text)
            text = re.sub('[0-9]{4}', '####', text)
            text = re.sub('[0-9]{3}', '###', text)
            text = re.sub('[0-9]{2}', '##', text)
        self.text = text
        return self

# nlp = spacy.load('en_core_web_sm')
# doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')
# for token in doc:
#    print(token.text)

# special_case = [{ORTH: u'gim', LEMMA: u'give', POS: u'VERB'}, {ORTH: u'me'}]
# nlp.tokenizer.add_special_case(u'gimme', special_case)

# nlp = spacy.load('en_core_web_sm')
# doc = nlp(u"This is a sentence. This is another sentence.")
# for sent in doc.sents:
#     print(sent.text)
