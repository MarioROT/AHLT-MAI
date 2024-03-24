#! /usr/bin/python3

import sys
import re
from os import listdir

from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize

from collections import Counter 

import nltk
nltk.download('averaged_perceptron_tagger')

## dictionary containig information from external knowledge resources
## WARNING: You may need to adjust the path to the resource files
external = {}
with open("../resources/HSDB.txt", encoding="utf-8") as h :
    for x in h.readlines() :
        external[x.strip().lower()] = "drug"
with open("../resources/DrugBank.txt", encoding="utf-8") as h :
    for x in h.readlines() :
        (n,t) = x.strip().lower().split("|")
        external[n] = t

def read_entities(datadir) :
    sentences = {}
    # process each file in input directory
    for f in listdir(datadir) :   
        # parse XML file, obtaining a DOM tree
        tree = parse(datadir+"/"+f)
        #print(tree.getElementsByTagName("sentence")[0].attributes)
        # process each sentence in the file
        for sentence in tree.getElementsByTagName("entity"):
            if not sentence.attributes['type'].value in sentences.keys():
                sentences[sentence.attributes["type"].value] = [sentence.attributes['text'].value]
            else:
                sentences[sentence.attributes["type"].value].append(sentence.attributes['text'].value)
                        
    return sentences

def generate_dicts(data_dir):
    final_lists = {'suffixes': {}, 'prefixes':{}}
    sentences = read_entities(data_dir)
    final_lists['suffixes']['drug'] = [i[0] for i in Counter(word[-5:] for word in sentences['drug']).most_common(40)]
    final_lists['suffixes']['drug_n'] = [i[0] for i in Counter(word[-5:] for word in sentences['drug_n']).most_common(7)]
    final_lists['suffixes']['brand'] = [i[0] for i in Counter(word[-5:] for word in sentences['brand']).most_common(10)]
    final_lists['suffixes']['group'] = [i[0] for i in Counter(word[-5:] for word in sentences['group']).most_common(70)]
    final_lists['prefixes']['drug'] = [i[0] for i in Counter(word[:3] for word in sentences['drug']).most_common(10)]
    final_lists['prefixes']['drug_n'] = [i[0] for i in Counter(word[:3] for word in sentences['drug_n']).most_common(10)]
    final_lists['prefixes']['brand'] = [i[0] for i in Counter(word[:3] for word in sentences['brand']).most_common(14)]
    final_lists['prefixes']['group'] = [i[0] for i in Counter(word[:3] for word in sentences['group']).most_common(10)]

    return final_lists

trends = generate_dicts('../data/train')

## --------- tokenize sentence ----------- 
## -- Tokenize sentence, returning tokens and span offsets

def tokenize(txt):
    offset = 0
    tks = []
    ## word_tokenize splits words, taking into account punctuations, numbers, etc.
    for t in word_tokenize(txt):
        ## keep track of the position where each token should appear, and
        ## store that information with the token
        offset = txt.find(t, offset)
        tks.append((t, offset, offset+len(t)-1))
        offset += len(t)

    ## tks is a list of triples (word,start,end)
    return tks


## --------- get tag ----------- 
##  Find out whether given token is marked as part of an entity in the XML

def get_tag(token, spans) :
    (form,start,end) = token
    for (spanS,spanE,spanT) in spans :
        if start==spanS and end<=spanE : return "B-"+spanT
        elif start>=spanS and end<=spanE : return "I-"+spanT

    return "O"

def get_unique_in_order(seq):
    # get unique chars in the original order
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def get_vowel_to_consonant_ratio(word):
    vowels = "aeiou"
    vowel_count = sum(1 for char in word if char in vowels)
    consonant_count = sum(1 for char in word if char.isalpha() and char not in vowels)
    if consonant_count == 0:
        return 0
    return vowel_count / consonant_count


def is_chemical_name(token):
    chemical_suffixes = ['ium', 'ine', 'ide', 'ate', 'ite', 'ogen', 'acid']
    # Simple heuristic - you could use a more sophisticated check
    return any(token.lower().endswith(suffix) for suffix in chemical_suffixes)


## --------- Feature extractor ----------- 
## -- Extract features for each token in given sentence
def extract_features(tokens):
    window_size=1

    nltk_pos_tags = nltk.pos_tag([token[0] for token in tokens])

    # for each token, generate list of features and add it to the result
    result = []
    for k in range(0, len(tokens)):
        tokenFeatures = []
        t = tokens[k][0]
        pos_tag = nltk_pos_tags[k][1]

        # Basic features
        tokenFeatures.append("form=" + t)
        tokenFeatures.append("formlower=" + t.lower())  # Lowercased form
        tokenFeatures.append("suf3=" + t[-3:])
        if len(t) >= 4:
            tokenFeatures.append("suf4=" + t[-4:])
        if t.lower() in external : tokenFeatures.append("external=" + external[t.lower()])
        
        if t.istitle(): tokenFeatures.append("isTitle")  # Is Title
        if t.isupper(): tokenFeatures.append("isUpper")  # Is Title
        if t.islower(): tokenFeatures.append("isLower")  # Is Title

        if t[-5:] in trends['suffixes']['drug_n']: tokenFeatures.append("trend=" + "drug_n")
        elif t[-5:] in trends['suffixes']['drug']: tokenFeatures.append("trend=" + "drug")
        elif t[-5:] in trends['suffixes']['brand']: tokenFeatures.append("trend=" + "brand")
        elif t[-5:] in trends['suffixes']['group']: tokenFeatures.append("trend=" + "group")

        if t[-5:] in trends['prefixes']['drug_n']: tokenFeatures.append("trend=" + "drug_n")
        elif t[-5:] in trends['prefixes']['drug']: tokenFeatures.append("trend=" + "drug")
        if t[-5:] in trends['prefixes']['brand']: tokenFeatures.append("trend=" + "brand")
        elif t[-5:] in trends['prefixes']['group']: tokenFeatures.append("trend=" + "group")

        # Prefix features
        tokenFeatures.append("pref3=" + t[:3])  # First 3 characters
        tokenFeatures.append("pref2=" + t[:2])  # First 2 characters
        tokenFeatures.append("pref1=" + t[:1])  # First character

        # BoS and EoS
        if k==0:
            tokenFeatures.append("BoS")
        elif k+1==len(tokens):
            tokenFeatures.append("EoS")

        # Word shape features
        word_shape = get_word_shape(t)
        word_shape = str(get_unique_in_order(word_shape))
        tokenFeatures.append("word_shape=" + word_shape)

        # special chars - no difference
        if '%' in t:
            tokenFeatures.append("percentage")
        if '/' in t:
            tokenFeatures.append("slash")
        if '#' in t:
            tokenFeatures.append("hash")
        if '-' in t:
            tokenFeatures.append("dash")

        if any(sub in t.lower() for sub in ['alpha', 'beta', 'gamma']):
            tokenFeatures.append('hasChemicalModifier')     

        # for i in range(1, min(6, len(t)+1)):
        #     tokenFeatures.append(f"prefix{i}={t[:i]}")
        #     tokenFeatures.append(f"suffix{i}={t[-i:]}")

        # if re.search(r'\d', t):  # Contains a digit
        #     tokenFeatures.append("containsDigit")
        # if re.search(r'[A-Za-z]+\d+', t):  # Letter(s) followed by digit(s)
        #     tokenFeatures.append("letterFollowedByDigit")
        # if re.search(r'\(', t) or re.search(r'\)', t):  # Contains parentheses
        #     tokenFeatures.append("containsParentheses")

        # if t.isupper():
        #     tokenFeatures.append("allCaps")
        # elif t[0].isupper() and t[1:].islower():
        #     tokenFeatures.append("initCap")
        # elif any(char.isupper() for char in t[1:]):
        #     tokenFeatures.append("innerCap")

        # if is_chemical_name(t):
        #     tokenFeatures.append('isChemicalName')

        # Orthographic features
        # tokenFeatures.append("isTitle=" + str(t.istitle()))
        # tokenFeatures.append("isLower=" + str(t.islower()))
        # tokenFeatures.append("hasDigit=" + str(any(char.isdigit() for char in t)))
        tokenFeatures.append("hasDash=" + str('-' in t))
        tokenFeatures.append("hasSpecial=" + str(any(not char.isalnum() and char != '-' for char in t)))

        # # Token length
        tokenFeatures.append("length=" + str(len(t)))
        # if len(t) > 16: tokenFeatures.append("length=12")

        # tokenFeatures.append('numCaps={}'.format(sum(1 for char in t if char.isupper())))
        # tokenFeatures.append('numSpecialChars={}'.format(sum(1 for char in t if not char.isalnum())))

        # # Vowel-to-consonant ratio
        # vcr = get_vowel_to_consonant_ratio(t)
        # tokenFeatures.append("vowelConsonantRatio=" + str(vcr))

        # # POS Tagging feature
        # tokenFeatures.append("POS=" + pos_tag)

        # # Character N-Grams (2-grams as example)
        # char_ngrams = [t[i:i+2] for i in range(len(t)-1)]
        # for ngram in char_ngrams:
        #     tokenFeatures.append("char_ngram=" + ngram)

        # # Orthographic features
        # if any(char.isupper() for char in t[1:]):
        #     tokenFeatures.append("internalUpper")
        if any(char.isdigit() for char in t):
            tokenFeatures.append("hasDigit")

        # Context words

        for i in range(1, window_size + 1):
            if k - i >= 0:
                tPrev = tokens[k - i][0]
                tokenFeatures.append("formPrev" + str(i) + "=" + tPrev)
                tokenFeatures.append("formlowerPrev" + str(i) + "=" + tPrev.lower())  # Lowercased form of previous token
                tokenFeatures.append("suf3Prev" + str(i) + "=" + tPrev[-3:])
                if len(tPrev) >= 4:
                    tokenFeatures.append("suf4Prev" + str(i) + "=" + tPrev[-4:])
                if tPrev.lower() in external : tokenFeatures.append("externalPrev"+ str(i) + "=" + external[tPrev.lower()])
            if k - i == 0:
                tokenFeatures.append("BoS" + str(i))

            if k + i < len(tokens):
                tNext = tokens[k + i][0]
                tokenFeatures.append("formNext" + str(i) + "=" + tNext)
                tokenFeatures.append("formlowerNext" + str(i) + "=" + tNext.lower())  # Lowercased form of next token
                tokenFeatures.append("suf3Next" + str(i) + "=" + tNext[-3:])
                if len(tNext) >= 4:
                    tokenFeatures.append("suf4Next" + str(i) + "=" + tNext[-4:])
                if tNext.lower() in external : tokenFeatures.append("externalNext"+ str(i) + "=" + external[tNext.lower()])
            if k + i + 1== len(tokens):
                tokenFeatures.append("EoS" + str(i))

        result.append(tokenFeatures)

    return result


def get_word_shape(word):
    word_shape = ""
    for char in word:
        if char.isupper():
            word_shape += "X"  # Uppercase letter
        elif char.islower():
            word_shape += "x"  # Lowercase letter
        elif char.isdigit():
            word_shape += "d"  # Digit
        # else:
        #     word_shape += char  # Non-alphanumeric
    return word_shape


## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  baseline-NER.py target-dir
## --
## -- Extracts Drug NE from all XML files in target-dir, and writes
## -- them in the output format requested by the evalution programs.
## --


# directory with files to process
datadir = sys.argv[1]

# process each file in directory
for f in listdir(datadir) :
   
    # parse XML file, obtaining a DOM tree
    tree = parse(datadir+"/"+f)
   
    # process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences :
        sid = s.attributes["id"].value   # get sentence id
        spans = []
        stext = s.attributes["text"].value    # get sentence text
        entities = s.getElementsByTagName("entity")
        for e in entities :
            # for discontinuous entities, we only get the first span
            # (will not work, but there are few of them)
            (start,end) = e.attributes["charOffset"].value.split(";")[0].split("-")
            typ =  e.attributes["type"].value
            spans.append((int(start),int(end),typ))
            

        # convert the sentence to a list of tokens
        tokens = tokenize(stext)
        # extract sentence features
        features = extract_features(tokens)

        # print features in format expected by crfsuite trainer
        for i in range (0,len(tokens)) :
            # see if the token is part of an entity
            tag = get_tag(tokens[i], spans) 
            print (sid, tokens[i][0], tokens[i][1], tokens[i][2], tag, "\t".join(features[i]), sep='\t')

        # blank line to separate sentences
        print()
