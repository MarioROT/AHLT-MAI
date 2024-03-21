#! /usr/bin/python3

import sys
from typing import final
sys.path.append('../')
from os import listdir,system
import re

from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize

import util.evaluator as evaluator

import neptune
from dotenv import dotenv_values
from collections import Counter 

import string
import nltk
nltk.download('stopwords')

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
 

## --------- tokenize sentence ----------- 
## -- Tokenize sentence, returning tokens and span offsets

def tokenize(txt):
    offset = 0
    tks = []
    for t in word_tokenize(txt):
    # stopwords=set(nltk.corpus.stopwords.words('english'))
    # signs = string.punctuation
    # minwords_len = 2

    # for t in clean_sentence(txt, False, stopwords, minwords_len, False):
        offset = txt.find(t, offset)
        tks.append((t, offset, offset+len(t)-1))
        offset += len(t)
    return tks

## -----------------------------------------------
## -- check if a token is a drug part, and of which type

def remove_signs(wrd,signs):
        wrd = list(wrd)
        wrd = [word for word in wrd if not any(caracter in signs for caracter in word)]
        wrd = ''.join(wrd)
        return wrd

def clean_sentence(sentence, lowercase = True, stopwords = False, minwords_len = False, signs = False):
    sentence = sentence.split(' ')
    if lowercase:
        sentence = [word.lower() for word in sentence]
    if signs:
        sentence = [word if not any(caracter in signs for caracter in word) else remove_signs(word, signs) for word in sentence]
    if stopwords:
        sentence = [word for word in sentence if word not in stopwords and word.isalpha()]
    if minwords_len:
        sentence = [word for word in sentence if len(word) > minwords_len]
    return sentence

# suffixes = ['azole', 'idine', 'amine', 'mycin']
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

def classify_token(txt):

   # WARNING: This function must be extended with 
   #          more and better rules
   # match_external = [key for key in external.keys() if txt.lower() in key]
   num_re = re.compile(r'\-?\d{1,10}\.?\d{0,10}')

   if txt.lower() in external : return external[txt.lower()]
   # if match_external: return external[match_external[0]]
   if len(tokenize(txt))==1:
       if txt[-5:] in trends['suffixes']['drug_n'] : return "drug_n"
       elif txt[-5:] in trends['suffixes']['group'] : return "group"
       # elif txt[-5:] in trends['suffixes']['drug'] : return "drug"
       # elif txt[-5:] in trends['suffixes']['brand'] : return "brand"
       # elif num_re.findall(txt): return "drug_n"
       # elif txt.isupper() : return "brand"
       # elif txt[:3] in trends['prefixes']['drug'] : return "drug"
       # elif txt[:3] in trends['prefixes']['drug_n'] : return "drug_n"
       elif txt[:3] in trends['prefixes']['brand'] : return "brand"
       # elif txt[:3] in trends['prefixes']['drug'] : return "group"
   return "NONE"
   

## --------- Entity extractor ----------- 
## -- Extract drug entities from given text and return them as
## -- a list of dictionaries with keys "offset", "text", and "type"

def extract_entities(stext) :
    
    # tokenize text
    tokens = tokenize(stext)
             
    result = []
    

    # how many tokens should be checked starting from actual one
    # e.g. windows_max_size = 5, check token 0, then token 0+1, then token 0+1+2 ... token 0+1+2+3+4
    window_max_size = 5
    i = 0
    # for each token
    while i < len(tokens):
        drug_type = "NONE"
        for j in range(1, 1 + window_max_size):
            if i+j <len(tokens):
                new_token_start = tokens[i:i+j][0][1]  # get start of first token
                new_token_end = tokens[i:i+j][-1][2]  # get end of last token
                multiple_token_txt = stext[new_token_start:new_token_end+1]  # glue tokens

                new_drug_type = classify_token(multiple_token_txt) # classify token(s)

                if new_drug_type != "NONE": # save only the one with the most tokens
                    k = i+j + 1
                    token_start = new_token_start
                    token_end = new_token_end
                    drug_type = new_drug_type  

        if drug_type != "NONE":
            e = { "offset" : str(token_start)+"-"+str(token_end),
                  "text" : stext[token_start:token_end+1],
                  "type" : drug_type
                 }
            result.append(e)
            i = k
        else:
            i += 1
                    
    return result
      
## --------- main function ----------- 

def nerc(datadir, outfile) :
   
    # open file to write results
    outf = open(outfile, 'w')

    # process each file in input directory
    for f in listdir(datadir) :
      
        # parse XML file, obtaining a DOM tree
        tree = parse(datadir+"/"+f)
      
        # process each sentence in the file
        sentences = tree.getElementsByTagName("sentence")
        for s in sentences :
            sid = s.attributes["id"].value   # get sentence id
            stext = s.attributes["text"].value   # get sentence text
            
            # extract entities in text
            entities = extract_entities(stext)
         
            # print sentence entities in format requested for evaluation
            for e in entities :
                print(sid,
                      e["offset"],
                      e["text"],
                      e["type"],
                      sep = "|",
                      file=outf)
            
    outf.close()


   
## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  baseline-NER.py target-dir
## --
## -- Extracts Drug NE from all XML files in target-dir
## --

try:
    use_neptune = sys.argv[3]
except:
    use_neptune = None

if use_neptune:
    config = dotenv_values("../.env")

    run = neptune.init_run(
        project="projects.mai.bcn/AHLT",
        api_token=config['NPT_MAI_PB'],
        tags=['NERC', 'baseline-NER']
    )  # your credentials

# directory with files to process
# datadir = sys.argv[1]
# outfile = sys.argv[2]
p = {"task":"NER", "datadir": sys.argv[1], "outfile": sys.argv[2]}
nerc(p["datadir"],p["outfile"])

if use_neptune:
    run["parameters"] = p
    run["results/results"].upload(p["outfile"])
    evaluator.evaluate(p["task"], '/'.join(p["datadir"].split('/')[:-1]+['']), p["outfile"], run if use_neptune else None)
    run.stop()
else:
    evaluator.evaluate(p["task"], p["datadir"], p["outfile"], run if use_neptune else None)

