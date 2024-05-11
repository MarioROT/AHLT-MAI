#! /usr/bin/python3

import sys, os
sys.path.append('../')
from xml.dom.minidom import parse

# folder where this file is located
THISDIR=os.path.abspath(os.path.dirname(__file__))
# go two folders up and locate "util" folder there
LABDIR=os.path.dirname(os.path.dirname(THISDIR))
UTILDIR=os.path.join(LABDIR, "util")
# add "util" to search path so python can find "deptree"
sys.path.append(UTILDIR)

from util.deptree import *
import patterns


## ------------------- 
## -- Convert a pair of drugs and their context in a feature vector


## ORIGINAL
# def extract_features(tree, entities, e1, e2) :
#    feats = set()

#    # get head token for each gold entity
#    tkE1 = tree.get_fragment_head(entities[e1]['start'],entities[e1]['end'])
#    tkE2 = tree.get_fragment_head(entities[e2]['start'],entities[e2]['end'])

#    if tkE1 is not None and tkE2 is not None:

#       # features for tokens in between E1 and E2
#       for tk in range(tkE1+1, tkE2) :
#          if not tree.is_stopword(tk):
#             word  = tree.get_word(tk)
#             lemma = tree.get_lemma(tk).lower()
#             tag = tree.get_tag(tk)
#             feats.add("lib=" + lemma)
#             feats.add("wib=" + word)
#             feats.add("lpib=" + lemma + "_" + tag)
            
#             # feature indicating the presence of an entity in between E1 and E2
#             if tree.is_entity(tk, entities) :
#                feats.add("eib")

#       # features about paths in the tree
#       lcs = tree.get_LCS(tkE1,tkE2)
      
#       path1 = tree.get_up_path(tkE1,lcs)
#       path1 = "<".join([tree.get_lemma(x)+"_"+tree.get_rel(x) for x in path1])
#       feats.add("path1="+path1)

#       path2 = tree.get_down_path(lcs,tkE2)
#       path2 = ">".join([tree.get_lemma(x)+"_"+tree.get_rel(x) for x in path2])
#       feats.add("path2="+path2)

#       path = path1+"<"+tree.get_lemma(lcs)+"_"+tree.get_rel(lcs)+">"+path2      
#       feats.add("path="+path)

#       lemma = patterns.check_LCS_svo(tree,tkE1,tkE2)
#       if lemma is not None:
#          feats.add("LCS_svo="+lemma)
      
#    return feats



verb_list = ['diminish','augment','exhibit','experience','counteract','potentiate',
             'enhance','reduce','antagonize', 'include', 'block', 'elicit', 'reveal'] + \
            ['impair','inhibit','displace','accelerate','bind','induce','decrease',
             'elevate','delay', 'cause', 'depress'] + \
            ['exceed'] + \
            ['suggest', 'identify', 'interact'] 


def extract_features(tree, entities, e1, e2, verb_list):
   feats = set()

   # get head token for each gold entity
   tkE1 = tree.get_fragment_head(entities[e1]['start'], entities[e1]['end'])
   tkE2 = tree.get_fragment_head(entities[e2]['start'], entities[e2]['end'])

   if tkE1 is not None and tkE2 is not None:
      # Features for tokens before E1
      for tk in range(1, tkE1):  # Starts from 1 to skip the root
         if not tree.is_stopword(tk):
            lemma = tree.get_lemma(tk).lower()
            feats.add("lbe1=" + lemma)
            if lemma in verb_list:
               feats.add("verb_before_e1=" + lemma)

      # Features for tokens in between E1 and E2
      for tk in range(tkE1 + 1, tkE2):
         if not tree.is_stopword(tk):
            word = tree.get_word(tk)
            lemma = tree.get_lemma(tk).lower()
            tag = tree.get_tag(tk)
            feats.add("lib=" + lemma)
            feats.add("wib=" + word)
            feats.add("lpib=" + lemma + "_" + tag)
            if tree.is_entity(tk, entities):
               feats.add("eib")
            if lemma in verb_list:
               feats.add("verb_between=" + lemma)

      # Features for tokens after E2
      for tk in range(tkE2 + 1, tree.get_n_nodes()):
         if not tree.is_stopword(tk):
            lemma = tree.get_lemma(tk).lower()
            feats.add("lae2=" + lemma)
            if lemma in verb_list:
               feats.add("verb_after_e2=" + lemma)

      # LCS and Path features
      lcs = tree.get_LCS(tkE1, tkE2)
      if lcs is not None:
         lcs_lemma = tree.get_lemma(lcs).lower()
         lcs_tag = tree.get_tag(lcs) 
         feats.add("lcs_tag=" + lcs_tag)
         feats.add("lcs_lemma=" + lcs_lemma)

      path1 = tree.get_up_path(tkE1, lcs)
      path1 = "<".join([tree.get_lemma(x) + "_" + tree.get_tag(x) for x in path1])
      feats.add("path1=" + path1)

      path2 = tree.get_down_path(lcs, tkE2)
      path2 = ">".join([tree.get_lemma(x) + "_" + tree.get_tag(x) for x in path2])
      feats.add("path2=" + path2)

      full_path = path1 + "<" + lcs_lemma + "_" + tree.get_rel(lcs) + ">" + path2  
      feats.add("full_path=" + full_path)

      lemma = patterns.check_LCS_svo(tree,tkE1,tkE2)
      if lemma is not None:
         feats.add("LCS_svo="+lemma)

      lemma = patterns.check_wib(tree,tkE1,tkE2, entities, e1, e2)
      if lemma is not None:
         feats.add("wib="+lemma)

      ## no improvement
      # lemma = patterns.check_lcs_verb_with_should(tree,tkE1,tkE2)
      # if lemma is not None:
      #    feats.add("verb_like_should="+lemma)

      ## no improvement
      # lemma = patterns.check_verbs_after_and(tree,tkE1,tkE2)
      # if lemma is not None:
      #    feats.add("verb_after_and="+lemma)

   return feats


## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  extract_features targetdir
## --
## -- Extracts feature vectors for DD interaction pairs from all XML files in target-dir
## --

# directory with files to process
datadir = sys.argv[1]

# process each file in directory
for f in os.listdir(datadir) :

    # parse XML file, obtaining a DOM tree
    tree = parse(datadir+"/"+f)

    # process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences :
        sid = s.attributes["id"].value   # get sentence id
        stext = s.attributes["text"].value   # get sentence text
        # load sentence entities
        entities = {}
        ents = s.getElementsByTagName("entity")
        for e in ents :
           id = e.attributes["id"].value
           offs = e.attributes["charOffset"].value.split("-")           
           entities[id] = {'start': int(offs[0]), 'end': int(offs[-1])}

        # there are no entity pairs, skip sentence
        if len(entities) <= 1 : continue

        # analyze sentence
        analysis = deptree(stext)

        # for each pair in the sentence, decide whether it is DDI and its type
        pairs = s.getElementsByTagName("pair")
        for p in pairs:
            # ground truth
            ddi = p.attributes["ddi"].value
            if (ddi=="true") : dditype = p.attributes["type"].value
            else : dditype = "null"
            # target entities
            id_e1 = p.attributes["e1"].value
            id_e2 = p.attributes["e2"].value
            # feature extraction

            # feats = extract_features(analysis,entities,id_e1,id_e2) 
            feats = extract_features(analysis,entities,id_e1,id_e2, verb_list) 
            # resulting vector
            print(sid, id_e1, id_e2, dditype, "\t".join(feats), sep="\t")

