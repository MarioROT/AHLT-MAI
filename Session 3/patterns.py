## ------------------- 
## -- check pattern:  LCS is a verb, one entity is under its "nsubj" and the other under its "obj"      

def check_LCS_svo(tree,tkE1,tkE2):
   ##### K: Added few words based on the observations, up devel ~11% 

   if tkE1 is not None and tkE2 is not None:
      lcs = tree.get_LCS(tkE1,tkE2)

      if tree.get_tag(lcs)[0:2] == "VB" :      
         path1 = tree.get_up_path(tkE1,lcs)
         path2 = tree.get_up_path(tkE2,lcs)
         func1 = tree.get_rel(path1[-1]) if path1 else None
         func2 = tree.get_rel(path2[-1]) if path2 else None
         
         if (func1=='nsubj' and func2=='obj') or (func1=='obj' and func2=='nsubj') :
            lemma = tree.get_lemma(lcs).lower()
            if lemma in ['diminish','augment','exhibit','experience','counteract','potentiate',
                         'enhance','reduce','antagonize', 'include', 'block', 'elicit', 'reveal'] :
               return 'effect'
            if lemma in ['impair','inhibit','displace','accelerate','bind','induce','decrease',
                         'elevate','delay', 'cause', 'depress'] :
               return 'mechanism'
            if lemma in ['exceed'] :
               return 'advise'
            if lemma in ['suggest', 'identify', 'interact'] :
               return 'int'
         
   return None

def check_lcs_verb_with_should(tree,tkE1,tkE2):
    """
    checking if the lowest common subsumer of two drug entities in a dependency tree 
    is a verb associated with modal verbs like "should"
    """    
    if tkE1 is not None and tkE2 is not None:
        # Find the lowest common subsumer (LCS) between the two entities
        lcs = tree.get_LCS(tkE1, tkE2)
        
        if lcs is not None:
            # Check if the LCS is a verb
            if tree.get_tag(lcs).startswith("VB"):
                # Check if any child has the lemma "should"
                children = tree.get_children(lcs)
                for child in children:
                    if tree.get_lemma(child).lower() == "should":
                        return 'advise'  # Return 'advise' if LCS has a "should" child
                    if tree.get_lemma(child).lower() in ["must" , "ought"]:
                        # , "have" , "got" , "need" , "obliged" , "required"
                        return 'advise'  # Return 'advise' if LCS has a "should" child
                    
    
    return None

## ------------------- 
## -- check pattern:  A word in between both entities belongs to certain list

def check_wib(tree,tkE1,tkE2,entities,e1,e2):

   if tkE1 is not None and tkE2 is not None:
      # get actual start/end of both entities
      l1,r1 = entities[e1]['start'],entities[e1]['end']
      l2,r2 = entities[e2]['start'],entities[e2]['end']
      
      p = []
      for t in range(tkE1+1,tkE2) :
         # get token span
         l,r = tree.get_offset_span(t)
         # if the token is in between both entities
         if r1 < l and r < l2:
            lemma = tree.get_lemma(t).lower()
            if lemma in ['tendency','stimulate','regulate','prostate','modification',
                         'augment','accentuate','exacerbate', 'counteract', 'potassium', 
                         'augment', 'death', 'narcotic', 'natriuretic', 'norepinephrine',
                         'hypotensive', 'bleeding', 'effectiveness', 'antihypertensive'] :
               return 'effect'
            if lemma in ['react','faster','presumably','induction','substantially','minimally',
                         'elevate', 'dexamethasone', 'competitive', 'prostaglandin', 'stimulator',
                         'triazolam', 'modify', 'compound', 'through', 'microsomal',
                         'apparently'] :
               return 'mechanism'
            if lemma in ['exceed','extreme','cautiously', 'motility', 'withdraw', 'least']:
               return 'advise'
            if lemma in ['interact', 'gastrointestinal'] :
               return 'int'

   return None

def check_verbs_after_and(tree, tkE1, tkE2):
   if tkE1 is not None and tkE2 is not None:
      # Ensure entities are in the correct order (e1 before e2)
      if tkE1 > tkE2:
         tkE1, tkE2 = tkE2, tkE1
      
      # Find the position of the conjunction "and"
      and_position = None
      for t in range(tkE1 + 1, tkE2):
         if tree.get_lemma(t).lower() == 'and':
            and_position = t
            break
      
      if and_position:
         # Check for verbs after the "and" conjunction
         for t in range(and_position + 1, tree.get_n_nodes()):
            if tree.get_tag(t).startswith("VB"):  # Verbs
               lemma = tree.get_lemma(t).lower()
               # Check for interaction types based on specific verb lists
               # if lemma in ['stimulate', 'potentiate', 'enhance', 'augment']:
               #    return 'effect'
               if lemma in ['modify', 'change', 'influence', 'impair','inhibit','displace','accelerate','bind','induce','decrease',
                         'elevate','delay', 'cause', 'depress']:
                  return 'mechanism'
               # elif lemma in ['warn', 'caution']:
               #    return 'advise'
               # elif lemma in ['interact', 'cooperate']:
               #    return 'int'
      
   return None

def check_LCS_is_monitor(tree, tkE1, tkE2):

   if tkE1 is not None and tkE2 is not None:
      lcs = tree.get_LCS(tkE1, tkE2)

      if tree.get_tag(lcs)[0:2] == "VB" and tree.get_lemma(lcs) in ['monitor']:
         return 'advise'

   return None

def check_LCS_obj(tree, tkE1, tkE2):
   
   if tkE1 is not None and tkE2 is not None:
      lcs = tree.get_LCS(tkE1, tkE2)
      
      for c in tree.get_children(lcs):
         if tree.get_rel(c) == 'obj':
            k = tree.get_lemma(lcs).lower()+'_'+tree.get_lemma(c).lower()
            
            if k in ['increase_response', 'diminish_response', 'regulate_proliferation',
                     'prolong_time', 'increase_irritability', 'contain_epinephrine',
                     'cause_arrhythmia', 'have_consequence', 'exacerbate_hypertension',
                     'evaluate_possibility', 'produce_effect', 'increase_risk',
                     'experience_reduction', 'modify_effect', 'take_capecitabine',
                     'take_anticoagulant']:
               return 'effect'
            
            if k in ['form_complex', 'increase_level', 'increase_clearance', 
                     'affect_concentration', 'increase_area', 'have_which']:
               return 'mechanism'
            
            if k in ['tell_doctor', 'exceed_dose', 'cause_hypermagnesemia']:
               return 'advise'
               
   return None

