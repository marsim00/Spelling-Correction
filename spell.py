"""Spelling Corrector.

Copyright 2007 Peter Norvig. 
Open source code under MIT license: http://www.opensource.org/licenses/mit-license.php
"""

import re, collections

def words(text): return re.findall('[a-z]+', text.lower()) 

def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model

with open('big.txt') as file:
    NWORDS = train(words(file.read()))

alphabet = 'abcdefghijklmnopqrstuvwxyz'

def edits1(word):
   s = [(word[:i], word[i:]) for i in range(len(word) + 1)]
   deletes    = [a + b[1:] for a, b in s if b]
   transposes = [a + b[1] + b[0] + b[2:] for a, b in s if len(b)>1]
   replaces   = [a + c + b[1:] for a, b in s for c in alphabet if b]
   inserts    = [a + c + b     for a, b in s for c in alphabet]
   return set(deletes + transposes + replaces + inserts)

def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

def known(words): return set(w for w in words if w in NWORDS)

def correct(word):
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    return max(candidates, key=NWORDS.get)

################ Testing code from here on ################

def spelltest(tests, verbose=False):
    import time
    n, bad, start = 0, 0, time.process_time()

    for target,wrongs in tests.items():
        if target in NWORDS:
            for wrong in wrongs.split():
                n += 1
                w = correct(wrong)
                if w!=target:
                    bad += 1
                    if verbose:
                        print ('correct(%r) => %r (%d); expected %r (%d)' % (
                            wrong, w, NWORDS[w] if w in NWORDS else 0, target,
                            NWORDS[target] if target in NWORDS else 0))
    return dict(bad=bad, n=n, pct=int(100. - 100.*bad/n), secs=int(time.process_time()-start) )

from testsets import *

print ("**** tests 1 - verbose ****")
print (spelltest(tests1(), verbose=True))
print ("\n**** tests 2 - summary ****")
print (spelltest(tests2()))


    
