import time
import numpy as np
from SP_func import *
start = time.process_time()

#ALGORITHM INITIALIZATION
print("Start")
#reading the error corpus
#the data is structured as follows: one word per line. The correct word marked with $ sign, next lines without this sign are word's misspellings
# while corpus reading words and misspellings containing "." and "_" are filtered out; all items are made lower case
print("Reading the error corpus")
birkbeck_errors = {}
with open("missp.dat", "r") as file:
    correct_w = ""
    misspells = []
    for line in file.readlines():
        if line.startswith("$"):
            if misspells and "_" not in correct_w.rstrip() and "." not in correct_w.rstrip():
                #adding word boundary signs to each word
                birkbeck_errors["`" + correct_w.rstrip() + "`"] = ["`" + error.rstrip() + "`" for error in misspells]
            misspells = []
            correct_w = line[1:].lower()
        else:
            if "_" not in line and "." not in line: misspells.append(line.lower())

print("Reading the language corpus")
#reading the language corpus
big_txt = sentence_reader("big.txt", '\'?[a-zA-Z]+[-_]?[a-zA-Z\']+\'?|\'?[a-zA-Z]+[a-zA-Z\']*\'?')

#word counter
NWORDS = Counter(word for word in big_txt)
del NWORDS["<#>"] #we don't need the end of sentence sign, as it is used only for creating n-grams from big_txt variable

print("Creating a confusion matrix")
#creating a counter of all characters in the language corpus, exluding <#> (end of sentence sign)
char_counter = Counter(char for word in big_txt for char in word if char not in "#<>")

#creating an alphabet based on what characters are in the corpus
all_chars = [char for char in char_counter.keys() if char != "`"]
#sorting the alphabet and adding the "`" (beginning/end of the word sign) at the 0 position (for convenience)
all_chars.sort()
alphabet ="`" +''.join(all_chars)
#the resulting alphabet was then manually added to the alpha() function in the SP_func

#retrieving all possible character pairs
char_pairs = []
for char1 in char_counter:
    for char2 in char_counter:
        char_pair = char1 + char2
        char_pairs.append(char_pair)

#creating a counter of character pairs in the corpus
char_pairs_counter = Counter(pair for word in big_txt for pair in char_pairs if pair in word)

#creating a dictionary with empty confusion matrices for each error type
confusion = {key:np.zeros((len(alphabet), len(alphabet))) for key in ["del", "ins", "sub", "trans"]}

#filling in confusion matrices with error counts
fill_conf_mtrx(birkbeck_errors, confusion)

#the correction function that returns the string with the best correction for the particular misspelling
def SP_correct(typo, model=NWORDS, pre_ctx = (), post_ctx = (), allow_missp = True):
    LH = p_typo_char("`" + typo + "`", confusion, NWORDS, char_pairs_counter, char_counter, allow_missp)
    return best_correction(cand_posterior(LH, model, pre_ctx, post_ctx), NWORDS)[0]

print("Algorithm initialized", time.process_time()-start)

#EVALUATION

#test function for the unigram model
def spelltest_unigram(tests, verbose=False, allow_missp = True):

    # if verbose, prints failed corrections, their counts in the corpus and counts of the expected corrections
    # if allow_missp, the algorithm uses misspellings that exist in the corpus as correction candidates themselves

    n, bad, start = 0, 0, time.process_time()
    # n - number of processed misspellings
    # bad - corrections that did not match the target word

    for target,wrongs in tests.items():
        # target - correct word, wrongs - its misspellings
        if "`"+target+"`" in NWORDS:
            for wrong in wrongs.split():
                n += 1
                w = SP_correct(wrong, allow_missp=allow_missp)
                if w[1:-1]!=target:
                    bad += 1
                    if verbose:
                        print ('correct(%r) => %r (%d); expected %r (%d)' % (
                            wrong, w[1:-1], NWORDS[w] if w in NWORDS else 0, target,
                            NWORDS["`"+target+"`"] if "`"+target+"`" in NWORDS else 0))

    return dict(bad=bad, n=n, pct=int(100. - 100.*bad/n), secs=int(time.process_time()-start))

#test function for the n-gram models (n>1)
def spelltest_ngram(tests, model, ctx=(), post_ctx = False, verbose=False, allow_missp = True):

    # model - the tuple with two counters: n-gram count and (n-1)-gram count in the corpus
    # ctx - the context in which misspelling is passed
    # post_ctx - if True, then the n-gram model tested with the pre-text and post-text both of a n-1 size
    # verbose - if True prints failed corrections, their counts in the corpus and counts of the expected corrections
    # allow_missp - if True the algorithm uses misspellings that exist in the corpus as correction candidates themselves

    n, bad, start = 0, 0, time.process_time()

    for target,wrongs in tests.items():
        for wrong in wrongs.split():
            if target in ctx:
                n += 1
                w = SP_correct(wrong, model, ctx[target][0], ctx[target][1], allow_missp=allow_missp) if post_ctx else SP_correct(wrong, model, ctx[target], allow_missp=allow_missp)
                if w[1:-1]!=target:
                    bad += 1
                    if verbose:
                        print('correct(%r) => %r (%d); expected %r (%d)' % (
                            wrong, w[1:-1], NWORDS[w] if w in NWORDS else 0, target,
                            NWORDS["`"+target+"`"] if "`"+target+"`" in NWORDS else 0))
    return dict(bad=bad, n=n, pct=int(100. - 100.*bad/n), secs=int(time.process_time()-start))

from testsets import *

#Model creation

NGRAM = 2
LM_2gram = ngram_count(big_txt, NGRAM)
print("Creating the context for the bigram model")
print("...for the test 1")
n2_ctx_test1 = ctx_dict(tests1(), big_txt, NWORDS, NGRAM)
print("...for the test 2")
n2_ctx_test2 = ctx_dict(tests2(), big_txt, NWORDS, NGRAM)
print("Done")

print("Creating the context for the bigram model with the post-text")
print("...for the test 1")
n2p_ctx_test1 = ctx_dict(tests1(), big_txt, NWORDS, NGRAM, True)
print("...for the test 2")
n2p_ctx_test2 = ctx_dict(tests2(), big_txt, NWORDS, NGRAM, True)
print("Done")

NGRAM = 3
LM_3gram = ngram_count(big_txt, NGRAM)
print("Creating the context for the trigram model")
print("...for the test 1")
n3_ctx_test1 = ctx_dict(tests1(), big_txt, NWORDS, NGRAM)
print("...for the test 2")
n3_ctx_test2 = ctx_dict(tests2(), big_txt, NWORDS, NGRAM)
print("Done")

#the function that creates the test output
def tests(missp_only = False, verbose_t1 = True, verbose_t2 = False):

    # missp_only - if False, both configurations (when a word-misspelling can be a correction candidate and when it cannot) are tested;
    #               if True - only the first configuration is tested
    #verbose_t1/t2 - if true, prints failed corrections for the test 1/test 2

    print("UNIGRAM TEST")
    print("Test 1")
    print(spelltest_unigram(tests1(), verbose=verbose_t1))
    print("Test 2")
    print(spelltest_unigram(tests2(), verbose=verbose_t2))

    if not missp_only:
        print("Misspelling as a correction is not allowed")
        print("Test 1")
        print(spelltest_unigram(tests1(), allow_missp=missp_only))
        print("Test 2")
        print(spelltest_unigram(tests2(), allow_missp=missp_only))

    print("BIGRAM TEST")
    print("Test 1")
    print(spelltest_ngram(tests1(), LM_2gram, n2_ctx_test1, verbose=verbose_t1))
    print("Test 2")
    print(spelltest_ngram(tests2(), LM_2gram, n2_ctx_test2, verbose=verbose_t2))

    if not missp_only:
        print("Misspelling as a correction is not allowed")
        print("Test 1")
        print(spelltest_ngram(tests1(), LM_2gram, n2_ctx_test1, allow_missp=missp_only))
        print("Test 2")
        print(spelltest_ngram(tests2(), LM_2gram, n2_ctx_test2, allow_missp=missp_only))

    print("BIGRAM WITH POST-TEXT TEST")
    print("Test 1")
    print(spelltest_ngram(tests1(), LM_2gram, n2p_ctx_test1, verbose=verbose_t1))
    print("Test 2")
    print(spelltest_ngram(tests2(), LM_2gram, n2p_ctx_test2, verbose=verbose_t2))

    if not missp_only:
        print("Misspelling as a correction is not allowed")
        print("Test 1")
        print(spelltest_ngram(tests1(), LM_2gram, n2p_ctx_test1, allow_missp=missp_only))
        print("Test 2")
        print(spelltest_ngram(tests2(), LM_2gram, n2p_ctx_test2, allow_missp=missp_only))

    print("TRIGRAM TEST")
    print("Test 1")
    print(spelltest_ngram(tests1(), LM_3gram, n3_ctx_test1, verbose=verbose_t1))
    print("Test 2")
    print(spelltest_ngram(tests2(), LM_3gram, n3_ctx_test2, verbose=verbose_t2))

    if not missp_only:
        print("Misspelling as a correction is not allowed")
        print("Test 1")
        print(spelltest_ngram(tests1(), LM_3gram, n3_ctx_test1, allow_missp=missp_only))
        print("Test 2")
        print(spelltest_ngram(tests2(), LM_3gram, n3_ctx_test2, allow_missp=missp_only))

tests() # set missp_only=True if you don't want to test both configurations for each model (reduces the runtime)
print("Done")
print(time.process_time()- start)