from decimal import Decimal
from collections import Counter
from copy import deepcopy

# ALPHABET

def alpha(alphabet="`'-_abcdefghijklmnopqrstuvwxyz"): return alphabet

#returns the index of a character in the alphabet
def l_n(l):
    alphabet = alpha()
    return alphabet.index(l)

# CORPUS READER

def sentence_reader(corpus_path, pattern):
    from nltk.tokenize import sent_tokenize, RegexpTokenizer
    import re

    with open(corpus_path, "r") as corp:
        corpus = corp.read()

    #sentence tokenization
    sents = sent_tokenize(corpus)

    blank = "`" #the sign of the word boundary

    #word tokenizer
    w_tokenizer = RegexpTokenizer(pattern)

    final_sents = []
    for sent in sents:
        #refining the result of the tokenization (splitting the sentence further if it was not tokenized properly)
        if re.search("\n[A-Z]|\t|\n{2,}", sent):
            splits = [split for split in re.split("\n{2,}|\t", sent)]

            for split in splits:
                #tokenization of a sentence into word
                split = w_tokenizer.tokenize(split)
                split = [blank+word.lower()+blank for word in split] #adding word boundary sign before and after each word
                #only sentences that contain more than one word are allowed (e.g. to avoid adding chapter numbers ("VII.") as a whole sentence)
                if len(split)>1:
                    final_sents.append(split+ ["<#>"])
        #tokenization of a sentence into word if a sentence did not need a refinement
        else:
            if len(sent)>1:
                tok_sent = w_tokenizer.tokenize(sent)
                tok_sent = [blank+word.lower()+blank for word in tok_sent]
                final_sents.append(tok_sent + ["<#>"])

    return [word for sent in final_sents for word in sent]

# CONFUSION MATRIX

# defining the error type of a misspelling in the error corpus
def edit1_errors(word, missp, raw_conf_mtrx):

    # word - correct word
    # missp - misspelling
    # raw_conf_mtrx - a dictionary with empty confusion matrices

    # a list with the information about errors
    errors = []

    # lists with characters from the word and the misspelling
    word_chars = [letter for letter in word]
    missp_chars = [letter for letter in missp]

    #their copies
    word_chars_copy = deepcopy(word_chars)
    missp_chars_copy = deepcopy(missp_chars)

    #the logic of an algorithm determining the error type is described in the project report in detail

    #delete
    if len(word) - len(missp) == 1:
        for i in range(len(missp_chars)):
            if not word_chars[i] == missp_chars[i]:
                word_chars_copy.remove(word_chars[i])
                if word_chars_copy == missp_chars:
                    #increments a corresponding cell in the confusion matrix
                    raw_conf_mtrx["del"][l_n(word[i-1])][l_n(word[i])] += 1
                    #adds a misspelling, the error type, its coordinates and the correct word to a list
                    errors = (missp, "del", (l_n(word[i - 1]), l_n(word[i])), word)

    # insertion
    if len(word) - len(missp) == -1:
        for i in range(len(word)):
            if not word_chars[i] == missp_chars[i]:
                missp_chars_copy.remove(missp_chars[i])
                if missp_chars_copy == word_chars:
                    raw_conf_mtrx["ins"][l_n(word[i-1])][l_n(missp[i])] += 1
                    errors = (missp, "ins", (l_n(word[i - 1]), l_n(missp[i])), word)

    #substitution and transposition
    if len(word) == len(missp):
        word_chars = [letter for letter in word]
        missp_chars = [letter for letter in missp]

        match = [word_chars[i] == missp_chars[i] for i in range(len(word))]
        #substitution
        if match.count(False) == 1:
            raw_conf_mtrx["sub"][l_n(word[match.index(False)])][l_n(missp[match.index(False)])] += 1
            errors = (missp, "sub", (l_n(word[match.index(False)]), l_n(missp[match.index(False)])), word)

        #transposition
        if match.count(False) == 2 and sorted(word_chars) == sorted(missp_chars):
            enum_match = list(enumerate(match))
            if [enum_match[i - 1][1] == enum_match[i][1] == False and enum_match[i - 1][0] == enum_match[i][0] - 1 for i in range(len(enum_match))].count(
                    True) == 1:
                raw_conf_mtrx["trans"][l_n(word[match.index(False)])][l_n(missp[match.index(False)])] += 1
                errors = (missp, "trans", (l_n(word[match.index(False)]), l_n(missp[match.index(False)])), word)

    return errors

# filling in the confusion matrix for the whole error corpus
def fill_conf_mtrx(error_dict, conf_mtrx):
    for word in error_dict:
        misspells = error_dict[word]
        used_mp = []

        for missp in error_dict[word]:
            #for each misspelling for the word find the 1-ED error and increment corresponding cells in the confusion matrix are incremented
            edits1 = edit1_errors(word, missp, conf_mtrx)
            if edits1:
                for misspell in misspells:
                    #for each 1-ED misspelling, which was not used for the 2-ED errors determination...
                    if misspell not in used_mp:
                        #...determine 1-ED error from 1-ED misspelling (=2-ED error from the initial word) and increment confusion matrix for this error
                        edits2 = edit1_errors(edits1[0], misspell, conf_mtrx)
                        if edits2:
                            #add 1-ED misspell to the list of used misspells
                            used_mp.append(misspell)
                            #increment the cell for the 1-ED error one more time (to account for 2 errors)
                            conf_mtrx[edits1[1]][edits1[2][0]][edits1[2][1]] += 1

# CORRECTION CANDIDATES

# generation of a 1 edit distance (ED) correction candidates
def generate_cors(typo):
    #locally defining an alphabet as all characters except "`" word boundary sign
    alphabet = alpha()[1:]

    #generation of all possible words at a 1-ED from a given typo (correction candidates)
    #each generated list is a list of tuples containing generated (potentially correct) word, error type (with respect to the typo, not the generation method!), coordinates of the error in the confusion matrix and the typo from which the candidate was generated
    deletes = [("`" + typo[1:ch] + typo[ch + 1:-1] + "`", "ins", (l_n(typo[ch - 1]), l_n(typo[ch])), typo) for ch in range(1, len(typo) - 1)]
    inserts = [("`" + typo[1:ch] + l + typo[ch:-1] + "`", "del", (l_n(typo[ch - 1]), l_n(l)), typo) for ch in range(1, len(typo)) for l in alphabet]
    subs = [("`" + typo[1:ch] + l + typo[ch + 1:-1] + "`", "sub", (l_n(l), l_n(typo[ch - 1])), typo) for ch in range(1, len(typo) - 1) for l in alphabet if l != typo[ch]]
    trans = []
    #transposition candidates are generated only for words of length >1 (aka >3 including word boundary signs)
    if len(typo)>3: trans = [("`" + typo[1:ch + 1] + typo[ch + 2] + typo[ch + 1] + typo[ch + 3:-1] + "`", "trans", (l_n(typo[ch + 2]), l_n(typo[ch + 1])), typo) for ch in range(len(typo) - 3) if typo[ch + 1] != typo[ch + 2]]

    return deletes+inserts+subs+trans

# generation of a 2-ED correction candidates (finding 1-ED words from 1-ED candidates generated in "generate_cors")
# + filtering out 2-ED candidates that are not in the corpus
#                             and that are actually 0-ED from the initial typo (cor1[3])
def generate_cors2(cors1, corpus): return [cor2 for cor1 in cors1 for cor2 in generate_cors(cor1[0]) if cor2[0] in corpus and cor2[0] !=cor1[3]]

# PROBABILISTIC MODEL

# noisy channel for one misspelling
def p_typo_char(missp, conf_mtrx, corpus_counter, corpus_pairs_counter, corpus_chr_counter, allow_missp = True):

    # missp - misspelling
    # conf_mtrx - confusion matrix
    # corpus_counter - a counter of words in the language corpus
    # corpus_pairs_counter - a counter of character pairs in the language corpus
    # corpus_chr_counter - a counter of every single character in the language corpus
    # allow_missp - if True, uses a misspelling itself as a 0-ED correction candidate (=no correction is possible), if the misspelling exists in the corpus

    #a dictionary with P(e|w) for each error type; internal dictionaries with error types will contain corresponding candidates and their P(e|w)
    P_ct_dict = {"del": {}, "ins": {}, "sub": {}, "trans": {}}
    #a separate dictionary with P(e|w) of 1-ED candidates
    P_ct_edit1 = {}

    #lists for 1-ED and 2-ED correction candidates
    corr_list1 = generate_cors(missp)
    corr_list2 = generate_cors2(corr_list1, corpus_counter)

    #a dictionary with correction candidates (1 for 1-ED, 2 for 2-ED)
    corrections = {1: corr_list1, 2: corr_list2}

    #checking if we can use the misspelling as a correction, and if so, adding it to the P(e|w) dictionary with a zero value
    if allow_missp:
        corrections[0] = missp if missp in corpus_counter else None
        if corrections[0]:
            P_ct_dict["missp"] = {}
            P_ct_dict["missp"][(missp,missp)] = 0

    #iterate over possible ED's (1 and 2)
    for edit in range(1,3):
        if corrections[edit]:
            # each corr is a tuple (explained in "generate_cors" function)
            for corr in corrections[edit]:
                cand_corr = corr[0] #correction candidate (string)
                src_typo = corr[3] #a typo the correction candidate was generated from

                #a pair of characters that constitute the error
                char1 = alpha()[corr[2][0]]
                char2 = alpha()[corr[2][1]]
                #their coordinates in the confusion matrix
                row = corr[2][0]
                col = corr[2][1]

                err_type = corr[1] #error type

                # the denominator (count of character pairs or single characters) depends on the error type (see project documentation)
                if err_type == "del" or "trans":
                    P_ct_dict[err_type][(missp,)+corr] = -Decimal((conf_mtrx[err_type][row][col]+1)/(corpus_pairs_counter[char1+char2]+len(corpus_pairs_counter))).ln()

                    if edit==1: P_ct_edit1[cand_corr] = P_ct_dict[err_type][(missp,)+corr] #add the calculated value to the dictionary with 1-ED P(e|w)
                    if edit==2: P_ct_dict[err_type][(missp,)+corr] += P_ct_edit1[src_typo] #sum the P(e|w) of 2-ED candidate with that of 1-ED candidate the 2-ED was generated from (to account for both errors)
                else:
                    #same for substitution and insertion
                    P_ct_dict[err_type][(missp,)+corr] = -Decimal((conf_mtrx[err_type][row][col]+1)/(corpus_chr_counter[char1]+len(corpus_chr_counter))).ln()
                    if edit==1: P_ct_edit1[cand_corr] = P_ct_dict[err_type][(missp,)+corr]
                    if edit==2: P_ct_dict[err_type][(missp,)+corr] += P_ct_edit1[src_typo]

    return P_ct_dict

# N-GRAM MODEL

# the function that counts all n-grams and (n-1)-grams in a language corpus
# used for n>1
def ngram_count(corpus, ngram:int):

    # corpus - language corpus
    # ngram - n-gram size

    from itertools import islice
    # creating a deepcopy of a corpus in order to not change the corpus itself
    corpus = deepcopy(corpus)

    #inserting n-1 sentence end signs, so that the first and last word have the pre-text and post-text of the appropriate size
    for i in range(ngram-1):
        corpus.insert(0, "<#>")
        corpus.append('<#>')

    #n-gram counter
    ngram_counter = Counter(zip(*[islice(corpus,j,None) for j in range(ngram)]))
    #n-1-gram counter
    ctx_counter = Counter(zip(*[islice(corpus,j,None) for j in range(ngram-1)]))

    return ngram_counter, ctx_counter

# prior/language model calculation for one correction candidate
def ngram_model(corr_cand, ngram_counter, pre_ctx = (), post_ctx = ()):

    # corr_cand - correction candidate
    # ngram_counter - either the tuple created in "ngram_count" (if n>1) or the counter of words in the corpus for the unigram model
    # pre_ctx and post_ctx - tuples of words preceding/succeeding the word (both are left empty for unigram models)

    #-ln(P(w)) calculation (for the unigram model) if a corpus word counter is passed
    if type(ngram_counter)!=tuple:
        denominator = sum(ngram_counter.values())+len(ngram_counter)
        ngram_prob = -Decimal((ngram_counter[corr_cand] + 1) / denominator).ln()
        return ngram_prob

    #creating variables for storing logprobs of a word with a pre-text and/or post-text
    ngram_pre_prob = 0
    ngram_post_prob = 0

    counter_ngram = ngram_counter[0] #n-gram counter
    counter_ctx = ngram_counter[1] #(n-1)-gram counter

    # logprob calculation for the word with pre-text (with Laplace smoothing)
    if pre_ctx:
        denominator = counter_ctx[pre_ctx] + len(counter_ctx) #pre-text counts
        pre_ngram = tuple(word for word in pre_ctx) + (corr_cand,) #pre-text + word counts (enumerator)
        ngram_pre_prob = -Decimal((counter_ngram[pre_ngram] + 1) / denominator).ln()

    # logprob calculation for the word with post-text (with Laplace smoothing)
    if post_ctx:
        denominator = counter_ctx[post_ctx] + len(counter_ctx) #post-text counts (n-1-gram counts)
        post_ngram = (corr_cand,) + tuple(word for word in post_ctx) #post-text + word counts (enumerator - n-gram counts)
        ngram_post_prob = - Decimal((counter_ngram[post_ngram] + 1) / denominator).ln()

    #returns some of logprobs for pre-text and post-text (if post-text is absent, its logprob is 0 and only pre-text logprob is returned)
    return ngram_post_prob+ngram_pre_prob

# BEST CORRECTION CHOICE

# -ln(P(w|e)) (posterior) calculation for all correction candidates of a one misspelling
def cand_posterior(likelihoods, ngram_counter, pre_ctx = (), post_ctx = ()):

    # likelihoods - a dictionary with noisy channel logprobs (generated in "p_typo_char") with a following structure:
        # likelihoods = {error type : {
        #                               (misspelling, correction, error type, error coordinates, the source misspelling for correction):
        #                                                                                                                                   -ln(P(e|w))}
    # ngram_counter - either the tuple created in "ngram_count" (if n>1) or the counter of words in the corpus for the unigram model
    # pre_ctx and post_ctx - tuples of words preceding/succeeding the word (both are left empty for unigram models)

    cand_logprobs = []

    #for error type in dictionary
    for err_type in likelihoods:
        #for tuple containing the information about the misspelling, correction candidate, etc. (see above)
        for typo_corr in likelihoods[err_type]:
            #add a tuple containing:  ((misspelling, correction), posterior); misspelling is the same in each tuple
            #posterior logprob is a sum of a likelihood and a correction prior (language model calculated with the "ngram_model" function)
            cand_logprobs.append((typo_corr[0:2], likelihoods[err_type][typo_corr]+ngram_model(typo_corr[1], ngram_counter, pre_ctx, post_ctx)))
    #return the list with obtained tuples
    return cand_logprobs

#choose the best correction for one misspelling
def best_correction(posterior, corpus_counter):

    # posterior - list with tuples created in "cand_posterior" function
    # corpus_counter - a counter of words in the language corpus

    #assign a tuple of a misspelling and the 'Infinity' value to the best_solution
    best_solution = (posterior[0][0][0], Decimal("Infinity"))
    #for tuple in a list
    for post_cand in posterior:
        corr = post_cand[0][1] #correction candidate
        post = post_cand[1]  # posterior of a candidate
        #choose the candidate if its posterior logprob is smaller than the one in the best_solution
        if post < best_solution[1] and corr in corpus_counter:
            best_solution = (corr, post)
    #return the correction candidate with the smallest posterior logprob
    return best_solution

# N-GRAM MODEL TESTING

#finding the most common context of an (n-1)-gram size for target words in test items
def test_ngram(target, corpus, ngram:int, post_ctx = False):

    # target - target word (expected correction)
    # corpus - language corpus
    # ngram - n-gram size
    # post_ctx - if True, find both pre-text and post-text of a n-1 size

    # creating a deepcopy of a corpus in order to not change the corpus itself
    corpus = deepcopy(corpus)

    # inserting n-1 sentence end signs, so that the first word has the context of the appropriate size
    for i in range(ngram - 1):
        corpus.insert(0, "<#>")

    #counting n-grams for the target word
    #if post_text counts n-grams containing n-1 words before the word + word + n-1 words after the word
    ctx_counter = Counter(tuple(corpus[i-ngram+1:i+ngram]) for i in range(len(corpus)) if corpus[i] == target) if post_ctx else Counter(tuple(corpus[i-ngram+1:i+1]) for i in range(len(corpus)) if corpus[i] == target)

    #writing the most common pre-text (n-1 preceding words) of a target to a variable (without a target itself)
    pre = ctx_counter.most_common(1)[0][0][:ngram - 1]

    #n-1 next words if post_ctx
    if post_ctx:
        post = ctx_counter.most_common(1)[0][0][ngram:]
        #return a tuple with the most common pre-text and a post-text
        return pre, post
    else:
        #return pre-text only
        return pre

#creating a dictionary with target words and corresponding most common contexts
def ctx_dict(test_dict, corpus, corpus_counter, ngram, post_ctx = False):

    # test_dict: a dictionary with target words as keys and their misspellings as values
    # corpus - language corpus
    # ngram - n-gram size
    # post_ctx - if True, find both pre-text and post-text of a n-1 size

    target_ctx = {} #a dictionary with targets and their most common contexts
    for target in test_dict:
        if "`"+target+"`" in corpus_counter:
            #if a target word in corpus, add it as key and its most common context as value
            target_ctx[target] = test_ngram("`"+target+"`", corpus, ngram, post_ctx)
    return target_ctx


