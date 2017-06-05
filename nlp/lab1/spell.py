import re
from collections import Counter
import numpy as np

def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('../input/big.txt').read()))

def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return word[1] / (WORDS[word[0]] + 0.0001)

def correction(word): 
    "Most probable spelling correction for word."
    corrected = min(candidates(word), key=P)
    return corrected[0]

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([(word, 0)]) or known(edits1(word)) or known(edits2(word)) or [(word, 0)]) #

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w[0] in WORDS)

KEYBOARD = [
    "qwertyuiop",
    "asdfghjkl",
    "zxcvbnm"
];

key_place = {}

VOWELS = "aeoiu"

def replace_score(c1, c2):
    ## mistyping a vowel for another vowel is more probable
    if c1 in VOWELS and c2 in VOWELS:
        return 5

    (c1, c2) = sorted([c1, c2])

    ## common mistakes
    if c1 == 'g' and c2 == 'j':
        return 5

    if c1 == 'c' and c2 == 's':
        return 5

    if c1 == 'd' and c2 == 't':
        return 5

    if c1 == 'b' and c2 == 'p':
        return 5

    r1, f1, r2, f2 = -1, -1, -1, -1

    ## calculate the row and column on the keyboard for c1 and c2
    if c1 in key_place:
        r1, f1 = key_place[c1]
    else:
        for i in range(3):
            if c1 in KEYBOARD[i]:
                r1 = i
                f1 = KEYBOARD[i].find(c1)
                break
        key_place[c1] = (r1, f1)

    if c2 in key_place:
        r2, f2 = key_place[c2]
    else:
        for i in range(3):
            if c2 in KEYBOARD[i]:
                r2 = i
                f2 = KEYBOARD[i].find(c2)
                break
        key_place[c2] = (r2, f2)

    if r1 + r2 > 0:
        if r1 == r2 and abs(f1 - f2) == 1:
            return 5
        elif r1 == r2 and abs(f1 - f2) == 2:
            return 30
        if abs(r1 - r2) < 1 and abs(f1 - f2) <= 1:
            return 10

    return 100

def delete_score(a, b):
    ## accidentally typing the same letter twice is more probable
    if len(b) > 1 and a == b[1]:
        return 5
    return 50

def insert_score(a, b):
    ## omitting one of the two consecutive letters is more probable
    if len(b) > 0 and a == b[0]:
        return 0.5
    return 5

def transpose_score(c1, c2):
    if c1 in VOWELS and c2 in VOWELS:
        return 1


    return 10

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = "abcdefghijklmnopqrstuvwxyz'"
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [(L + R[1:], delete_score(R[0], R)) for L, R in splits if R]
    transposes = [(L + R[1] + R[0] + R[2:], transpose_score(R[0], R[1])) for L, R in splits if len(R)>1]
    replaces   = [(L + c + R[1:], 2*replace_score(c, R[0])) for L, R in splits if R for c in letters]
    inserts    = [(L + c + R, insert_score(c, R))  for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return ((e2[0], e2[1] + e1[1]) for e1 in edits1(word) for e2 in edits1(e1[0]))

def spelltest(tests, verbose=False):
    "Run correction(wrong) on all (right, wrong) pairs; report results."
    import time
    start = time.clock()
    good, unknown = 0, 0
    n = len(tests)
    for right, wrong in tests:
        right = right.lower()
        wrong = wrong.lower()
        w = correction(wrong)
        good += (w == right)
        if w != right:
            unknown += (right not in WORDS)
            if verbose:
                print('correction({}) => {} ({}); expected {} ({})'
                      .format(wrong, w, WORDS[w], right, WORDS[right]))
    dt = time.clock() - start
    print('{:.0%} of {} correct ({:.0%} unknown) at {:.0f} words per second '
          .format(good / n, n, unknown / n, n / dt))
    
def Testset(lines):
    "Parse 'right: wrong1 wrong2' lines into [('right', 'wrong1'), ('right', 'wrong2')] pairs."
    return [(right, wrong)
            for (right, wrongs) in (line.split(':') for line in lines)
            for wrong in wrongs.split()]

def test_corpus(filename):
    print("Testing " + filename)
    spelltest(Testset(open('../input/' + filename)))     

test_corpus('spell-testset1.txt') # Development set
test_corpus('spell-testset2.txt') # Final test set

# Supplementary sets
test_corpus('aspell.txt')
test_corpus('wikipedia.txt')

# Long test, for the patient only
# test_corpus('birkbeck.txt') 