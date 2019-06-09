#!/usr/bin/python3
# -*- coding: utf-8 -*-

import random
import sys

from nltk.corpus import wordnet

# 1. Get WordNet nouns, excluding metadata
# 2. Replace underscore with space
# 3. Combine words into set
# 4. Convert set into tuple for random.choice
NOUNS = {x.name().split('.', 1)[0].replace('_', ' ') for x in wordnet.all_synsets('n')}

WORD_DELIM = ' '
SUFFIX = '|search'

EXAMPLES = """
could you find news for _
could you locate news about _
could you search about _
could you search for _
i want news about _
i want news discussing _
i want news talking about _
i want news with _
i would like to find news about _
i would like to hear about _
i would like to search for _
locate news about _
find news about _
search news for _
tell me about _
lookup _
search for _
news about _
news containing _
news regarding _
news related to _
please find articles about _
please locate news about _
please search about _
please search for articles about _
please search for _
please search news for _
please tell me about _
"""

def _generate_words():
    return WORD_DELIM.join(random.sample(NOUNS, random.randint(1, 3)))

def _generate_anded_words():
    yield _generate_words()
    for _ in range(random.randint(0, 3)):
        yield 'and'
        yield _generate_words()

def _generate_with_cluding():
    cludes = ['including', 'excluding']
    yield from _generate_anded_words()
    if random.random() > 0.75:
        yield cludes.pop(random.randint(0, 1))
        yield from _generate_anded_words()
        if random.random() > 0.6:
            yield cludes[0]
            yield from _generate_anded_words()

def _generate_search_keyphrase():
    for template in filter(len, EXAMPLES.splitlines()):
        if '_' not in template:
            raise ValueError(f'No _ in {template}')
        template = template.split('_')
        result = ''
        for i in range(len(template)-1):
            result += template[i] + WORD_DELIM.join(_generate_with_cluding())
        result += template[-1] + SUFFIX
        yield result

def main(epochs):
    for _ in range(epochs):
        for example in _generate_search_keyphrase():
            print(example)

if __name__ == '__main__':
    main(int(sys.argv[1]))
