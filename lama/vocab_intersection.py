# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from lama.modules import build_model_by_name
from tqdm import tqdm
import argparse
import spacy
import lama.modules.base_connector as base


CASED_MODELS = [
  {
    # "BERT BASE CASED"
    "lm": "bert",
    "bert_model_name": "bert-base-cased",
    "bert_model_dir": "pre-trained_language_models/bert/cased_L-12_H-768_A-12/",
    "bert_vocab_name": "vocab.txt"
  },
  {
    # "BERT LARGE CASED"
    "lm" : "bert",
    "bert_model_name": "bert-large-cased",
    "bert_model_dir": "pre-trained_language_models/bert/cased_L-24_H-1024_A-16/",
    "bert_vocab_name": "vocab.txt"
  },
  {
    # "BERT BASE MULTILINGUAL CASED"
    "lm" : "bert",
    "bert_model_name": "bert-base-multilingual-cased",
    "bert_model_dir": "pre-trained_language_models/bert/multi_cased_L-12_H-768_A-12/",
    "bert_vocab_name": "vocab.txt"
  }
]

CASED_COMMON_VOCAB_FILENAME = "pre-trained_language_models/common_vocab_cased.txt"

LOWERCASED_MODELS = [
 {
   # "BERT TINY UNCASED"
   "lm": "bert",
   "bert_model_name": "bert-tiny-uncased",
   "bert_model_dir": "pre-trained_language_models/bert/uncased_L-2_H-128_A-2/",
   "bert_vocab_name": "vocab.txt"
 },
 {
   # "BERT MINI UNCASED"
   "lm": "bert",
   "bert_model_name": "bert-mini-uncased",
   "bert_model_dir": "pre-trained_language_models/bert/uncased_L-4_H-256_A-4/",
   "bert_vocab_name": "vocab.txt"
 },
 {
   # "BERT SMALL UNCASED"
   "lm": "bert",
   "bert_model_name": "bert-small-uncased",
   "bert_model_dir": "pre-trained_language_models/bert/uncased_L-4_H-512_A-8/",
   "bert_vocab_name": "vocab.txt"
 },
 {
   # "BERT MEDIUM UNCASED"
   "lm": "bert",
   "bert_model_name": "bert-medium-uncased",
   "bert_model_dir": "pre-trained_language_models/bert/uncased_L-8_H-512_A-8/",
   "bert_vocab_name": "vocab.txt"
 },
 {
   # "BERT BASE UNCASED"
   "lm": "bert",
   "bert_model_name": "bert-base-uncased",
   "bert_model_dir": "pre-trained_language_models/bert/uncased_L-12_H-768_A-12/",
   "bert_vocab_name": "vocab.txt"
 },
 {
   # "BERT LARGE UNCASED"
   "lm": "bert",
   "bert_model_name": "bert-large-uncased",
   "bert_model_dir": "pre-trained_language_models/bert/uncased_L-24_H-1024_A-16/",
   "bert_vocab_name": "vocab.txt"
 },  
 {
    # "BERT BASE MULTILINGUAL UNCASED"
    "lm" : "bert",
    "bert_model_name": "bert-base-multilingual-uncased",
    "bert_model_dir": "pre-trained_language_models/bert/multilingual_L-12_H-768_A-12/",
    "bert_vocab_name": "vocab.txt"
  }
]

LOWERCASED_COMMON_VOCAB_FILENAME = "pre-trained_language_models/common_vocab_lowercased.txt"


def __vocab_intersection(models, filename):

    vocabularies = []

    for arg_dict in models:

        args = argparse.Namespace(**arg_dict)
        print(args)
        model = build_model_by_name(args.lm, args)

        vocabularies.append(model.vocab)
        print(type(model.vocab))

    if len(vocabularies) > 0:
        common_vocab = set(vocabularies[0])
        for vocab in vocabularies:
            common_vocab = common_vocab.intersection(set(vocab))

        # no special symbols in common_vocab
        for symbol in base.SPECIAL_SYMBOLS:
            if symbol in common_vocab:
                common_vocab.remove(symbol)

        # remove stop words
        from spacy.lang.en.stop_words import STOP_WORDS
        for stop_word in STOP_WORDS:
            if stop_word in common_vocab:
                print(stop_word)
                common_vocab.remove(stop_word)

        common_vocab = list(common_vocab)

        # remove punctuation and symbols
        nlp = spacy.load('en')
        manual_punctuation = ['(', ')', '.', ',']
        new_common_vocab = []
        for i in tqdm(range(len(common_vocab))):
            word = common_vocab[i]
            doc = nlp(word)
            token = doc[0]
            if(len(doc) != 1):
                print(word)
                for idx, tok in enumerate(doc):
                    print("{} - {}".format(idx, tok))
            elif word in manual_punctuation:
                pass
            elif token.pos_ == "PUNCT":
                print("PUNCT: {}".format(word))
            elif token.pos_ == "SYM":
                print("SYM: {}".format(word))
            else:
                new_common_vocab.append(word)
            # print("{} - {}".format(word, token.pos_))
        common_vocab = new_common_vocab

        # store common_vocab on file
        with open(filename, 'w') as f:
            for item in sorted(common_vocab):
                f.write("{}\n".format(item))


def main():
    # cased version
    __vocab_intersection(CASED_MODELS, CASED_COMMON_VOCAB_FILENAME)
    # lowercased version
    __vocab_intersection(LOWERCASED_MODELS, LOWERCASED_COMMON_VOCAB_FILENAME)


if __name__ == '__main__':
    main()
