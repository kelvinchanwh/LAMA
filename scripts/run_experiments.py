# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
from batch_eval_KB_completion import main as run_evaluation
from batch_eval_KB_completion import load_file
from lama.modules import build_model_by_name
import pprint
import statistics
from os import listdir
import os
from os.path import isfile, join
from shutil import copyfile
from collections import defaultdict
import time
import gc
import random

LMs = [
    {
        "lm": "bert",
        "label": "bert_base",
        "models_names": ["bert"],
        "bert_model_name": "bert-base-cased",
        "bert_model_dir": "pre-trained_language_models/bert/cased_L-12_H-768_A-12",
        "lowercase": False,
    },
    {
        "lm": "bert",
        "label": "bert_large",
        "models_names": ["bert"],
        "bert_model_name": "bert-large-cased",
        "bert_model_dir": "pre-trained_language_models/bert/cased_L-24_H-1024_A-16",
        "lowercase": False,
    },
    {
        "lm": "bert",
        "label": "bert_base_multilingual_cased",
        "models_names": ["bert"],
        "bert_model_name": "bert-base-multilingual-cased",
        "bert_model_dir": "pre-trained_language_models/bert/multi_cased_L-12_H-768_A-12",
        "lowercase": False,
    },
    {
        # "BERT TINY UNCASED"
        "lm": "bert",
        "label": "bert-tiny-uncased",
        "models_names": ["bert"],
        "bert_model_name": "bert-tiny-uncased",
        "bert_model_dir": "pre-trained_language_models/bert/uncased_L-2_H-128_A-2/",
        "lowercase": True,
    },
    {
        # "BERT MINI UNCASED"
        "lm": "bert",
        "label": "bert-mini-uncased",
        "models_names": ["bert"],
        "bert_model_name": "bert-mini-uncased",
        "bert_model_dir": "pre-trained_language_models/bert/uncased_L-4_H-256_A-4/",
        "lowercase": True,
    },
    {
        # "BERT SMALL UNCASED"
        "lm": "bert",
        "label": "bert-small-uncased",
        "models_names": ["bert"],
        "bert_model_name": "bert-small-uncased",
        "bert_model_dir": "pre-trained_language_models/bert/uncased_L-4_H-512_A-8/",
        "lowercase": True,
    },
    {
        # "BERT MEDIUM UNCASED"
        "lm": "bert",
        "label": "bert-medium-uncased",
        "models_names": ["bert"],
        "bert_model_name": "bert-medium-uncased",
        "bert_model_dir": "pre-trained_language_models/bert/uncased_L-8_H-512_A-8/",
        "lowercase": True,
    },
    {
        # "BERT BASE UNCASED"
        "lm": "bert",
        "label": "bert-base-uncased",
        "models_names": ["bert"],
        "bert_model_name": "bert-base-uncased",
        "bert_model_dir": "pre-trained_language_models/bert/uncased_L-12_H-768_A-12/",
        "lowercase": True,
    },
    {
        # "BERT LARGE UNCASED"
        "lm": "bert",
        "label": "bert-large-uncased",
        "models_names": ["bert"],
        "bert_model_name": "bert-large-uncased",
        "bert_model_dir": "pre-trained_language_models/bert/uncased_L-24_H-1024_A-16/",
        "lowercase": True,
    },  
    {
        # "BERT BASE MULTILINGUAL UNCASED"
        "lm" : "bert",
        "label": "bert-base-multilingual-uncased",
        "models_names": ["bert"],
        "bert_model_name": "bert-base-multilingual-uncased",
        "bert_model_dir": "pre-trained_language_models/bert/multilingual_L-12_H-768_A-12/",
        "lowercase": True,
    }
]

def splitConceptNet(data_path_pre="data/ConceptNet/"):
    lines_per_file = 7500
    smallfile = None
    numFiles = 1
    with open(data_path_pre + 'test.jsonl') as bigfile:
        lines = bigfile.readlines()
        random.Random(1).shuffle(lines)
        for lineno, line in enumerate(lines):
            if lineno % lines_per_file == 0:
                if smallfile:
                    smallfile.close()
                    numFiles += 1
                small_filename = data_path_pre + 'test_%d.jsonl'%(lineno/lines_per_file)
                smallfile = open(small_filename, "w")
            smallfile.write(line)
        if smallfile:
            smallfile.close()
            numFiles += 1
    return numFiles

def run_experiments(
    relations,
    data_path_pre,
    data_path_post,
    input_param={
        "lm": "bert",
        "label": "bert_large",
        "models_names": ["bert"],
        "bert_model_name": "bert-large-cased",
        "bert_model_dir": "pre-trained_language_models/bert/cased_L-24_H-1024_A-16",
    },
    use_negated_probes=False,
):
    model = None
    pp = pprint.PrettyPrinter(width=41, compact=True)

    all_Precision1 = []
    type_Precision1 = defaultdict(list)
    type_count = defaultdict(list)

    results_file = open("last_results.csv", "w+")

    for relation in relations:
        pp.pprint(relation)
        PARAMETERS = {
            "dataset_filename": "{}{}{}".format(
                data_path_pre, relation["relation"], data_path_post
            ),
            "common_vocab_filename": "pre-trained_language_models/common_vocab_%s.txt"%("lowercased" if input_param["lowercase"] else "cased"),
            "template": "",
            "bert_vocab_name": "vocab.txt",
            "batch_size": 32,
            "logdir": "output",
            "full_logdir": "output/results/{}/{}".format(
                input_param["label"], relation["relation"]
            ),
            "lowercase": input_param["lowercase"],
            "max_sentence_length": 100,
            "threads": -1,
            "interactive": False,
            "use_negated_probes": use_negated_probes,
        }

        if "template" in relation:
            PARAMETERS["template"] = relation["template"]
            if use_negated_probes:
                PARAMETERS["template_negated"] = relation["template_negated"]

        PARAMETERS.update(input_param)
        print(PARAMETERS)

        args = argparse.Namespace(**PARAMETERS)

        # see if file exists
        try:
            data = load_file(args.dataset_filename)
            if "TREx" in args.dataset_filename:
                for sample in data:
                    for evidence in sample["evidences"]:
                        evidence["masked_sentences"] = [evidence["masked_sentence"]]
                # [evidence["masked_sentences"] = [evidence["masked_sentence"]] for evidence in sample["evidences"] for sample in data]
        except Exception as e:
            print("Relation {} excluded.".format(relation["relation"]))
            print("Exception: {}".format(e))
            continue

        if model is None:
            [model_type_name] = args.models_names
            model = build_model_by_name(model_type_name, args)

        Precision1 = run_evaluation(args, shuffle_data=False, model=model)
        print("P@1 : {}".format(Precision1), flush=True)
        all_Precision1.append(Precision1)

        results_file.write(
            "{},{}\n".format(relation["relation"], round(Precision1 * 100, 2))
        )
        results_file.flush()

        if "type" in relation:
            type_Precision1[relation["type"]].append(Precision1)
            data = load_file(PARAMETERS["dataset_filename"])
            type_count[relation["type"]].append(len(data))

        gc.collect()

    mean_p1 = statistics.mean(all_Precision1)
    print("@@@ {} - mean P@1: {}".format(input_param["label"], mean_p1))
    results_file.close()

    for t, l in type_Precision1.items():

        print(
            "@@@ ",
            input_param["label"],
            t,
            statistics.mean(l),
            sum(type_count[t]),
            len(type_count[t]),
            flush=True,
        )
    gc.collect()

    return mean_p1, all_Precision1


def get_TREx_parameters(data_path_pre="data/"):
    relations = load_file("{}relations.jsonl".format(data_path_pre))
    data_path_pre += "TREx/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def get_GoogleRE_parameters():
    relations = [
        {
            "relation": "place_of_birth",
            "template": "[X] was born in [Y] .",
            "template_negated": "[X] was not born in [Y] .",
        },
        {
            "relation": "date_of_birth",
            "template": "[X] (born [Y]).",
            "template_negated": "[X] (not born [Y]).",
        },
        {
            "relation": "place_of_death",
            "template": "[X] died in [Y] .",
            "template_negated": "[X] did not die in [Y] .",
        },
    ]
    data_path_pre = "data/Google_RE/"
    data_path_post = "_test.jsonl"
    return relations, data_path_pre, data_path_post


def get_ConceptNet_parameters(data_path_pre="data/"):
    print("Splitting ConceptNet to save RAM")
    num_files = splitConceptNet()
    relations = [{"relation": "test_%d"%i} for i in range(num_files)]
    data_path_pre += "ConceptNet/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def get_Squad_parameters(data_path_pre="data/"):
    relations = [{"relation": "test"}]
    data_path_pre += "Squad/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def run_all_LMs(parameters):
    for ip in LMs:
        print(ip["label"])
        run_experiments(*parameters, input_param=ip, use_negated_probes=False)
        gc.collect()


if __name__ == "__main__":

    print("1. Google-RE")
    parameters = get_GoogleRE_parameters()
    run_all_LMs(parameters)

    print("2. T-REx")
    parameters = get_TREx_parameters()
    run_all_LMs(parameters)

    print("3. ConceptNet")
    parameters = get_ConceptNet_parameters()
    run_all_LMs(parameters)

    print("4. SQuAD")
    parameters = get_Squad_parameters()
    run_all_LMs(parameters)

