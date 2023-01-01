# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from transformers import MT5Tokenizer, MT5Config, MT5ForConditionalGeneration
import numpy as np
from lama.modules.base_connector import *
import torch.nn.functional as F
class MT5(Base_Connector):

    def __init__(self, args, vocab_subset = None):
        super().__init__()

        mt5_model_name = args.mt5_model_name
        dict_file = mt5_model_name

        # When using a cased model, make sure to pass do_lower_case=False directly to BaseTokenizer
        do_lower_case = False
        if 'uncased' in mt5_model_name:
            do_lower_case=True

        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # My envirnment uses CPU

        self.tokenizer = MT5Tokenizer.from_pretrained(mt5_model_name)
        self.config = MT5Config.from_pretrained(mt5_model_name)
        self.mlm = MT5ForConditionalGeneration.from_pretrained(mt5_model_name, config=self.config).to(self.DEVICE)

    def _filter(self, output, end_token='<extra_id_1>'):
        # The first token is <unk> (inidex at 0) and the second token is <extra_id_0> (indexed at 32099)
        _txt = self.tokenizer.decode(output[2:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if end_token in _txt:
            _end_token_index = _txt.index(end_token)
            return _txt[:_end_token_index]
        else:
            return _txt

    def get_id(self, string):
        # tokenized_text = self.tokenizer.tokenize(string)
        # indexed_string = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        # if self.map_indices is not None:
        #     # map indices to subset of the vocabulary
        #     indexed_string = self.convert_ids(indexed_string)

        encoded = self.tokenizer.encode_plus(string, add_special_tokens=True, return_tensors='pt')
        indexed_string = encoded['input_ids'].to(self.DEVICE)

        return indexed_string

    def generate_output(self, input_ids):
        # Generaing 20 sequences with maximum length set to 5
        outputs = self.mlm.generate(input_ids=input_ids, 
                                num_beams=200, num_return_sequences=10,
                                max_length=3)
        return outputs

