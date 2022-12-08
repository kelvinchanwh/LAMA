import torch
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
import pytorch_pretrained_bert.tokenization as btok
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM, BasicTokenizer, BertModel
import numpy as np
from lama.modules.base_connector import *
import torch.nn.functional as F

class T5(Base_Connector):

    def __init__(self, args, vocab_subset = None):
        super().__init__()

        t5_model_name = args.t5_model_name
        dict_file = t5_model_name

        # When using a cased model, make sure to pass do_lower_case=False directly to BaseTokenizer
        do_lower_case = False
        if 'uncased' in t5_model_name:
            do_lower_case=True

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = T5Tokenizer.from_pretrained(dict_file)

        def convert_word(word):
            if word == T5_UNK or word == T5_SEP or word == T5_PAD:
                return word
            return word[:-4] if word.endswith('</s>') else f'{word}##'

        # original vocab
        self.map_indices = None
        t5_vocab = sorted(self.tokenizer.get_vocab())
        self.vocab = [convert_word(word) for word in t5_vocab]
        self._init_inverse_vocab()

        # Add custom tokenizer to avoid splitting the ['MASK'] token
        # custom_basic_tokenizer = CustomBaseTokenizer(do_lower_case = do_lower_case)
        # self.tokenizer.basic_tokenizer = custom_basic_tokenizer

        # Load pre-trained model (weights)
        # ... to get prediction/generation
        self.try_cuda()
        self.T5_config = T5Config.from_pretrained(dict_file)
        self.masked_T5_model = T5ForConditionalGeneration.from_pretrained(dict_file, config=self.T5_config)

        # ... to get hidden states
        # self.T5_model = self.masked_T5_model.bert

        # self.mask_id = self.inverse_vocab[T5_MASK]

    def get_id(self, string):
        # tokenized_text = self.tokenizer.tokenize(string)
        # indexed_string = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        encoded = self.tokenizer.encode_plus(string, add_special_tokens=True, return_tensors='pt')
        return encoded

    def _cuda(self):
        self.masked_T5_model.cuda()

    def get_batch_generation(self, sentences_list, logger= None,
                             try_cuda=True):
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()
        self.masked_T5_model.to(self._model_device)
        self.masked_T5_model.eval()

        # for sentence in sentences_list:
            # _0_index = sentence.index('<extra_id_0>')
            # _result_prefix = sentence[:_0_index]
            # _result_suffix = sentence[_0_index+12:]  # 12 is the length of <extra_id_0>
            # token_ids_list.append(encoded.input_ids)
            # masked_indices_list.append(encoded.attention_mask)

        tokens_tensor, segments_tensor, attention_mask_tensor, masked_indices_list, tokenized_text_list = self.__get_input_tensors_batch(sentences_list)


        if logger is not None:
            logger.debug("\n{}\n".format(tokenized_text_list))

        with torch.no_grad():
            logits = self.masked_T5_model(
                input_ids=tokens_tensor.to(self._model_device),
                decoder_input_ids=segments_tensor.to(self._model_device),
                attention_mask=attention_mask_tensor.to(self._model_device),
            )

            log_probs = F.log_softmax(logits.logits, dim=-1).cpu()

        token_ids_list = []
        for sentences in sentences_list:
            token_ids_list.append(self.get_id(sentences))

        return log_probs, token_ids_list, masked_indices_list


    def __get_input_tensors_batch(self, sentences_list):
        tokens_tensors_list = []
        segments_tensors_list = []
        masked_indices_list = []
        tokenized_text_list = []
        max_tokens = 0
        for sentences in sentences_list:
            tokens_tensor, segments_tensor, masked_indices, tokenized_text = self.__get_input_tensors(sentences)
            tokens_tensors_list.append(tokens_tensor)
            segments_tensors_list.append(segments_tensor)
            masked_indices_list.append(masked_indices)
            tokenized_text_list.append(tokenized_text)
            # assert(tokens_tensor.shape[1] == segments_tensor.shape[1])
            if (tokens_tensor.shape[1] > max_tokens):
                max_tokens = tokens_tensor.shape[1]
        # print("MAX_TOKENS: {}".format(max_tokens))
        # apply padding and concatenate tensors
        # use [PAD] for tokens and 0 for segments
        final_tokens_tensor = None
        final_segments_tensor = None
        final_attention_mask = None
        for tokens_tensor, segments_tensor in zip(tokens_tensors_list, segments_tensors_list):
            dim_tensor = tokens_tensor.shape[1]
            attention_tensor = torch.full([1,dim_tensor], 1, dtype= torch.long)
            final_tokens_tensor = tokens_tensor
            final_segments_tensor = segments_tensor
            final_attention_mask = attention_tensor
            
        return final_tokens_tensor, final_segments_tensor, final_attention_mask, masked_indices_list, tokenized_text_list

    def __get_input_tensors(self, sentences):

        if len(sentences) > 1:
            print(sentences)
            raise ValueError("T5 accepts maximum two sentences in input for each data point")

        first_tokenized_sentence = self.tokenizer.tokenize(sentences[0])
        first_segment_id = np.zeros(len(first_tokenized_sentence), dtype=int).tolist()

        # add [SEP] token at the end
        first_tokenized_sentence.append(T5_SEP)
        first_segment_id.append(0)

        if len(sentences)>1 :
            second_tokenized_sentece = self.tokenizer.tokenize(sentences[1])
            second_segment_id = np.full(len(second_tokenized_sentece),1, dtype=int).tolist()

            # add [SEP] token at the end
            second_tokenized_sentece.append(T5_SEP)
            second_segment_id.append(1)

            tokenized_text = first_tokenized_sentence + second_tokenized_sentece
            segments_ids = first_segment_id + second_segment_id
        else:
            tokenized_text = first_tokenized_sentence
            segments_ids = first_segment_id

        # look for masked indices
        masked_indices = []
        for i in range(len(tokenized_text)):
            token = tokenized_text[i]
            if token == T5_MASK:
                masked_indices.append(i)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        return tokens_tensor, segments_tensors, masked_indices, tokenized_text