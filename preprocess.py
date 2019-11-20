# coding=utf-8
import json
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from utils import tokenization
import numpy as np
import pickle
import collections


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


class SquadExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self,
                 uuid,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.uuid = uuid
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "uuid: %s" % (tokenization.printable_text(self.uuid))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.question_text))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 qid,
                 uuid,
                 doc_span_index,
                 token_to_orig_map,
                 token_is_max_context,
                 doc_tokens,
                 ques_tokens,
                 start_position=None,
                 end_position=None):
        self.qid = qid
        self.uuid = uuid
        self.doc_span_index = doc_span_index
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.doc_tokens = doc_tokens
        self.ques_tokens = ques_tokens
        self.start_position = start_position
        self.end_position = end_position


# first preprocess to get tokens
'''

'''
def read_squad_examples(input_file, is_training):
    """Read a SQuAD json file into a list of SquadExample."""
    total = 0
    with open(input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in tqdm(input_data):
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                uuid = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                if is_training:
                    if len(qa["answers"]) > 1:
                        raise ValueError("For training, each question should have exactly 0 or 1 answer.")
                    answer = qa["answers"][0]
                    orig_answer_text = answer["text"]
                    ##按照字符计算从0开始，答案开始的位置
                    answer_offset = answer["answer_start"]
                    answer_length = len(orig_answer_text)
                    ##转成按照单词计算答案开始的位置
                    start_position = char_to_word_offset[answer_offset]
                    end_position = char_to_word_offset[answer_offset + answer_length - 1]
                    # Only add answers where the text can be exactly recovered from the
                    # document. If this CAN'T happen it's likely due to weird Unicode
                    # stuff so we will just skip the example.
                    #
                    # Note that this means for training mode, every example is NOT
                    # guaranteed to be preserved.
                    ##doc_tokens是列表里面存放每个单词
                    actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                    cleaned_answer_text = " ".join(tokenization.whitespace_tokenize(orig_answer_text))
                    if actual_text.find(cleaned_answer_text) == -1:
                        print("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                        continue

                total += 1

                example = SquadExample(
                    uuid=uuid,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position)
                examples.append(example)
                '''
                #每个问题构造一个example,同一个上下文中有多个问题，要将上下文复制多次
                ##所有段落，每个问题成为一个example，
                [uuid: 5733be284776f41900661182,
                 question_text: To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?,
                doc_tokens: [Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.], 
                start_position: 90, 
                end_position: 92, 
                
                uuid: 5733be284776f4190066117f, 
                question_text: What is in front of the Notre Dame Main Building?, 
                doc_tokens: [Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.], 
                start_position: 32, 
                end_position: 36,
                
                 uuid: 5733be284776f41900661180, 
                 question_text: The Basilica of the Sacred heart at Notre Dame is beside to which structure?, 
                 doc_tokens: [Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.], 
                 start_position: 49, 
                 end_position: 51]
                '''

    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length=384, doc_stride=128, max_query_length=64, is_training=True):
    """Loads a data file into a list of `InputBatch`s."""

    qid = 0
    features = []
    for example_index, example in enumerate(tqdm(examples)):
        uuid = example.uuid
        query_tokens = tokenizer.tokenize(example.question_text)
        #截取最大的询问长度
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        ## 这里是tokenizer之后 第sub_token对应的原始 单词的index；B和.都是对应index=2
        #<class 'list'>: [0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 33, 34, 35, 36, 36, 37, 37, 37, 37, 38, 39, 40, 41, 42, 43, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 54, 55, 55, 56, 57, 57, 57, 58, 59, 59, 59, 60, 61, 62, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 72, 73, 74, 74, 75, 76, 76, 77, 78, 79, 80, 81...
        tok_to_orig_index = []


        # 原始单词经过token之后，加入all_doc_tokens中单词的总量，一次性增加了B和.所以从2，直接到4，空时为0
        #<class 'list'>: [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 41, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 64, 66, 67, 70, 71, 74, 75, 76, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 89, 90, 92, 93, 95, 96, 97, 98, 99, 100, 101, 102, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
        orig_to_tok_index = []

        ### 这里添加用tokenizer分词之后的tokens
        #<class 'list'>: ['The', 'Joan', 'B', '.', 'Kroc', 'Institute', 'for', 'International', 'Peace', 'Studies', 'at', 'the', 'University', 'of', 'Notre', 'Dame', 'is', 'dedicated', 'to', 'research', ',', 'education', 'and', 'outreach', 'on', 'the', 'causes', 'of', 'violent', 'conflict', 'and', 'the', 'conditions', 'for', 'sustainable', 'peace', '.', 'It', 'offers', 'PhD', ',', 'Master', "'", 's', ',', 'and', 'undergraduate', 'degrees', 'in', 'peace', 'studies', '.', 'It', 'was', 'founded', 'in', '1986', 'through', 'the', 'donations', 'of', 'Joan', 'B', '.', 'Kroc', ',', 'the', 'wi', 'do', 'w', 'of', 'McDonald', "'", 's', 'owner', 'Ray', 'Kroc', '.', 'The', 'institute', 'was', 'inspired', 'by', 'the', 'vision', 'of', 'the', 'Rev', '.', 'Theodore', 'M', '.', 'Hesburgh', 'CSC', ',', 'President', 'Emeritus', 'of', 'the', 'University'...
        all_doc_tokens = []

        #example.doc_tokens
        #<class 'list'>: ['The', 'Joan', 'B.', 'Kroc', 'Institute', 'for', 'International', 'Peace', 'Studies', 'at', 'the', 'University', 'of', 'Notre', 'Dame', 'is', 'dedicated', 'to', 'research,', 'education', 'and', 'outreach', 'on', 'the', 'causes', 'of', 'violent', 'conflict', 'and', 'the', 'conditions', 'for', 'sustainable', 'peace.', 'It', 'offers', 'PhD,', "Master's,", 'and', 'undergraduate', 'degrees', 'in', 'peace', 'studies.', 'It', 'was', 'founded', 'in', '1986', 'through', 'the', 'donations', 'of', 'Joan', 'B.', 'Kroc,', 'the', 'widow', 'of', "McDonald's", 'owner', 'Ray', 'Kroc.', 'The', 'institute', 'was', 'inspired', 'by', 'the', 'vision', 'of', 'the', 'Rev.', 'Theodore', 'M.', 'Hesburgh', 'CSC,', 'President', 'Emeritus', 'of', 'the', 'University', 'of', 'Notre', 'Dame.', 'The', 'institute', 'has', 'contributed', 'to', 'international', 'policy', 'discussions', 'about', 'peace', 'building', 'practices.']
        for i, token in enumerate(example.doc_tokens):
            try:
                #原始单词对应token的index
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    ## 这里是tokenizer之后 第sub_token对应的原始 单词的index
                    tok_to_orig_index.append(i)
                    ## 这里添加用tokenizer分词之后的tokens
                    all_doc_tokens.append(sub_token)
            except Exception as e:
                print(e)
                pass
        ## 下面这一段是得到通过model.tokenizer分词之后的答案所在位置
        tok_start_position = None
        tok_end_position = None
        if is_training:
            #分词之后答案开始位置
            tok_start_position = orig_to_tok_index[example.start_position]
            ##结束位置不是在最后一个位置
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                ##？？没看懂，待会再看结束位置是在最后一个位置，len(all_doc_tokens)-1长度表示里面增加过多少次单词，即原始有多少个单词
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])

        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_seq_length:
                length = max_seq_length
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            doc_tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(doc_tokens)] = tok_to_orig_index[split_token_index]
                is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(doc_tokens)] = is_max_context
                doc_tokens.append(all_doc_tokens[split_token_index])

            start_position = None
            end_position = None
            if is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    continue
                else:
                    start_position = tok_start_position - doc_start
                    end_position = tok_end_position - doc_start

            features.append(InputFeatures(
                qid=qid,#从0开始，后面每次加1
                uuid=uuid,#和问题的id对应
                doc_span_index=doc_span_index,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                doc_tokens=doc_tokens,
                ques_tokens=query_tokens,
                start_position=start_position,
                end_position=end_position))
            '''
                qid=0
                uuid='5733be284776f41900661182'
                doc_span_index=0
                token_to_orig_map=<class 'dict'>: {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 6, 9: 7, 10: 8, 11: 9, 12: 10, 13: 10, 14: 10, 15: 11, 16: 12, 17: 13, 18: 14, 19: 15, 20: 16, 21: 17, 22: 18, 23: 19, 24: 20, 25: 20, 26: 21, 27: 22, 28: 23, 29: 24, 30: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 30, 37: 31, 38: 32, 39: 33, 40: 34, 41: 35, 42: 36, 43: 37, 44: 38, 45: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 43, 51: 44, 52: 45, 53: 46, 54: 46, 55: 46, 56: 47, 57: 48, 58: 49, 59: 50, 60: 51, 61: 52, 62: 53, 63: 54, 64: 55, 65: 56, 66: 57, 67: 58, 68: 58, 69: 59, 70: 60, 71: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 65, 77: 66, 78: 67, 79: 68, 80: 69, 81: 70, 82: 71, 83: 72, 84: 72, 85: 73, 86: 74, 87: 75, 88: 76, 89: 77, 90: 78, 91: 79, 92: 80, 93: 81, 94: 81, 95: 82, 96: 83, 97: 84, 98: 85, 99: 86...,
                token_is_max_context=<class 'dict'>: {0: True, 1: True, 2: True, 3: True, 4: True, 5: True, 6: True, 7: True, 8: True, 9: True, 10: True, 11: True, 12: True, 13: True, 14: True, 15: True, 16: True, 17: True, 18: True, 19: True, 20: True, 21: True, 22: True, 23: True, 24: True, 25: True, 26: True, 27: True, 28: True, 29: True, 30: True, 31: True, 32: True, 33: True, 34: True, 35: True, 36: True, 37: True, 38: True, 39: True, 40: True, 41: True, 42: True, 43: True, 44: True, 45: True, 46: True, 47: True, 48: True, 49: True, 50: True, 51: True, 52: True, 53: True, 54: True, 55: True, 56: True, 57: True, 58: True, 59: True, 60: True, 61: True, 62: True, 63: True, 64: True, 65: True, 66: True, 67: True, 68: True, 69: True, 70: True, 71: True, 72: True, 73: True, 74: True, 75: True, 76: True, 77: True, 78: True, 79: True, 80: True, 81: True, 82: True, 83: True, 84: True, 85: True, 86: True, 87: True, 88: True, 89: True, 90: True, 91: True, 92: True, 93: True, 94: True, 95: True, 96: True, 97: True, 98: True, 99: True...
                doc_tokens=<class 'list'>: ['Architecturally', ',', 'the', 'school', 'has', 'a', 'Catholic', 'character', '.', 'Atop', 'the', 'Main', 'Building', "'", 's', 'gold', 'dome', 'is', 'a', 'golden', 'statue', 'of', 'the', 'Virgin', 'Mary', '.', 'Immediately', 'in', 'front', 'of', 'the', 'Main', 'Building', 'and', 'facing', 'it', ',', 'is', 'a', 'copper', 'statue', 'of', 'Christ', 'with', 'arms', 'upraised', 'with', 'the', 'legend', '"', 'Venite', 'Ad', 'Me', 'Omnes', '"', '.', 'Next', 'to', 'the', 'Main', 'Building', 'is', 'the', 'Basilica', 'of', 'the', 'Sacred', 'Heart', '.', 'Immediately', 'behind', 'the', 'basilica', 'is', 'the', 'Grotto', ',', 'a', 'Marian', 'place', 'of', 'prayer', 'and', 'reflection', '.', 'It', 'is', 'a', 'replica', 'of', 'the', 'grotto', 'at', 'Lourdes', ',', 'France', 'where', 'the', 'Virgin', 'Mary'...
                ques_tokens=<class 'list'>: ['To', 'whom', 'did', 'the', 'Virgin', 'Mary', 'allegedly', 'appear', 'in', '1858', 'in', 'Lourdes', 'France', '?']
                start_position=103
                end_position=105
            '''

            qid += 1

    return features


def token_process(features, tokenizer, vocab_file):
    word_counter, unk_counter, char_counter = Counter(), Counter(), Counter()
    for feature in tqdm(features):
        doc_tokens = feature.doc_tokens
        ques_tokens = feature.ques_tokens
        for i, token in enumerate(doc_tokens):
            if token not in tokenizer.vocab:
                unk_counter[token] += 1
            else:
                word_counter[token] += 1
            for char in token:
                char_counter[char] += 1
        for token in ques_tokens:
            if token not in tokenizer.vocab:
                unk_counter[token] += 1
            else:
                word_counter[token] += 1
            for char in token:
                char_counter[char] += 1

    print('UNK / HIT :', len(unk_counter), '/', len(word_counter))
    print('CHAR num :', len(char_counter))

    # 过滤新vocab，将word_counter中未出现的从vocab中去除，出现过的赋值对应的下标
    word_embedding ,word_embedding_raw= tokenizer.get_word_embedding(word_counter, vocab_file, size=int(2.2e6), vec_size=300)
    char_embedding = tokenizer.get_char_embedding(char_counter, vec_size=64)

    return word_embedding, word_embedding_raw,char_embedding, tokenizer


def build_features(features, tokenizer, save_path, max_seq_length=384, max_query_length=64, char_limit=16, is_training=True):
    def convert_token_to_id(vocab, token):
        for each in (token, token.lower(), token.capitalize(), token.upper()):
            if each in vocab:
                return vocab[each]
        return vocab['--OOV--']
    def convert_char_to_id(vocab, char):
        if char in vocab:
            return vocab[char]
        return vocab['--OOV--']
    context_idxss = []
    ques_idxss = []
    context_char_idxss = []
    ques_char_idxss = []
    y1s = []
    y2s = []
    qids = []
    for feature in tqdm(features):
        try:
            qids.append(feature.qid)
            context_idxs = np.zeros([max_seq_length], dtype=np.int32)
            context_char_idxs = np.zeros([max_seq_length, char_limit], dtype=np.int32)
            ques_idxs = np.zeros([max_query_length], dtype=np.int32)
            ques_char_idxs = np.zeros([max_query_length, char_limit], dtype=np.int32)
            y1 = np.zeros([max_seq_length], dtype=np.float32)
            y2 = np.zeros([max_seq_length], dtype=np.float32)
            for i, token in enumerate(feature.doc_tokens):
                context_idxs[i] = convert_token_to_id(tokenizer.vocab, token)
                for j, char in enumerate(token):
                    if j == char_limit:
                        break
                    context_char_idxs[i, j] = convert_char_to_id(tokenizer.char_vocab, char)
            for i, token in enumerate(feature.ques_tokens):
                ques_idxs[i] = convert_token_to_id(tokenizer.vocab, token)
                for j, char in enumerate(token):
                    if j == char_limit:
                        break
                    ques_char_idxs[i, j] = convert_char_to_id(tokenizer.char_vocab, char)
            if is_training:
                y1[feature.start_position], y2[feature.end_position] = 1.0, 1.0
            context_idxss.append(np.expand_dims(context_idxs, axis=0))
            ques_idxss.append(np.expand_dims(ques_idxs, axis=0))
            context_char_idxss.append(np.expand_dims(context_char_idxs, axis=0))
            ques_char_idxss.append(np.expand_dims(ques_char_idxs, axis=0))
            if is_training:
                y1s.append(np.expand_dims(y1, axis=0))
                y2s.append(np.expand_dims(y2, axis=0))
        except Exception as e:
            print(e)
            pass
    context_idxss = np.concatenate(context_idxss, axis=0)
    ques_idxss = np.concatenate(ques_idxss, axis=0)
    context_char_idxss = np.concatenate(context_char_idxss, axis=0)
    ques_char_idxss = np.concatenate(ques_char_idxss, axis=0)
    if is_training:
        y1s = np.concatenate(y1s, axis=0)
        y2s = np.concatenate(y2s, axis=0)
    qids = np.array(qids)
    meta = {'qid': qids,
            'context_id': context_idxss,
            'question_id': ques_idxss,
            'context_char_id': context_char_idxss,
            'question_char_id': ques_char_idxss,
            'y_start': y1s,
            'y_end': y2s}
    print('save to', save_path, len(qids), 'features')
    '''
    
    <class 'dict'>: {'qid': array([0, 1, 2, 3, 4]), 'context_id': array([[1, 2, 3, ..., 0, 0, 0],
       [1, 2, 3, ..., 0, 0, 0],
       [1, 2, 3, ..., 0, 0, 0],
       [1, 2, 3, ..., 0, 0, 0],
       [1, 2, 3, ..., 0, 0, 0]]), 'question_id': array([[ 6, 48, 24,  3,  1,  1, 59, 45,  9, 63,  9,  1,  1, 19,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 1, 11,  9, 33,  7,  3,  1,  1, 35,  1, 19,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 3, 64,  7,  3,  1, 34, 18,  1,  1, 11, 57,  6, 22, 43, 19,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 1, 11,  3, 65, 18,  1,  1, 19,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 1, 55, 16, 28,  7,  3, 35,  1, 18,  1,  1, 19,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]]), 'context_char_id': array([[[ 2,  3,  4, ..., 11, 12,  0],
        [13,  0,  0, ...,  0,  0,  0],
        [ 7,  5,  8, ...,  0,  0,  0],
        ...,
        [ 0,  0,  0, ...,  0,  0,  0],
        [ 0,  0,  0, ...,  0,  0,  0],
        [ 0,  0,  0, ...,  0,  0,  0]],

       [[ 2,  3,  4, ..., 11, 12,  0],
        [13,  0,  0, ...,  0,  0,  0],
        [ 7,  5,  8, ...,  0,  0,  0],
        ...,
        [ 0,  0,  0, ...,  0,  0,  0],
        [ 0,  0,  0, ...,  0,  0,  0],
        [ 0,  0,  0, ...,  0,  0,  0]],

       [[ 2,  3,  4, ..., 11, 12,  0],
        [13,  0,  0, ...,  0,  0,  0],
        [ 7,  5,  8, ...,  0,  0,  0],
        ...,
        [ 0,  0,  0, ...,  0,  0,  0],
        [ 0,  0,  0, ...,  0,  0,  0],
        [ 0,  0,  0, ...,  0,  0,  0]],

       [[ 2,  3,  4, ..., 11, 12,  0],
        [13,  0,  0, ...,  0,  0,  0],
        [ 7,  5,  8, ...,  0,  0,  0],
        ...,
        [ 0,  0,  0, ...,  0,  0,  0],
        [ 0,  0,  0, ...,  0,  0,  0],
        [ 0,  0,  0, ...,  0,  0,  0]],

       [[ 2,  3,  4, ..., 11, 12,  0],
        [13,  0,  0, ...,  0,  0,  0],
        [ 7,  5,  8, ...,  0,  0,  0],
        ...,
        [ 0,  0,  0, ...,  0,  0,  0],
        [ 0,  0,  0, ...,  0,  0,  0],
        [ 0,  0,  0, ...,  0,  0,  0]]]), 'question_char_id': array([[[48, 15,  0, ...,  0,  0,  0],
        [29,  5, 15, ...,  0,  0,  0],
        [22,  6, 22, ...,  0,  0,  0],
        ...,
        [ 0,  0,  0, ...,  0,  0,  0],
        [ 0,  0,  0, ...,  0,  0,  0],
        [ 0,  0,  0, ...,  0,  0,  0]],

       [[50,  5, 10, ...,  0,  0,  0],
        [ 6, 14,  0, ...,  0,  0,  0],
        [ 6, 20,  0, ...,  0,  0,  0],
        ...,
        [ 0,  0,  0, ...,  0,  0,  0],
        [ 0,  0,  0, ...,  0,  0,  0],
        [ 0,  0,  0, ...,  0,  0,  0]],

       [[48,  5,  8, ...,  0,  0,  0],
        [21, 10, 14, ...,  0,  0,  0],
        [15, 26,  0, ...,  0,  0,  0],
        ...,
        [ 0,  0,  0, ...,  0,  0,  0],
        [ 0,  0,  0, ...,  0,  0,  0],
        [ 0,  0,  0, ...,  0,  0,  0]],

       [[50,  5, 10, ...,  0,  0,  0],
        [ 6, 14,  0, ...,  0,  0,  0],
        [ 7,  5,  8, ...,  0,  0,  0],
        ...,
        [ 0,  0,  0, ...,  0,  0,  0],
        [ 0,  0,  0, ...,  0,  0,  0],
        [ 0,  0,  0, ...,  0,  0,  0]],

       [[50,  5, 10, ...,  0,  0,  0],
        [14,  6,  7, ...,  0,  0,  0],
        [15, 20,  0, ...,  0,  0,  0],
        ...,
        [ 0,  0,  0, ...,  0,  0,  0],
        [ 0,  0,  0, ...,  0,  0,  0],
        [ 0,  0,  0, ...,  0,  0,  0]]]), 'y_start': array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]], dtype=float32), 'y_end': array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)}
    '''
    with open(save_path, 'wb') as f:
        pickle.dump(meta, f)


if __name__ == '__main__':

    # Load tokenizer
    tokenizer = tokenization.FullTokenizer(vocab_file='glove.42B.300d.txt1', do_lower_case=False)

    train_examples = read_squad_examples(input_file='original_data/train_sample.json', is_training=True)
    dev_examples = read_squad_examples(input_file='original_data/train_sample.json', is_training=False)

    

    train_features = convert_examples_to_features(train_examples, tokenizer, max_seq_length=400, max_query_length=50,is_training=True)
    dev_features = convert_examples_to_features(dev_examples, tokenizer, max_seq_length=400, max_query_length=50, is_training=False)



    total_features = []
    total_features.extend(train_features)
    total_features.extend(dev_features)
    word_embedding, word_embedding_raw,char_embedding, tokenizer = token_process(total_features, tokenizer, 'glove.42B.300d.txt1')
    # with open('glove.42B.300d.txt','w') as f:
    #     json.dump(word_embedding_raw,f,indent=2)
    print('保存完毕')
    print(word_embedding.shape)
    print(len(tokenizer.vocab))
    print(char_embedding.shape)
    print(len(tokenizer.char_vocab))

    preprocessDatasetPath = Path('./dataset_wordpiece/')
    if not preprocessDatasetPath.is_dir():
        preprocessDatasetPath.mkdir()

    np.save('./dataset_wordpiece/word_emb_mat.npy', word_embedding)
    np.save('./dataset_wordpiece/char_emb_mat.npy', char_embedding)
    
    # with open('./dataset_wordpiece/dev_examples.pkl', 'wb') as p:
    #     pickle.dump(dev_examples, p)
    # with open('./dataset_wordpiece/dev_features.pkl', 'wb') as p:
    #     pickle.dump(dev_features, p)


    build_features(train_features, tokenizer, './dataset_wordpiece/trainset_wordpiece.pkl', is_training=True)
    build_features(dev_features, tokenizer, './dataset_wordpiece/devset_wordpiece.pkl', is_training=False)
