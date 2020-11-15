# coding=utf-8
"""
@Time:          2020/11/15 1:53 下午
@Author:        Aopolin
@File:          test.py
@Contact:       aopolin.ii@gmail.com
@Description:
"""
from Code.config import SquadConfig
import json
from transformers import AutoModelForQuestionAnswering, AutoConfig, AutoTokenizer
from transformers.tokenization_utils_base import TruncationStrategy
import numpy as np
from transformers.data.datasets import squad
from transformers.data.processors.squad import SquadFeatures
from tqdm import tqdm

class SquadExample:
    """
    A single training/test example for the Squad dataset, as loaded from disk.

    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
        self,
        qas_id,
        question_text,
        context_text,
        answer_text,
        start_position_character,
        title,
        answers=[],
        is_impossible=False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers

        self.start_position, self.end_position = 0, 0

        # 将答案中的起始字符与context句子中的单词对应
        doc_tokens = []                                     # 以空格或者tab为分隔符，将context分隔成doc_tokens
        char_to_word_offset = []                            # 第i个字符对应的第j个单词
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start and end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]                                                 # 不能超出最大的单词个数

def is_whitespace(c):
    if c == " " or c == "\t"  or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def read_squad_example(input_file, is_training):
    """
    读入squad数据
    :param input_file:
    :param is_training:
    :return:
    """
    with open(input_file, "r", encoding="utf-8") as reader:
        input_data = json.load(reader)["data"]

    examples = []

    for entry in input_data:
        title = entry["title"]
        count = 0
        # 这里的paragraph表示每一个paragraph中的qas-pair对
        for paragraph in entry["paragraphs"]:
            context_text = paragraph["context"]
            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position_character = None
                answer_text = None
                answers = []

                is_impossible = qa.get("is_impossible", False)
                if not is_impossible:
                    if is_training:
                        answer = qa["answers"][0]
                        answer_text = answer["text"]
                        start_position_character = answer["answer_start"]
                    else:
                        answers = qa["answers"]

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    context_text=context_text,
                    answer_text=answer_text,
                    start_position_character=start_position_character,
                    title=title,
                    is_impossible=is_impossible,
                    answers=answers,
                )
                examples.append(example)
    return examples

def whitespace_tokenize(text):
    """
    Runs basic whitespace cleaning and splitting on a piece of text.
    :param text:
    :return:
    """
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """
    Returns tokenized answer spans that better match the annotated answer.
    :param doc_tokens:
    :param input_start:
    :param input_end:
    :param tokenizer:
    :param orig_answer_text:
    :return:

    Example:
        doc_tokens:
            ['beyonce', 'gi', '##selle', 'knowles', '-', 'carter', '(', '/', 'bi', '##ː', '##ˈ', '##j',
            '##ɒ', '##nse', '##ɪ', '/', 'bee', '-', 'yo', '##n', '-', 'say', ')', '(', 'born', 'september', '4', ',',
            '1981', ')', 'is', 'an', 'american', 'singer', ',', 'songwriter', ',', 'record', 'producer', 'and', 'actress',
            '.', 'born', 'and', 'raised', 'in', 'houston', ',', 'texas', ',', 'she', 'performed', 'in', 'various',
            'singing', 'and', 'dancing', 'competitions', 'as', 'a', 'child', ',', 'and', 'rose', 'to', 'fame', 'in',
            'the', 'late', '1990s', 'as', 'lead', 'singer', 'of', 'r', '&',  'b', 'girl', '-', 'group', 'destiny', "'",
            's', 'child', '.', 'managed', 'by', 'her', 'father', ',', 'mathew', 'knowles', ',', 'the', 'group', 'became', '
            one', 'of', 'the', 'world', "'", 's', 'best', '-', 'selling', 'girl', 'groups', 'of', 'all', 'time', '.',
            'their', 'hiatus', 'saw', 'the', 'release', 'of', 'beyonce', "'", 's', 'debut', 'album', ',', 'dangerously',
            'in', 'love', '(', '2003', ')', ',', 'which', 'established', 'her', 'as', 'a', 'solo', 'artist', 'worldwide',
            ',', 'earned', 'five', 'grammy', 'awards', 'and', 'featured', 'the', 'billboard', 'hot', '100', 'number', '-',
             'one', 'singles', '"', 'crazy', 'in', 'love', '"', 'and', '"', 'baby', 'boy', '"', '.']
        input_start:
            66
        input_end:
            69
        tokenizer:
            <transformers.tokenization_bert.BertTokenizer object at 0x000001C1CE9D4F98>
        orig_answer_text:
            in the 1990s
    """
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))
    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)

def squad_convert_examples_to_features(
        examples,
        tokenizer,
        max_seq_length,
        doc_stride,
        max_query_length,
        is_training,
        padding_strategy="max_length",
        return_dataset=False,
        threads=1,
        tqdm_enabled=False,
):
    """
    问题若超过max_query_length则会被阶段前半部分，
    文档若超过max_seq_length则会使用滑窗法
    :param examples:
    :param tokenizer:
    :param max_seq_length:
    :param doc_stride:
    :param max_query_length:
    :param is_training:
    :param padding_strategy:
    :param return_dataset:
    :param threads:
    :param tqdm_enabled:
    :return:
    """
    features = []
    unique_id = 1000000000

    def squad_convert_example_to_features(
            example, max_seq_length, doc_stride, max_query_length, padding_strategy, is_training
    ):
        features = []
        if is_training and not example.is_impossible:
            # Get start and end position
            start_position = example.start_position
            end_position = example.end_position

            # If the answer cannot be found in the text, then skip this example.
            actual_text = " ".join(example.doc_tokens[start_position: (end_position + 1)])
            cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
            if actual_text.find(cleaned_answer_text) == -1:
                logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                return []

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            if tokenizer.__class__.__name__ in [
                "RobertaTokenizer",
                "LongformerTokenizer",
                "BartTokenizer",
                "RobertaTokenizerFast",
                "LongformerTokenizerFast",
                "BartTokenizerFast",
            ]:
                sub_tokens = tokenizer.tokenize(token, add_prefix_space=True)
            else:
                sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1

            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
            )

        spans = []

        truncated_query = tokenizer.encode(
            example.question_text, add_special_tokens=False, truncation=True, max_length=max_query_length
        )

        # Tokenizers who insert 2 SEP tokens in-between <context> & <question> need to have special handling
        # in the way they compute mask of added tokens.
        tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
        sequence_added_tokens = tokenizer.max_len - tokenizer.max_len_single_sentence
        sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair

        span_doc_tokens = all_doc_tokens
        while len(spans) * doc_stride < len(all_doc_tokens):

            # Define the side we want to truncate / pad and the text/pair sorting
            if tokenizer.padding_side == "right":
                texts = truncated_query
                pairs = span_doc_tokens
                truncation = TruncationStrategy.ONLY_SECOND.value
            else:
                texts = span_doc_tokens
                pairs = truncated_query
                truncation = TruncationStrategy.ONLY_FIRST.value

            encoded_dict = tokenizer.encode_plus(  # TODO(thom) update this logic
                texts,
                pairs,
                truncation=truncation,
                padding=padding_strategy,
                max_length=max_seq_length,
                return_overflowing_tokens=True,
                stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
                return_token_type_ids=True,
            )

            paragraph_len = min(
                len(all_doc_tokens) - len(spans) * doc_stride,
                max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
            )

            if tokenizer.pad_token_id in encoded_dict["input_ids"]:
                if tokenizer.padding_side == "right":
                    non_padded_ids = encoded_dict["input_ids"][
                                     : encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
                else:
                    last_padding_id_position = (
                            len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(
                        tokenizer.pad_token_id)
                    )
                    non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1:]

            else:
                non_padded_ids = encoded_dict["input_ids"]

            tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

            token_to_orig_map = {}
            for i in range(paragraph_len):
                index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
                token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

            encoded_dict["paragraph_len"] = paragraph_len
            encoded_dict["tokens"] = tokens
            encoded_dict["token_to_orig_map"] = token_to_orig_map
            encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
            encoded_dict["token_is_max_context"] = {}
            encoded_dict["start"] = len(spans) * doc_stride
            encoded_dict["length"] = paragraph_len

            spans.append(encoded_dict)

            if "overflowing_tokens" not in encoded_dict or (
                    "overflowing_tokens" in encoded_dict and len(encoded_dict["overflowing_tokens"]) == 0
            ):
                break
            span_doc_tokens = encoded_dict["overflowing_tokens"]

        for doc_span_index in range(len(spans)):
            for j in range(spans[doc_span_index]["paragraph_len"]):
                is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
                index = (
                    j
                    if tokenizer.padding_side == "left"
                    else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
                )
                spans[doc_span_index]["token_is_max_context"][index] = is_max_context

        for span in spans:
            # Identify the position of the CLS token
            cls_index = span["input_ids"].index(tokenizer.cls_token_id)

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # Original TF implem also keep the classification token (set to 0)
            p_mask = np.ones_like(span["token_type_ids"])
            if tokenizer.padding_side == "right":
                p_mask[len(truncated_query) + sequence_added_tokens:] = 0
            else:
                p_mask[-len(span["tokens"]): -(len(truncated_query) + sequence_added_tokens)] = 0

            pad_token_indices = np.where(span["input_ids"] == tokenizer.pad_token_id)
            special_token_indices = np.asarray(
                tokenizer.get_special_tokens_mask(span["input_ids"], already_has_special_tokens=True)
            ).nonzero()

            p_mask[pad_token_indices] = 1
            p_mask[special_token_indices] = 1

            # Set the cls index to 0: the CLS index can be used for impossible answers
            p_mask[cls_index] = 0

            span_is_impossible = example.is_impossible
            start_position = 0
            end_position = 0
            if is_training and not span_is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = span["start"]
                doc_end = span["start"] + span["length"] - 1
                out_of_span = False

                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    out_of_span = True

                if out_of_span:
                    start_position = cls_index
                    end_position = cls_index
                    span_is_impossible = True
                else:
                    if tokenizer.padding_side == "left":
                        doc_offset = 0
                    else:
                        doc_offset = len(truncated_query) + sequence_added_tokens

                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            features.append(
                SquadFeatures(
                    span["input_ids"],
                    span["attention_mask"],
                    span["token_type_ids"],
                    cls_index,
                    p_mask.tolist(),
                    example_index=0,
                    # Can not set unique_id and example_index here. They will be set after multiple processing.
                    unique_id=0,
                    paragraph_len=span["paragraph_len"],
                    token_is_max_context=span["token_is_max_context"],
                    tokens=span["tokens"],
                    token_to_orig_map=span["token_to_orig_map"],
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=span_is_impossible,
                    qas_id=example.qas_id,
                )
            )
        return features

    for example in tqdm(examples, total=len(examples), desc="convert squad examples to features"):
        """
        example:
            qas_id: 56be85543aeaaa14008c9063,
            question_text: When did Beyonce start becoming popular?
            doc_tokens: [Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an 
                        American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, 
                        she performed in various singing and dancing competitions as a child, and rose to fame in the 
                        late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew 
                        Knowles, the group became one of the world's best-selling girl groups of all time. Their 
                        hiatus saw the release of Beyoncé's debut album, Dangerously in Love (2003), which established 
                        her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 
                        number-one singles "Crazy in Love" and "Baby Boy".]
            start_position: 39
            end_position: 42
        """
        feature = squad_convert_example_to_features(example,
                                                     max_seq_length=max_seq_length,
                                                     max_query_length=max_query_length,
                                                     doc_stride=doc_stride,
                                                     padding_strategy=padding_strategy,
                                                     is_training=True)
        features.append(feature)

        # query_tokens = tokenizer.tokenize(example.question_text)
        # # query_tokens: ["when", "did", "beyonce", "start", "becoming", "popular", "?"]
        #
        # if is_training and not example.is_impossible:
        #     # Get start and position
        #     start_position = example.start_position
        #     end_position = example.end_position
        #
        #     # If the answer cannot be found in the text, then skip this example.
        #     actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)])
        #     cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
        #     if actual_text.find(cleaned_answer_text) == -1:
        #         logger.warning("Could not find answer: '%s' vs. '%s'", acutal_text, cleaned_answer_text)
        #         return []
        #
        # tok_to_orig_index = []                              # token对应原始的单词下标
        # orig_to_tok_index = []                              # 原始单词对应的token下标
        # all_doc_tokens = []                                 # 所有原先的context进行tokenize之后组成的list
        # for (i, token) in enumerate(example.doc_tokens):
        #     orig_to_tok_index.append(len(all_doc_tokens))
        #     sub_tokens = tokenizer.tokenize(token)
        #     """
        #     sub_tokens: 原先的token以空格切分为一个个的词，现在对一个单词再进行切分
        #     ['beyonce']
        #     ['qi', '##selle']
        #     ['knowles', '-', 'carter']
        #     ['(', '/', 'bi', '##:', '##', '##j', '##ɒ', '##nse', '##r', '/']
        #     """
        #
        #     for sub_token in sub_tokens:
        #         tok_to_orig_index.append(i)
        #         all_doc_tokens.append(sub_token)
        # """
        # tok_to_orig_index
        #     [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 6, 7, 7, 8, 8, 9, 10, 11, 12,
        #     12, 13, 13, 14, 15, 16, 17, 17, 18, 19, 20, 21, 22, 22, 23, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
        #     34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 47, 47, 48, 48, 48, 49, 49, 49, 50, 50, 51, 52, 53,
        #     54, 54, 55, 56, 56, 57, 58, 59, 60, 61, 62, 63, 63, 63, 64, 64, 64, 65, 66, 67, 68, 69, 69, 70, 71, 72, 73,
        #     74, 75, 76, 76, 76, 77, 78, 78, 79, 80, 81, 82, 82, 82, 82, 83, 84, 85, 86, 87, 88, 89, 90, 90, 91, 92, 93,
        #     94, 95, 96, 97, 98, 99, 100, 101, 101, 101, 102, 103, 103, 104, 105, 105, 106, 107, 107, 108, 108, 108]
        #
        # orig_to_tok_index
        #  [0, 1, 3, 6, 16, 23, 25, 26, 28, 30, 31, 32, 33, 35, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 50, 51, 52, 53,
        #  54, 55, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 77, 80, 83, 85, 86, 87, 88, 90,
        #  91, 93, 94, 95, 96, 97, 98, 99, 102, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 120, 121, 123,
        #  124, 125, 126, 130, 131, 132, 133, 134, 135, 136, 137, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
        #  152, 153, 155, 156, 158, 159, 161]
        #
        # all_doc_tokens
        #     ['beyonce', 'gi', '##selle', 'knowles', '-', 'carter', '(', '/', 'bi', '##ː', '##ˈ', '##j', '##ɒ', '##nse',
        #     '##ɪ', '/', 'bee', '-', 'yo', '##n', '-', 'say', ')', '(', 'born', 'september', '4', ',', '1981', ')', 'is',
        #      'an', 'american', 'singer', ',', 'songwriter', ',', 'record', 'producer', 'and', 'actress', '.', 'born',
        #      'and', 'raised', 'in', 'houston', ',', 'texas', ',', 'she', 'performed', 'in', 'various', 'singing', 'and',
        #      'dancing', 'competitions', 'as', 'a', 'child', ',', 'and', 'rose', 'to', 'fame', 'in', 'the', 'late',
        #      '1990s', 'as', 'lead', 'singer', 'of', 'r', '&', 'b', 'girl', '-', 'group', 'destiny', "'", 's', 'child',
        #      '.', 'managed', 'by', 'her', 'father', ',', 'mathew', 'knowles', ',', 'the', 'group', 'became', 'one',
        #      'of', 'the', 'world', "'", 's', 'best', '-', 'selling', 'girl', 'groups', 'of', 'all', 'time', '.',
        #      'their', 'hiatus', 'saw', 'the', 'release', 'of', 'beyonce', "'", 's', 'debut', 'album', ',', 'dangerously',
        #       'in', 'love', '(', '2003', ')', ',', 'which', 'established', 'her', 'as', 'a', 'solo', 'artist',
        #       'worldwide', ',', 'earned', 'five', 'grammy', 'awards', 'and', 'featured', 'the', 'billboard', 'hot',
        #       '100', 'number', '-', 'one', 'singles', '"', 'crazy', 'in', 'love', '"', 'and', '"', 'baby', 'boy', '"', '.']
        # """
        #
        # if is_training and not example.is_impossible:
        #     tok_start_position = orig_to_tok_index[example.start_position]
        #     if example.end_position < len(example.doc_tokens) - 1:
        #         tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        #     else:
        #         tok_end_position = len(all_doc_tokens) - 1
        #
        #     (tok_start_position, tok_end_position) = _improve_answer_span(all_doc_tokens,
        #                                                                   tok_start_position,
        #                                                                   tok_end_position,
        #                                                                   tokenizer,
        #                                                                   example.answer_text)
        # # 滑窗法
        # spans = []
        #
        # # 对问题进行encode
        # truncated_query = tokenizer.encode(
        #     example.question_text, add_special_tokens=False, truncation=True, max_length=max_query_length
        # )
        #
        # # Tokenizers who insert 2SEP tokens in-between <context> & <question> need to have special handling
        # # in the way they compute mask of added tokens
        # tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
        # sequence_added_tokens = tokenizer.max_len - tokenizer.max_len_single_sentence
        # sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair
        #
        # span_doc_tokens = all_doc_tokens
        # while len(spans) * doc_stride < len(all_doc_tokens):
        #
        #     # Define the side we want to truncate / pad and the text/pair sorting
        #     if tokenizer.padding_side == "right":
        #         texts = truncated_query
        #         pairs = span_doc_tokens
        #         truncation = TruncationStrategy.ONLY_SECOND.value
        #     else:
        #         texts = span_doc_tokens
        #         pairs = truncated_query
        #         truncation = TruncationStrategy.ONLY_FIRST.value
        #
        #     # 将输入的文本和答案转换成模型接受的输入
        #     encoded_dict = tokenizer.encode_plus(
        #         texts,
        #         pairs,
        #         truncation=truncation,
        #         padding=padding_strategy,
        #         max_length=max_seq_length,
        #         return_overflowing_tokens=True,
        #         stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
        #         return_token_type_ids=True,
        #     )
        #
        #     paragraph_len = min(
        #         len(all_doc_tokens) - len(spans) * doc_stride,
        #         max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
        #     )
        #
        #     if tokenizer.pad_token_id in encoded_dict["input_ids"]:
        #         if tokenizer.padding_side == "right":
        #             non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
        #         else:
        #             last_padding_id_position = (
        #                 len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id)
        #             )
        #             non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1:]
        #
        #     else:
        #         non_padded_ids = encoded_dict["input_ids"]
        #
        #     tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)
        #
        #     token_to_orig_map = {}
        #     for i in range(paragraph_len):
        #         index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
        #         token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]
        #
        #     encoded_dict["paragraph_len"] = paragraph_len
        #     encoded_dict["tokens"] = tokens
        #     encoded_dict["token_to_orig_map"] = token_to_orig_map
        #     encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
        #     encoded_dict["token_is_max_context"] = {}
        #     encoded_dict["start"] = len(spans) * doc_stride
        #     encoded_dict["length"] = paragraph_len
        #     spans.append(encoded_dict)
        #
        #     if "overflowing_tokens" not in encoded_dict or (
        #         "overflowing_tokens" in encoded_dict and len(encoded_dict["overflowing_tokens"]) == 0
        #     ):
        #         break
        #     span_doc_tokens = encoded_dict["overflowing_tokens"]
        #
        # for doc_span_index in range(len(spans)):
        #     for j in range(spans[doc_span_index]["paragraph_len"]):
        #         is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
        #         index = (
        #             j
        #             if tokenizer.padding_side == "left"
        #             else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
        #         )
        #         spans[doc_span_index]["token_is_max_context"][index] = is_max_context
        #
        # for span in spans:
        #     # Identify the position of the CLS token
        #     cls_index = span["input_ids"].index(tokenizer.cls_token_id)
        #
        #     # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        #     # Original TF implem also keep the classification token (set to 0)
        #     p_mask = np.ones_like(span["token_type_ids"])
        #     if tokenizer.padding_side == "right":
        #         p_mask[len(truncated_query) + sequence_added_tokens :] = 0
        #     else:
        #         p_mask[-len(span["tokens"]) : -(len(truncated_query) + sequence_added_tokens)] = 0
        #
        #     pad_token_indices = np.where(span["input_ids"] == tokenizer.pad_token_id)
        #     special_token_indices = np.asarray(
        #         tokenizer.get_special_tokens_mask(span["input_ids"], already_has_special_tokens=True)
        #     ).nonzero()
        #
        #     p_mask[pad_token_indices] = 1
        #     p_mask[special_token_indices] = 1
        #
        #     # Set the cls index to 0: the CLS index can be used for impossible answers
        #     p_mask[cls_index] = 0
        #
        #     span_is_impossible = example.is_impossible
        #     start_position = 0
        #     end_position = 0
        #     if is_training and not span_is_impossible:
        #         # For training, if our document chunk does not contain an annotation
        #         # we throw it out, since there is nothing to predict.
        #         doc_start = span["start"]
        #         doc_end = span["start"] + span["length"] - 1
        #         out_of_span = False
        #
        #         if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
        #             out_of_span = True
        #
        #         if out_of_span:
        #             start_position = cls_index
        #             end_position = cls_index
        #             span_is_impossible = True
        #         else:
        #             if tokenizer.padding_side == "left":
        #                 doc_offset = 0
        #             else:
        #                 doc_offset = len(truncated_query) + sequence_added_tokens
        #
        #             start_position = tok_start_position - doc_start + doc_offset
        #             end_position = tok_end_position - doc_start + doc_offset
        #
        #     features.append(
        #         SquadFeatures(
        #             span["input_ids"],
        #             span["attention_mask"],
        #             span["token_type_ids"],
        #             cls_index,
        #             p_mask.tolist(),
        #             example_index=0,  # Can not set unique_id and example_index here. They will be set after multiple processing.
        #             unique_id=0,
        #             paragraph_len=span["paragraph_len"],
        #             token_is_max_context=span["token_is_max_context"],
        #             tokens=span["tokens"],
        #             token_to_orig_map=span["token_to_orig_map"],
        #             start_position=start_position,
        #             end_position=end_position,
        #             is_impossible=span_is_impossible,
        #             qas_id=example.qas_id,
        #         )
        #     )
    return features


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

if __name__ == "__main__":
    opt = SquadConfig()
    opt.model_type = opt.model_type.lower()
    examples = read_squad_example(opt.train_file, True)

    config = AutoConfig.from_pretrained(opt.bert_dir)
    tokenizer = AutoTokenizer.from_pretrained(opt.bert_dir)
    model = AutoModelForQuestionAnswering.from_pretrained(opt.bert_dir)

    features = squad_convert_examples_to_features(examples=examples, tokenizer=tokenizer, max_seq_length=opt.max_seq_length,
                                       doc_stride=opt.doc_stride, max_query_length=opt.max_query_length, is_training=True)

    model.to(opt.device)
