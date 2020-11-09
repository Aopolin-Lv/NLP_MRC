# coding=utf-8
"""
@Time:          2020/11/9 9:15 下午
@Author:        Aopolin
@File:          run_squad.py
@Contact:       aopolin.ii@gmail.com
@Description:   Finetuning the library models for question-answering on SQUAD2.0
"""
import argparse
from transformers import MODEL_FOR_QUESTION_ANSWERING_MAPPING

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ",".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        requried=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
         + "If no data dir or train/predict files are specified, will run with tensorflow_datasets",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    # 用来处理SQuAD2.0中不可回答的问题
    parser.add_argument(
        "version_2_with_negative",
        action="store_true",                # 只要运行时该变量有传参就将该变量设为True
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "-null_score_diff_threshold",
        type=float,
        default=0.0,
        # 如果null_score-best_non_null大于阈值，则预测为null
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )