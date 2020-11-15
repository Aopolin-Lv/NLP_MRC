# coding=utf-8
"""
@Time:          2020/11/14 2:15 下午
@Author:        Aopolin
@File:          config.py
@Contact:       aopolin.ii@gmail.com
@Description:
"""
class SquadConfig(object):
    def __init__(self):
        self.SQUAD_DIR = "../../dataset/squad2.0"
        self.model_type = "bert"
        self.model_name_or_path = "bert-base-uncased"
        self.do_train = True
        self.do_eval = True
        self.do_lower_case = True
        self.train_file = self.SQUAD_DIR + "/train-v2.0_small.json"
        self.predict_file = self.SQUAD_DIR + "/dev-v2.0.json"
        self.per_gpu_train_batch = 12
        self.learning_rate = 3e-5
        self.num_train_epochs = 2.0
        self.max_seq_length = 384
        self.doc_stride = 128
        self.output_dir = "/tmp/debug_squad/"

        self.max_query_length = 64
        self.config_name = ""
        self.tokenizer_name = ""
        self.cache_dir = ""
        self.version_2_with_negative = True
        self.null_score_diff_threshold = 0.0
        self.evaluate_during_training = False
        self.per_gpu_eval_batch_size = 8
        self.gradient_accumulation_steps = 1

        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.max_steps = -1
        self.warmup_steps = 0
        self.n_best_size = 20
        self.max_answer_length = 30
        self.verbose_logging = False
        self.lang_id = 0
        self.logging_steps = 500
        self.save_steps = 500
        self.no_cuda = True
        self.seed = 42
        self.local_rank = -1
        self.fp16_opt_level = "01"

        self.server_ip = ""
        self.server_port = ""
        self.threads = 1
        self.fp16 = False
        self.overwrite_cache = False
        self.eval_all_checkpoints = False
        self.data_dir = ""

        self.bert_dir = "../../Model_files/bert-base-uncased/"
        self.device = "cpu"