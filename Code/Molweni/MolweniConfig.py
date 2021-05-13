# coding=utf-8
"""
@Time:          2020/11/14 2:15 下午
@Author:        Aopolin
@File:          MolweniConfig.py
@Contact:       aopolin.ii@gmail.com
@Description:
"""


class Config(object):
    def __init__(self):
        self.SQUAD_DIR = "../../Dataset/squad2.0"
        self.MOLWENI_DIR = "../../Dataset/Molweni"

        self.model_type = "bert"  # ["distilbert", "albert", "bert", "xlnet", ...]
        self.model_name_or_path = "bert-base-uncased"
        self.output_dir = "/tmp/debug_squad/"  # 输出目录路径
        self.data_dir = ""
        self.train_file = self.MOLWENI_DIR + "/train_small.json"
        self.predict_file = self.MOLWENI_DIR + "/dev_small.json"
        self.config_name = ""
        self.tokenizer_name = ""
        self.cache_dir = ""
        self.version_2_with_negative = True
        self.null_score_diff_threshold = 0.0
        self.n_gpu = 0

        self.max_seq_length = 384
        self.doc_stride = 128
        self.max_query_length = 64

        self.do_train = True
        self.do_eval = True

        self.evaluate_during_training = False
        self.do_lower_case = True
        self.per_gpu_train_batch_size = 12
        self.per_gpu_eval_batch_size = 8
        self.learning_rate = 3e-5
        self.gradient_accumulation_steps = 1    # Number of updates steps to accumulate before performing a backward/update pass

        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.num_train_epochs = 1.0  # 训练的epoch数
        self.max_steps = -1  # 最多运行多少步，若设置>0, 将会覆盖num_train_epochs
        self.warmup_steps = 0
        self.n_best_size = 20
        self.max_answer_length = 30
        self.verbose_logging = False
        self.lang_id = 0
        self.logging_steps = 500  # 打log的步长
        self.save_steps = 2000  # 保存模型及其参数的步长

        self.eval_all_checkpoints = False
        self.no_cuda = True
        self.overwrite_cache = False        # 重写缓存文件

        self.seed = 42  # 随机种子
        self.local_rank = -1  # 分布式计算用到的进程编号，-1表示不使用分布式
        self.fp16 = False
        self.fp16_opt_level = "01"

        self.server_ip = ""
        self.server_port = ""
        self.threads = 1

        self.bert_dir = "../../Model_files/bert-base-uncased/"
        self.device = "cpu"
