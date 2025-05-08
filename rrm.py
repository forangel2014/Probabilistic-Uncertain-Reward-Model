import datasets
import transformers
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    BitsAndBytesConfig,
    is_torch_tpu_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig
from trl.trainer.utils import RewardDataCollatorWithPadding
import torch
from itertools import combinations  # 添加这个import
from accelerate import DistributedDataParallelKwargs
from dataclasses import dataclass, field
import os
import sys
from typing import Optional, List
import logging
from transformers import DataCollatorWithPadding
from typing import Any, Dict, List, Optional, Union
from transformers.utils import PaddingStrategy
import numpy as np
MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 分别处理 chosen 和 rejected 的特征
        chosen_features = []
        rejected_features = []
        for feature in features:
            # check if the keys are named as expected
            if (
                "input_ids_chosen" not in feature
                or "input_ids_rejected" not in feature
                or "attention_mask_chosen" not in feature
                or "attention_mask_rejected" not in feature
            ):
                raise ValueError(
                    "The features should include `input_ids_chosen`, `attention_mask_chosen`, `input_ids_rejected` and `attention_mask_rejected`"
                )

            chosen_features.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            rejected_features.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )

        # 分别对 chosen 和 rejected 进行填充
        chosen_batch = self.tokenizer.pad(
            chosen_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        rejected_batch = self.tokenizer.pad(
            rejected_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # 合并所有特征
        batch = {
            "input_ids_chosen": chosen_batch["input_ids"],
            "attention_mask_chosen": chosen_batch["attention_mask"],
            "input_ids_rejected": rejected_batch["input_ids"],
            "attention_mask_rejected": rejected_batch["attention_mask"],
            #"chosen_score": torch.tensor([feature["chosen_score"] for feature in features]),
            #"rejected_score": torch.tensor([feature["rejected_score"] for feature in features]),
            "return_loss": True,
        }
        
        return batch

# Define the trainer
def compute_metrics(eval_pred):
    result = {}
    pos_predictions_scores = eval_pred.predictions[0]
    neg_predictions_scores = eval_pred.predictions[1]
    # We assume that the first sample is preferred by default in groundtruth
    result['accuracy'] = np.sum(
        pos_predictions_scores > neg_predictions_scores) / len(pos_predictions_scores)
    return result

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    lora_r: Optional[int] = field(default=16)
    lora_alpha: Optional[int] = field(default=32)
    target_modules: Optional[str] = field(
        default='q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj',
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    load_in_bits: Optional[int] = field(default=8)
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )
        if type(self.target_modules)==str:
            self.target_modules = self.target_modules.split(',')

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_on_inputs: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    # train_files: Optional[List[str]]  = field(default=None, metadata={"help": "The input training data file (a text file)."})
    # validation_files: Optional[List[str]]  = field(
    #     default=None,
    #     metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    # )
    train_file_path: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    valid_file_path: Optional[str] = field(default=None, metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")
                
class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.is_world_process_zero:
            print('+++++++++++++++++save call back++++++++++++++++')
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )
            kwargs["model"].save_pretrained(checkpoint_folder)

            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
            return control

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # pdb.set_trace()
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # # 设置随机种子以保证每次启动时训练数据的顺序相同
    # set_seed(training_args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    train_dataset = datasets.load_from_disk(data_args.train_file_path)
    eval_dataset = datasets.load_from_disk(data_args.valid_file_path)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, padding=True, truncation=True, max_length=1024)

    # 设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 添加数据增强逻辑，为前2000步的样本构造额外样本
    if training_args.do_train:
        # 限制只处理前2000个样本
        sample_count = min(2000, len(train_dataset))
        original_samples = train_dataset.select(range(sample_count))
        
        # 创建新的训练样本列表
        new_samples = []
        
        # 对每个样本进行处理
        for i in range(sample_count):
            current_sample = original_samples[i]
            
            # 为当前样本寻找不同的样本进行混合
            for j in range(sample_count):
                if i == j:
                    continue
                    
                other_sample = original_samples[j]
                
                # 构造新样本1: (x_i, y_w_i, y_l_j) - 用自己的好回答和其他样本的坏回答
                # 找到Question部分的分界点（通过寻找"Answer:"标记）
                chosen_text = tokenizer.decode(current_sample["input_ids_chosen"])
                answer_pos = chosen_text.find("\nAnswer:")
                if answer_pos != -1:
                    # 构造新的rejected样本，用当前样本的问题加上其他样本的坏回答
                    other_rejected_text = tokenizer.decode(other_sample["input_ids_rejected"])
                    other_answer_pos = other_rejected_text.find("\nAnswer:")
                    if other_answer_pos != -1:
                        other_answer = other_rejected_text[other_answer_pos:]
                        question_part = chosen_text[:answer_pos]
                        new_rejected_text = question_part + other_answer
                        
                        # 对新构造的文本进行tokenize
                        new_rejected_tokens = tokenizer.encode(new_rejected_text, truncation=True)
                        
                        # 创建新样本
                        new_sample = current_sample.copy()
                        new_sample["input_ids_rejected"] = new_rejected_tokens
                        new_sample["attention_mask_rejected"] = [1] * len(new_rejected_tokens)
                        new_samples.append(new_sample)
                
                # 构造新样本2: (x_i, y_l_i, y_w_j) - 用自己的坏回答和其他样本的好回答
                rejected_text = tokenizer.decode(current_sample["input_ids_rejected"])
                answer_pos = rejected_text.find("\nAnswer:")
                if answer_pos != -1:
                    # 构造新的chosen样本，用当前样本的问题加上其他样本的好回答
                    other_chosen_text = tokenizer.decode(other_sample["input_ids_chosen"])
                    other_answer_pos = other_chosen_text.find("\nAnswer:")
                    if other_answer_pos != -1:
                        other_answer = other_chosen_text[other_answer_pos:]
                        question_part = rejected_text[:answer_pos]
                        new_chosen_text = question_part + other_answer
                        
                        # 对新构造的文本进行tokenize
                        new_chosen_tokens = tokenizer.encode(new_chosen_text, truncation=True)
                        
                        # 创建新样本
                        new_sample = current_sample.copy()
                        new_sample["input_ids_chosen"] = new_chosen_tokens
                        new_sample["attention_mask_chosen"] = [1] * len(new_chosen_tokens)
                        new_samples.append(new_sample)
                
                # 每个原始样本只需要构造两个新样本，所以只处理一次循环
                break
        
        # 将新样本添加到训练数据集的最开头
        if new_samples:
            # 将样本列表转换为特征字典
            features_dict = {}
            for key in new_samples[0].keys():
                features_dict[key] = [sample[key] for sample in new_samples]
            
            # 创建新的数据集并将原始数据集连接到后面（新样本在前）
            new_dataset = datasets.Dataset.from_dict(features_dict)
            # 修改这一行，将顺序调整为先新样本后原始样本
            augmented_dataset = datasets.concatenate_datasets([new_dataset, train_dataset])
            
            # 更新训练数据集
            train_dataset = augmented_dataset
            print(f"原始训练样本数量: {len(datasets.load_from_disk(data_args.train_file_path))}")
            print(f"增强后的训练样本数量: {len(train_dataset)}")
            print(f"增强样本已添加到训练数据集最开头")
        else:
            print("警告：没有生成任何新样本进行数据增强")

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    model = AutoModelForSequenceClassification.from_pretrained(            
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            #config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            #torch_dtype=torch_dtype,
            load_in_8bit=True if model_args.load_in_bits==8 else False,
            #quantization_config=bnb_config if model_args.load_in_bits==4 else None,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)}, 
            num_labels=1
        )
    
    # for param in model.model.parameters():
    #     param.requires_grad = False

    # 确保模型知道padding token
    model.config.pad_token_id = tokenizer.pad_token_id

    # 基于training_args创建 RewardConfig
    reward_config = RewardConfig(**training_args.to_dict())
    reward_config.remove_unused_columns = False

    # # 添加数据加载配置
    # reward_config.dataloader_drop_last = False  # 确保不丢弃最后不完整的batch
    # reward_config.dataloader_shuffle_train = False  # 关闭训练集shuffle
    # reward_config.dataloader_shuffle_eval = False   # 关闭验证集shuffle

    data_collator = RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=1024)

    trainer = RewardTrainer(
        model=model,
        args=reward_config,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 开始训练
    trainer.train()

    # 保存模型
    trainer.save_model()


if __name__ == "__main__":
    
    logger = logging.getLogger(__name__)

    main()
