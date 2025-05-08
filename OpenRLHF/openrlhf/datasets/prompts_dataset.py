import traceback
from torch.utils.data import Dataset
from tqdm import tqdm


def preprocess_data(
        data, 
        tokenizer,
        max_prompt_len,
        input_template=None, 
        input_key="input", 
        meta_keys="", 
        apply_chat_template=None,
    ) -> str:
    try:
        if apply_chat_template:
            chat = data[input_key]
            if isinstance(chat, str):
                chat = [{"role": "user", "content": chat}]
            prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        else:
            prompt = data[input_key]
            if input_template:
                prompt = input_template.format(prompt)

        prompt_token_len = len(tokenizer(prompt)['input_ids'])
        if prompt_token_len > max_prompt_len:
            return None, f"prompt length `{prompt_token_len}` is larger than max_prompt_len `{max_prompt_len}`."
        
        sample = {"prompt": prompt}
        meta_keys = [k.strip() for k in meta_keys.split(',') if k.strip()]

        for meta_key in meta_keys:
            try:
                sample[meta_key] = data[meta_key]
            except:
                sample[meta_key] = ""
        
        return sample, ""
    except:
        return None, traceback.format_exc()


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        meta_keys = getattr(self.strategy.args, "meta_keys", None)
        max_prompt_len = getattr(self.strategy.args, "prompt_max_len", 1024)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt, exception_str = preprocess_data(
                data, 
                tokenizer, 
                max_prompt_len, 
                input_template, 
                input_key, 
                meta_keys, 
                apply_chat_template
            )
            if prompt:
                self.prompts.append(prompt)
            else:
                print(f'[Prepare Dataset Warning] One data dropped caused by: {exception_str}.')

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]
