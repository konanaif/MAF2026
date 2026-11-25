"""Portable RoBERTa/LoRA adapters for the same binary toxicity task."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from .runtime import pad_seqs

from .constants import MODEL_IDS


class Encoded(Dataset):
    def __init__(self, rows, tokenizer, max_length, prompt=None):
        self.labels = torch.tensor([int(row['toxicity_label'] == 'toxic') for row in rows])
        self.pad_id = tokenizer.pad_token_id
        self.causal = prompt is not None
        self.items = []
        if self.causal:
            body = Path(prompt).read_text(encoding='utf-8').split('## System prompt\n', 1)[1]
            system, user = (part.strip() for part in body.split('## User template\n', 1))
            def render(text):
                return tokenizer.apply_chat_template([{'role': 'system', 'content': system}, {'role': 'user', 'content': user.format(text=text)}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            budget = max_length - len(tokenizer.encode(render(''), add_special_tokens=False)) - 8
            if budget < 32:
                raise ValueError('max_length leaves too little space for the transcript')
            self.truncated = 0
            for row in rows:
                text = row['text']
                tokens = tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > budget:
                    self.truncated += 1
                    text = tokenizer.decode(tokens[-budget:], skip_special_tokens=True)
                ids = tokenizer.encode(render(text), add_special_tokens=False)
                current_budget = budget
                while len(ids) > max_length and current_budget > 16:
                    current_budget -= min(8, len(ids) - max_length)
                    text = tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)[-current_budget:], skip_special_tokens=True)
                    ids = tokenizer.encode(render(text), add_special_tokens=False)
                if len(ids) > max_length:
                    raise ValueError('Cannot preserve the complete classification prompt')
                self.items.append({'input_ids': ids, 'attention_mask': [1] * len(ids)})
        else:
            encoded = tokenizer([r['text'] for r in rows], padding='max_length', max_length=max_length, truncation=True)
            # Sequence classifiers need only these inputs; no NER tag alignment is involved.
            self.items = [{k: encoded[k][i] for k in ('input_ids', 'attention_mask')} for i in range(len(rows))]
            self.truncated = sum(len(tokenizer.encode(row['text'])) > max_length for row in rows)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index], int(self.labels[index]), index

    def collate(self, items):
        length = max(len(item[0]['input_ids']) for item in items)
        result = {}
        for key, pad in (('input_ids', self.pad_id), ('attention_mask', 0)):
            if not self.causal:
                result[key] = pad_seqs([encoded[key] for encoded, _, _ in items], pad, max_length=length)
                continue
            values = []
            for encoded, _, _ in items:
                padding = [pad] * (length - len(encoded[key]))
                values.append(padding + encoded[key] if self.causal else encoded[key] + padding)
            result[key] = torch.tensor(values)
        result['labels'] = torch.tensor([y for _, y, _ in items])
        result['index'] = torch.tensor([i for _, _, i in items])
        return result


class Classifier:
    def __init__(self, model_name, cfg, device, checkpoint=None):
        self.causal = model_name == 'qwen3-4b'
        self.device = torch.device(device)
        self.cfg = cfg
        pretrained = cfg['model_path'] or MODEL_IDS[model_name]
        options = {'local_files_only': cfg['offline']}
        if cfg['revision']:
            options['revision'] = cfg['revision']
        tokenizer_source = Path(checkpoint) / 'tokenizer' if checkpoint else pretrained
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **({} if checkpoint else options))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.causal:
            if self.device.type != 'cuda':
                raise ValueError('Qwen training requires a CUDA device; use RoBERTa for CPU smoke checks')
            from peft import LoraConfig, PeftModel, get_peft_model
            self.tokenizer.padding_side = 'left'
            model = AutoModelForCausalLM.from_pretrained(pretrained, torch_dtype=torch.bfloat16, **options).to(self.device)
            model.config.use_cache = False
            if checkpoint:
                self.model = PeftModel.from_pretrained(model, Path(checkpoint) / 'adapter', is_trainable=False)
            else:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
                model.enable_input_require_grads()
                self.model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05, bias='none', task_type='CAUSAL_LM', target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
            ids = [self.tokenizer.encode(v, add_special_tokens=False) for v in ('0', '1')]
            if any(len(x) != 1 for x in ids):
                raise ValueError('0/1 must be single-token labels')
            self.label_ids = [ids[0][0], ids[1][0]]
        else:
            source = Path(checkpoint) / 'model' if checkpoint else pretrained
            self.model = AutoModelForSequenceClassification.from_pretrained(source, num_labels=2, id2label={0: 'nontoxic', 1: 'toxic'}, label2id={'nontoxic': 0, 'toxic': 1}, **({} if checkpoint else options)).to(self.device)
            if not checkpoint:
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    def encode(self, rows):
        return Encoded(rows, self.tokenizer, self.cfg['max_length'], self.cfg['prompt'] if self.causal else None)

    def forward(self, batch, features=False):
        inputs = {k: v.to(self.device) for k, v in batch.items() if k in ('input_ids', 'attention_mask')}
        if self.causal:
            output = self.model(**inputs, output_hidden_states=features, logits_to_keep=1)
            logits = output.logits[:, -1, self.label_ids].float()
            hidden = output.hidden_states[-1][:, -1, :] if features else None
        else:
            output = self.model(**inputs, output_hidden_states=features)
            logits = output.logits.float()
            hidden = output.hidden_states[-1][:, 0, :] if features else None
        return logits, hidden

    @torch.no_grad()
    def infer(self, dataset, features=False):
        self.model.eval()
        logits = np.empty((len(dataset), 2), np.float32)
        hidden = None
        for batch in DataLoader(dataset, batch_size=self.cfg['eval_batch_size'], shuffle=False, collate_fn=dataset.collate):
            values, representations = self.forward(batch, features)
            index = batch['index'].numpy()
            logits[index] = values.cpu().numpy()
            if features:
                reps = representations.float().cpu().numpy()
                if hidden is None:
                    hidden = np.empty((len(dataset), reps.shape[1]), np.float32)
                hidden[index] = reps
        return logits, hidden

    def trainable_state(self):
        return {n: p.detach().cpu().clone() for n, p in self.model.named_parameters() if p.requires_grad}

    def set_trainable_state(self, state):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in state:
                    param.copy_(state[name].to(param))

    def save(self, directory):
        directory = Path(directory)
        self.model.save_pretrained(directory / ('adapter' if self.causal else 'model'))
        self.tokenizer.save_pretrained(directory / 'tokenizer')
