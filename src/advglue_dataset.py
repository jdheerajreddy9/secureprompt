import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
from dataclasses import dataclass
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from collections import defaultdict

@dataclass
class AdvGLUEExample:
    idx: int
    label: int
    original_text: str
    adversarial_text: str
    task_name: str
    method: str

class AdvGLUEDataset(Dataset):
    def __init__(self, examples: List[AdvGLUEExample], tokenizer, max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize both original and adversarial texts
        original_encoding = self.tokenizer(
            example.original_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        adversarial_encoding = self.tokenizer(
            example.adversarial_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'original_input_ids': original_encoding['input_ids'].squeeze(),
            'original_attention_mask': original_encoding['attention_mask'].squeeze(),
            'adversarial_input_ids': adversarial_encoding['input_ids'].squeeze(),
            'adversarial_attention_mask': adversarial_encoding['attention_mask'].squeeze(),
            'label': torch.tensor(example.label, dtype=torch.long),
            'task': example.task_name,
            'method': example.method
        }

class AdvGLUEHandler:
    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
    def load_dataset(self) -> Dict[str, List[AdvGLUEExample]]:
        """Load and parse AdvGLUE dataset from JSON file"""
        try:
            dev_file = self.data_dir / "dev.json"
            if not dev_file.exists():
                raise FileNotFoundError(f"AdvGLUE dataset not found at {dev_file}")

            with open(dev_file, 'r') as f:
                data = json.load(f)

            examples = {
                'sst2': self._parse_sst2(data.get('sst2', [])),
                'qqp': self._parse_qqp(data.get('qqp', [])),
                'mnli': self._parse_mnli(data.get('mnli', [])),
                'mnli-mm': self._parse_mnli(data.get('mnli-mm', []), task_name='mnli-mm'),
                'qnli': self._parse_qnli(data.get('qnli', [])),
                'rte': self._parse_rte(data.get('rte', []))
            }

            self.logger.info(f"Loaded {sum(len(v) for v in examples.values())} examples from AdvGLUE dataset")
            return examples

        except Exception as e:
            self.logger.error(f"Error loading AdvGLUE dataset: {str(e)}")
            raise

    def _parse_sst2(self, items: List[Dict]) -> List[AdvGLUEExample]:
        """Parse SST-2 (sentiment analysis) examples"""
        examples = []
        for item in items:
            if all(k in item for k in ['idx', 'label', 'sentence']):
                examples.append(AdvGLUEExample(
                    idx=item['idx'],
                    label=int(item['label']),
                    original_text=item.get('original_sentence', item['sentence']),
                    adversarial_text=item['sentence'],
                    task_name='sst2',
                    method=item.get('method', 'unknown')
                ))
        return examples

    def _parse_qqp(self, items: List[Dict]) -> List[AdvGLUEExample]:
        """Parse QQP (question pair) examples"""
        examples = []
        for item in items:
            if all(k in item for k in ['idx', 'label', 'question1', 'question2']):
                examples.append(AdvGLUEExample(
                    idx=item['idx'],
                    label=int(item['label']),
                    original_text=f"{item['question1']} [SEP] {item.get('original_question2', item['question2'])}",
                    adversarial_text=f"{item['question1']} [SEP] {item['question2']}",
                    task_name='qqp',
                    method=item.get('method', 'unknown')
                ))
        return examples

    def _parse_mnli(self, items: List[Dict], task_name: str = 'mnli') -> List[AdvGLUEExample]:
        """Parse MNLI (natural language inference) examples"""
        examples = []
        for item in items:
            if all(k in item for k in ['idx', 'label', 'premise', 'hypothesis']):
                examples.append(AdvGLUEExample(
                    idx=item['idx'],
                    label=int(item['label']),
                    original_text=f"{item['premise']} [SEP] {item.get('original_hypothesis', item['hypothesis'])}",
                    adversarial_text=f"{item['premise']} [SEP] {item['hypothesis']}",
                    task_name=task_name,
                    method=item.get('method', 'unknown')
                ))
        return examples

    def _parse_qnli(self, items: List[Dict]) -> List[AdvGLUEExample]:
        """Parse QNLI (question-answering NLI) examples"""
        examples = []
        for item in items:
            if all(k in item for k in ['idx', 'label', 'question', 'sentence']):
                examples.append(AdvGLUEExample(
                    idx=item['idx'],
                    label=int(item['label']),
                    original_text=f"{item['question']} [SEP] {item.get('original_sentence', item['sentence'])}",
                    adversarial_text=f"{item['question']} [SEP] {item['sentence']}",
                    task_name='qnli',
                    method=item.get('method', 'unknown')
                ))
        return examples

    def _parse_rte(self, items: List[Dict]) -> List[AdvGLUEExample]:
        """Parse RTE (recognizing textual entailment) examples"""
        examples = []
        for item in items:
            if all(k in item for k in ['idx', 'label', 'sentence1', 'sentence2']):
                examples.append(AdvGLUEExample(
                    idx=item['idx'],
                    label=int(item['label']),
                    original_text=f"{item['sentence1']} [SEP] {item.get('original_sentence2', item['sentence2'])}",
                    adversarial_text=f"{item['sentence1']} [SEP] {item['sentence2']}",
                    task_name='rte',
                    method=item.get('method', 'unknown')
                ))
        return examples

    def prepare_data(self, examples: Dict[str, List[AdvGLUEExample]]) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare training, validation, and test datasets"""
        # Combine all examples
        all_examples = []
        for task_examples in examples.values():
            all_examples.extend(task_examples)

        # Split into train, validation, and test sets
        train_examples, temp_examples = train_test_split(all_examples, test_size=0.3, random_state=42)
        val_examples, test_examples = train_test_split(temp_examples, test_size=0.5, random_state=42)

        # Create datasets
        train_dataset = AdvGLUEDataset(train_examples, self.tokenizer)
        val_dataset = AdvGLUEDataset(val_examples, self.tokenizer)
        test_dataset = AdvGLUEDataset(test_examples, self.tokenizer)

        # Create dataloaders with smaller batch size
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8)
        test_loader = DataLoader(test_dataset, batch_size=8)

        return train_loader, val_loader, test_loader
    

    def get_attack_statistics(self, examples: Dict[str, List[AdvGLUEExample]]) -> Dict[str, Any]:
        """
        Calculate statistics about the attack patterns in the dataset
        
        Args:
            examples: Dictionary of task name to list of examples
            
        Returns:
            Dictionary containing statistics about the dataset
        """
        stats = {
            'total_examples': 0,
            'tasks': {},
            'methods': defaultdict(int),
            'labels': defaultdict(int)
        }
        
        for task_name, task_examples in examples.items():
            # Count examples per task
            stats['tasks'][task_name] = len(task_examples)
            stats['total_examples'] += len(task_examples)
            
            # Count methods and labels
            for example in task_examples:
                stats['methods'][example.method] += 1
                stats['labels'][example.label] += 1
        
        return dict(stats)