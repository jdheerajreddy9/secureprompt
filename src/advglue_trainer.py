import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import logging
from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

class AdvGLUETrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.logger = logging.getLogger(__name__)
        
    def _convert_labels(self, batch_labels):
        """Convert and validate labels for BERT input"""
        labels = batch_labels.clone()
        # Map labels to [0, num_labels - 1] range
        unique_labels = torch.unique(labels)
        for idx, label in enumerate(unique_labels):
            labels[batch_labels == label] = idx
        return labels

    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        warmup_steps: int = 0,
        num_labels: int = 3  # Set to maximum number of labels across tasks
    ):
        """Train the model on AdvGLUE dataset"""
        try:
            optimizer = AdamW(self.model.parameters(), lr=learning_rate)
            total_steps = len(train_loader) * epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )

            best_val_loss = float('inf')
            for epoch in range(epochs):
                self.logger.info(f"Starting epoch {epoch + 1}/{epochs}")
                
                # Training
                self.model.train()
                train_loss = 0
                train_preds, train_labels = [], []
                
                for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")):
                    try:
                        # Move batch to device and process labels
                        inputs = {
                            'input_ids': batch['adversarial_input_ids'].to(self.device),
                            'attention_mask': batch['adversarial_attention_mask'].to(self.device),
                        }
                        
                        # Process labels
                        labels = batch['label'].to(self.device)
                        labels = self._convert_labels(labels)
                        
                        # Verify labels are in correct range
                        if labels.max() >= num_labels or labels.min() < 0:
                            self.logger.warning(f"Batch {batch_idx}: Invalid labels found: min={labels.min()}, max={labels.max()}")
                            continue
                        
                        optimizer.zero_grad()
                        outputs = self.model(**inputs, labels=labels)
                        
                        loss = outputs.loss
                        train_loss += loss.item()
                        
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        
                        preds = torch.argmax(outputs.logits, dim=-1)
                        train_preds.extend(preds.cpu().numpy())
                        train_labels.extend(labels.cpu().numpy())
                        
                    except Exception as e:
                        self.logger.error(f"Error in batch {batch_idx}: {str(e)}")
                        continue

                # Validation
                val_loss, val_metrics = self.evaluate(val_loader, num_labels)
                
                # Log metrics
                train_metrics = self._compute_metrics(train_preds, train_labels)
                self.logger.info(
                    f"Epoch {epoch + 1}/{epochs}:\n"
                    f"Train Loss: {train_loss/len(train_loader):.4f}\n"
                    f"Val Loss: {val_loss:.4f}\n"
                    f"Train Metrics: {train_metrics}\n"
                    f"Val Metrics: {val_metrics}"
                )
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), 'best_model.pt')
                    
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            raise

    def evaluate(self, dataloader, num_labels: int = 3):
        """Evaluate the model on a dataset"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                try:
                    inputs = {
                        'input_ids': batch['adversarial_input_ids'].to(self.device),
                        'attention_mask': batch['adversarial_attention_mask'].to(self.device),
                    }
                    
                    # Process labels
                    labels = batch['label'].to(self.device)
                    labels = self._convert_labels(labels)
                    
                    if labels.max() >= num_labels or labels.min() < 0:
                        continue
                    
                    outputs = self.model(**inputs, labels=labels)
                    total_loss += outputs.loss.item()
                    
                    preds = torch.argmax(outputs.logits, dim=-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                except Exception as e:
                    self.logger.error(f"Error in evaluation batch {batch_idx}: {str(e)}")
                    continue

        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else float('inf')
        metrics = self._compute_metrics(all_preds, all_labels)
        
        return avg_loss, metrics

    def _compute_metrics(self, preds, labels):
        """Compute evaluation metrics"""
        try:
            if len(preds) == 0 or len(labels) == 0:
                return {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0
                }
                
            accuracy = accuracy_score(labels, preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds, average='weighted', zero_division=0
            )
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        except Exception as e:
            self.logger.error(f"Error computing metrics: {str(e)}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }