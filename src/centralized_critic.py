# src/centralized_critic.py

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class CentralizedCritic(nn.Module):
    """
    The 'God's Eye' Value Network for MAPPO. 
    Observes the global state S and predicts a scalar Value V(S) to guide the actors.
    """
    def __init__(self, model_name_or_path, device="cuda"):
        super(CentralizedCritic, self).__init__()
        self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
        # Ensure padding token exists for batch processing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # num_labels=1 converts the transformer into a regression model outputting a scalar
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, 
            num_labels=1,           
            torch_dtype=torch.float16
        ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        self.loss_fn = nn.MSELoss()

    def forward(self, global_states_text):
        """
        Takes raw text representations of the global state S and returns V(S).
        global_states_text: list of strings (concat of prompt, payload, rules, and output).
        """
        inputs = self.tokenizer(
            global_states_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=2048
        ).to(self.device)
        
        # Forward pass to get logits (which act as our scalar value)
        outputs = self.model(**inputs)
        
        # Squeeze to get a 1D tensor of values: shape (batch_size,)
        values = outputs.logits.squeeze(-1)
        return values

    def update_critic(self, global_states, target_returns):
        """
        Updates the value network using Mean Squared Error against the GAE returns.
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Predict V(S)
        predicted_values = self.forward(global_states)
        
        # Ensure targets are on the correct device and match dtype
        target_returns = target_returns.to(self.device).to(predicted_values.dtype)
        
        # Calculate Value Loss
        loss = self.loss_fn(predicted_values, target_returns)
        
        # Backpropagation with gradient clipping to prevent explosion
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()