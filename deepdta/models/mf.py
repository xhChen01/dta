import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
  def __init__(self, n_drugs, n_targets, n_factors=20):
    super().__init__()
    self.model_name = "matrix factorization"
    self.drug_embeddings = nn.Embedding(n_drugs, n_factors)
    self.target_embeddings = nn.Embedding(n_targets, n_factors)
    self.drug_bias = nn.Embedding(n_drugs, 1)
    self.target_bias = nn.Embedding(n_targets, 1)
    
    nn.init.normal_(self.drug_embeddings.weight, std=0.01)
    nn.init.normal_(self.target_embeddings.weight, std=0.01)
    nn.init.zeros_(self.drug_bias.weight)
    nn.init.zeros_(self.target_bias.weight)
      
  def forward(self, drug_ids, target_ids):
    drug_emb = self.drug_embeddings(drug_ids)
    target_emb = self.target_embeddings(target_ids)
    drug_b = self.drug_bias(drug_ids).squeeze()
    target_b = self.target_bias(target_ids).squeeze()
    
    interaction = torch.sum(drug_emb * target_emb, dim=1)
    return interaction + drug_b + target_b