import torch
import torch.nn as nn
from torch.nn.functional import dropout

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepDTA(nn.Module):
    
    def __init__(self,ligands_features, proteins_features, max_smi_len, max_seq_len, charsmiset_size, charseqset_size,
                 num_filters=32, filter_length1=4, filter_length2=8):
        """
        Args:
            max_smi_len (int): Maximum length of SMILES strings
            max_seq_len (int): Maximum length of protein sequences
            charsmiset_size (int): Size of SMILES character set (for one-hot encoding)
            charseqset_size (int): Size of protein sequence character set (for one-hot encoding)
            num_filters (int): Number of filters in the first convolutional layer
            filter_length1 (int): Kernel size for drug SMILES convolution
            filter_length2 (int): Kernel size for protein sequence convolution
        """
        super(DeepDTA, self).__init__()
        # 使用register_buffer确保这些张量与模型在同一设备上
        self.register_buffer('ligands_features', torch.tensor(ligands_features, dtype=torch.int32, requires_grad=False))
        self.register_buffer('proteins_features', torch.tensor(proteins_features, dtype=torch.int32, requires_grad=False))
        self.drug_embeddings = nn.Embedding(max_smi_len, charsmiset_size)
        self.target_embeddings = nn.Embedding(max_seq_len, charseqset_size)
    
        # Drug (SMILES) processing pathway
        self.drug_conv1 = nn.Conv1d(in_channels=charsmiset_size, out_channels=num_filters,
                                  kernel_size=filter_length1, padding=0, stride=1)
        self.drug_conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters*2,
                                  kernel_size=filter_length1, padding=0, stride=1)
        self.drug_conv3 = nn.Conv1d(in_channels=num_filters*2, out_channels=num_filters*3,
                                  kernel_size=filter_length1, padding=0, stride=1)
        
        # Protein sequence processing pathway
        self.protein_conv1 = nn.Conv1d(in_channels=charseqset_size, out_channels=num_filters,
                                     kernel_size=filter_length2, padding=0, stride=1)
        self.protein_conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters*2,
                                     kernel_size=filter_length2, padding=0, stride=1)
        self.protein_conv3 = nn.Conv1d(in_channels=num_filters*2, out_channels=num_filters*3,
                                     kernel_size=filter_length2, padding=0, stride=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(num_filters*3 * 2, 1024)  # *2 because we concatenate drug and protein features
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 1)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        
        # ReLU activation
        self.relu = nn.ReLU()
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization for linear layers"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, drug_input, protein_input):
      
        # Note: PyTorch Conv1d expects input shape (batch_size, channels, length)
        # while Keras Conv1D expects (batch_size, length, channels)
        # So we need to permute the dimensions


        
        # 根据输入的drug_input编号将drug映射到ligands_features上
        drug = self.ligands_features[drug_input]  # (batch, channels, length)
        protein = self.proteins_features[protein_input]  # (batch, channels, length)
        # 将drug和protein的维度增加一维
        drug = self.drug_embeddings(drug)  # (batch, 1, channels, length)
        protein = self.target_embeddings(protein)  # (batch, 1, channels, length)
        drug = drug.permute(0, 2, 1)  # (batch, channels, length)
        protein = protein.permute(0, 2, 1)  # (batch, channels, length)
    
        # Process drug SMILES
        drug = self.relu(self.drug_conv1(drug))
        drug = self.relu(self.drug_conv2(drug))
        drug = self.relu(self.drug_conv3(drug))
        # Global Max Pooling
        drug = torch.max(drug, dim=2)[0]  # (batch, filters*3)
        
        # Process protein sequence
        protein = self.relu(self.protein_conv1(protein))
        protein = self.relu(self.protein_conv2(protein))
        protein = self.relu(self.protein_conv3(protein))
        # Global Max Pooling
        protein = torch.max(protein, dim=2)[0]  # (batch, filters*3)

        ## 通过维度拼接的方式来连接多个FC层的方式来预测
        
        # Concatenate drug and protein features
        combined = torch.cat((drug, protein), dim=1)  # (batch, filters*3*2)
        
        # Fully connected layers with dropout
        x = self.relu(self.fc1(combined))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        
        # # Output layer (no activation for regression)
        output = self.fc4(x)
        output = output.squeeze(dim=1)
        
        return output