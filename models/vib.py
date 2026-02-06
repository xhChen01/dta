import torch
import torch.nn as nn
import torch.nn.functional as F

class VIB(nn.Module):
    def __init__(self, **params):

        super().__init__()
        
        z_dim = params.get('z_dim', 64)
        max_smi_len = params.get('max_smi_len')
        max_seq_len = params.get('max_seq_len')
        charsmiset_size = params.get('charsmiset_size')
        charseqset_size = params.get('charseqset_size')
        num_filters = params.get('num_filters', 32)
        filter_length1 = params.get('filter_length1', 4)
        filter_length2 = params.get('filter_length2', 8)
        # 使用register_buffer确保这些张量与模型在同一设备上
        self.register_buffer('ligands_features', torch.tensor(params.get('ligands_features'), dtype=torch.int32, requires_grad=False))
        self.register_buffer('proteins_features', torch.tensor(params.get('proteins_features'), dtype=torch.int32, requires_grad=False))
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
        self.fc1 = nn.Linear(num_filters*3 * 2, 1024) 
        # ReLU activation
        self.relu = nn.ReLU()

        self.encoder = nn.Sequential(
            nn.Linear(num_filters*3 * 2, num_filters*3),
            nn.ReLU(),
            nn.Linear(num_filters*3, 2 * z_dim)  # 输出均值 μ 和对数方差 logσ²
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, 1)
        )
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
    
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, drug_ids, target_ids):
        # 根据输入的drug_input编号将drug映射到ligands_features上
        drug = self.ligands_features[drug_ids]  # (batch, channels, length)
        protein = self.proteins_features[target_ids]  # (batch, channels, length)
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
        x = torch.cat((drug, protein), dim=1)  # (batch, filters*3*2)
        
        mu_logvar = self.encoder(x)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)  # T ~ q(t|x)
        score = self.decoder(z).squeeze(-1)
        return score, mu, logvar
    