from torch.utils.data import Dataset, DataLoader
import torch


class AffinityDataset(Dataset):
    def __init__(self, drugs, targets, affinities):
        self.drugs = torch.LongTensor(drugs)
        self.targets = torch.LongTensor(targets)
        self.affinities = torch.FloatTensor(affinities)
        
    def __len__(self):
        return len(self.drugs)
    
    def __getitem__(self, idx):
        return self.drugs[idx], self.targets[idx], self.affinities[idx]