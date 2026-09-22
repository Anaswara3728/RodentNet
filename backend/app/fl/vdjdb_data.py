import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset

class AATokenizer:
    """Tokenizes Amino Acid sequences into padded integer tensors."""
    def __init__(self, max_len: int = 25):
        self.max_len = max_len
        self.aa_list = list("ACDEFGHIKLMNPQRSTVWY")
        
        self.PAD_IDX = 0
        self.UNK_IDX = len(self.aa_list) + 1
        
        self.aa2idx = {aa: i + 1 for i, aa in enumerate(self.aa_list)}
        self.vocab_size = len(self.aa_list) + 2

    def encode(self, seq: str) -> torch.Tensor:
        if not isinstance(seq, str):
            seq = ""
        indices = [self.aa2idx.get(aa.upper(), self.UNK_IDX) for aa in seq]
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        pad_len = self.max_len - len(indices)
        indices.extend([self.PAD_IDX] * pad_len)
        return torch.tensor(indices, dtype=torch.long)

class VDJdbDataset(Dataset):
    """PyTorch Dataset for real VDjDB TCR-Epitope pairs or synthetic planted data."""
    def __init__(self, tsv_path: str = None, synthetic: bool = False, max_tcr_len: int = 25, max_epi_len: int = 20, synthetic_size: int = 2000):
        self.tcr_tokenizer = AATokenizer(max_len=max_tcr_len)
        self.epi_tokenizer = AATokenizer(max_len=max_epi_len)
        
        if synthetic:
            self._generate_synthetic(synthetic_size)
        else:
            self._load_real(tsv_path)
            
    def _load_real(self, tsv_path: str):
        if not tsv_path or not os.path.exists(tsv_path):
            raise FileNotFoundError(f"VDjDB TSV not found at {tsv_path}")
            
        df = pd.read_csv(tsv_path, sep='\t', low_memory=False)
        if 'Species' in df.columns:
            df = df[df['Species'] == 'HomoSapiens'].copy()
            
        df = df.dropna(subset=['CDR3', 'Epitope'])
        self.data = df.reset_index(drop=True)
        
        self.unique_epitopes = self.data['Epitope'].unique()
        self.epi2idx = {epi: idx for idx, epi in enumerate(self.unique_epitopes)}
        
    def _generate_synthetic(self, size: int):
        """Generates synthetic data with a planted TCR<->Epitope motif rule for offline smoke tests."""
        # 3 artificial epitopes
        epitopes = ["GILGFVFTL", "NLVPMVATV", "GLCTLVAML"]
        self.unique_epitopes = np.array(epitopes)
        self.epi2idx = {epi: idx for idx, epi in enumerate(epitopes)}
        
        records = []
        for _ in range(size):
            epi = np.random.choice(epitopes)
            # Planted rule: Epitope determines the first 3 AAs of the TCR
            if epi == "GILGFVFTL":
                tcr = "CAS" + "".join(np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"), 10))
            elif epi == "NLVPMVATV":
                tcr = "CASS" + "".join(np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"), 9))
            else:
                tcr = "CSV" + "".join(np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"), 10))
                
            records.append({"CDR3": tcr, "Epitope": epi})
            
        self.data = pd.DataFrame(records)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tcr_seq = row['CDR3']
        epi_seq = row['Epitope']
        epi_id = self.epi2idx[epi_seq]
        
        tcr_tensor = self.tcr_tokenizer.encode(tcr_seq)
        epi_tensor = self.epi_tokenizer.encode(epi_seq)
        
        return tcr_tensor, epi_tensor, epi_id

def load_vdjdb_partition(num_clients: int, client_id: int, tsv_path: str = None, synthetic: bool = False):
    """
    Returns non-IID partitioned train/eval DataLoaders.
    Partitioning is done by Epitope groups so clients hold different label distributions.
    """
    dataset = VDJdbDataset(tsv_path=tsv_path, synthetic=synthetic)
    
    # Non-IID Partitioning: Group indices by epitope
    epitope_groups = {}
    for i in range(len(dataset)):
        epi_id = dataset.data.iloc[i]['Epitope']
        if epi_id not in epitope_groups:
            epitope_groups[epi_id] = []
        epitope_groups[epi_id].append(i)
        
    # Assign specific epitopes to specific clients to create severe non-IID distribution
    client_indices = []
    unique_epis = list(epitope_groups.keys())
    
    # Assign epitopes round-robin to clients to ensure they get completely different panels
    for i, epi in enumerate(unique_epis):
        if i % num_clients == client_id:
            client_indices.extend(epitope_groups[epi])
            
    # Fallback if a client gets nothing (e.g. fewer epitopes than clients)
    if not client_indices:
        client_indices = list(range(len(dataset)))
        
    # 80/20 train/test split for this client's local data
    np.random.seed(42 + client_id)
    np.random.shuffle(client_indices)
    split_idx = int(0.8 * len(client_indices))
    
    train_idx = client_indices[:split_idx]
    test_idx = client_indices[split_idx:]
    
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=32, shuffle=False)
    
    # We also return the catalogue of all unique epitope tokens (needed for evaluation against unseen/all epitopes)
    all_epi_tokens = torch.stack([dataset.epi_tokenizer.encode(epi) for epi in dataset.unique_epitopes])
    
    return train_loader, test_loader, all_epi_tokens
