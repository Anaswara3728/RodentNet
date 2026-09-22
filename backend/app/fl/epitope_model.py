import torch
import torch.nn as nn
import torch.nn.functional as F

class RBFKANLayer(nn.Module):
    """Pure-torch, fixed-grid RBF layer for MPC compatibility.
    Grid centers are a FIXED BUFFER (not learnable) -> MPC additive safe.
    Only linear mixing weights are learnable.
    """
    def __init__(self, in_features, out_features, grid_size=8, grid_range=[-2.0, 2.0]):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        
        # Fixed buffer for RBF centers (not learnable -> MPC additive safe)
        grid = torch.linspace(grid_range[0], grid_range[1], grid_size)
        self.register_buffer("grid", grid)
        
        # Base linear path (SiLU)
        self.base_weight = nn.Linear(in_features, out_features)
        
        # RBF mixing path
        self.rbf_weight = nn.Linear(grid_size * in_features, out_features, bias=False)
        
        # Variance for RBF
        self.register_buffer("sigma", torch.tensor((grid_range[1] - grid_range[0]) / (grid_size - 1)))

    def forward(self, x):
        # Base path
        base_out = self.base_weight(F.silu(x))
        
        # RBF path
        # x shape: (B, in_features)
        # grid shape: (grid_size,)
        # Reshape for broadcasting: x -> (B, in_features, 1), grid -> (1, 1, grid_size)
        x_expanded = x.unsqueeze(-1)
        grid_expanded = self.grid.view(1, 1, -1)
        
        # RBF evaluation: exp(-(x - c)^2 / (2 * sigma^2))
        rbf_eval = torch.exp(-((x_expanded - grid_expanded) ** 2) / (2 * self.sigma ** 2))
        
        # Flatten RBF output: (B, in_features * grid_size)
        rbf_flat = rbf_eval.view(x.size(0), -1)
        rbf_out = self.rbf_weight(rbf_flat)
        
        return base_out + rbf_out

class FastKAN(nn.Module):
    """Stack of RBFKANLayer blocks serving as the interaction metric."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.layer1 = RBFKANLayer(in_dim, hidden_dim)
        self.layer2 = RBFKANLayer(hidden_dim, out_dim)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class ResidueMotifEncoder(nn.Module):
    """Shared encoder for both TCR CDR3 and Epitope sequences."""
    def __init__(self, vocab_size=25, embed_dim=64, channels=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.conv1 = nn.Conv1d(embed_dim, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        
    def forward(self, x):
        # x: (B, L)
        x = self.embedding(x) # (B, L, E)
        x = x.transpose(1, 2) # (B, E, L) for Conv1d
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        x = self.pool(x).squeeze(-1) # (B, channels)
        return x

class PersonalCalibration(nn.Module):
    """Learnable temperature scalar (local per-client)."""
    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))
        
    def forward(self, logits):
        temp = torch.exp(self.log_temperature).clamp(min=0.01, max=100.0)
        return logits / temp

class SiameseEpitopeKANNet(nn.Module):
    def __init__(self, vocab_size=25, embed_dim=64, conv_channels=128, proj_dim=64):
        super().__init__()
        
        # SHARED — federated (Matches existing SHARED_PREFIXES = "features.", "projection.")
        self.features = nn.ModuleDict({
            'tcr': ResidueMotifEncoder(vocab_size, embed_dim, conv_channels),
            'epitope': ResidueMotifEncoder(vocab_size, embed_dim, conv_channels)
        })
        
        self.projection = nn.ModuleDict({
            'tcr': nn.Sequential(
                nn.Linear(conv_channels, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, proj_dim)
            ),
            'epitope': nn.Sequential(
                nn.Linear(conv_channels, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, proj_dim)
            ),
            'kan': FastKAN(proj_dim * 2, proj_dim, 1) # Takes concatenated (TCR, Epi) -> Score
        })
        
        # LOCAL — never federated
        self.personal_head = PersonalCalibration()
        
    def encode_tcr(self, tokens):
        feat = self.features['tcr'](tokens)
        proj = self.projection['tcr'](feat)
        return F.normalize(proj, p=2, dim=1)
        
    def encode_epitope(self, tokens):
        feat = self.features['epitope'](tokens)
        proj = self.projection['epitope'](feat)
        return F.normalize(proj, p=2, dim=1)
        
    def score(self, tcr_emb, epi_emb):
        # Concatenate for FastKAN
        x = torch.cat([tcr_emb, epi_emb], dim=-1)
        raw_score = self.projection['kan'](x).squeeze(-1)
        return self.personal_head(raw_score)
        
    def pairwise_logits(self, tcr_batch, epi_batch):
        """Computes BxB interaction matrix."""
        B_tcr = tcr_batch.size(0)
        B_epi = epi_batch.size(0)
        
        # Expand for pairwise matching
        tcr_exp = tcr_batch.unsqueeze(1).expand(B_tcr, B_epi, -1).reshape(-1, tcr_batch.size(1))
        epi_exp = epi_batch.unsqueeze(0).expand(B_tcr, B_epi, -1).reshape(-1, epi_batch.size(1))
        
        scores = self.score(tcr_exp, epi_exp)
        return scores.view(B_tcr, B_epi)

    def forward(self, tcr_tokens, epitope_tokens):
        tcr_emb = self.encode_tcr(tcr_tokens)
        epi_emb = self.encode_epitope(epitope_tokens)
        return self.score(tcr_emb, epi_emb)

def info_nce_loss(logits):
    """Symmetric InfoNCE over the BxB pairwise-logit matrix."""
    B = logits.size(0)
    targets = torch.arange(B, device=logits.device)
    
    loss_tcr_epi = F.cross_entropy(logits, targets)
    loss_epi_tcr = F.cross_entropy(logits.t(), targets)
    
    return (loss_tcr_epi + loss_epi_tcr) / 2.0

def train_one_epoch(model, loader, device, lr=1e-3):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    total_loss = 0
    for tcr_batch, epi_batch, _ in loader:
        tcr_batch = tcr_batch.to(device)
        epi_batch = epi_batch.to(device)
        
        optimizer.zero_grad()
        
        tcr_emb = model.encode_tcr(tcr_batch)
        epi_emb = model.encode_epitope(epi_batch)
        
        logits = model.pairwise_logits(tcr_emb, epi_emb)
        
        loss = info_nce_loss(logits)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / max(len(loader), 1)

def evaluate_model(model, loader, catalogue_tokens, device):
    """Evaluates nearest-epitope top-1 accuracy against a catalogue of known epitopes."""
    model.eval()
    catalogue_tokens = catalogue_tokens.to(device)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        cat_emb = model.encode_epitope(catalogue_tokens)
        
        for tcr_batch, _, epi_ids in loader:
            tcr_batch = tcr_batch.to(device)
            epi_ids = epi_ids.to(device)
            
            tcr_emb = model.encode_tcr(tcr_batch)
            
            logits = model.pairwise_logits(tcr_emb, cat_emb)
            preds = logits.argmax(dim=1)
            
            correct += (preds == epi_ids).sum().item()
            total += tcr_batch.size(0)
            
    accuracy = correct / max(total, 1)
    return 0.0, accuracy
