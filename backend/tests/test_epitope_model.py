import torch
import pytest

from app.fl.epitope_model import (
    RBFKANLayer,
    FastKAN,
    ResidueMotifEncoder,
    SiameseEpitopeKANNet,
    info_nce_loss,
    PersonalCalibration
)

def test_rbf_kan_layer():
    layer = RBFKANLayer(in_features=16, out_features=32, grid_size=8)
    x = torch.randn(4, 16)
    out = layer(x)
    assert out.shape == (4, 32)
    
    # Check that grid is a fixed buffer, not a parameter (Required for MPC)
    assert "grid" in dict(layer.named_buffers())
    assert "grid" not in dict(layer.named_parameters())

def test_residue_motif_encoder():
    encoder = ResidueMotifEncoder(vocab_size=25, embed_dim=64, channels=128)
    # Batch of 4, sequence length 20
    x = torch.randint(0, 25, (4, 20))
    out = encoder(x)
    assert out.shape == (4, 128)

def test_siamese_epitope_kan_net():
    model = SiameseEpitopeKANNet(vocab_size=25, embed_dim=64, conv_channels=128, proj_dim=64)
    
    tcr = torch.randint(0, 25, (4, 30))
    epi = torch.randint(0, 25, (4, 20))
    
    # Test forward pass
    scores = model(tcr, epi)
    assert scores.shape == (4,)
    
    # Test pairwise logits (4 TCRs vs 4 Epitopes -> 4x4 matrix)
    tcr_emb = model.encode_tcr(tcr)
    epi_emb = model.encode_epitope(epi)
    logits = model.pairwise_logits(tcr_emb, epi_emb)
    assert logits.shape == (4, 4)

def test_info_nce_loss():
    # 4x4 pairwise matrix
    logits = torch.randn(4, 4, requires_grad=True)
    loss = info_nce_loss(logits)
    
    assert loss.dim() == 0
    loss.backward()
    assert logits.grad is not None

def test_fedper_parameter_split():
    model = SiameseEpitopeKANNet()
    
    shared_names = []
    local_names = []
    
    for name, param in model.named_parameters():
        if name.startswith("features.") or name.startswith("projection."):
            shared_names.append(name)
        elif name.startswith("personal_head."):
            local_names.append(name)
            
    assert len(shared_names) > 0, "Should have shared features/projection parameters"
    assert len(local_names) > 0, "Should have local personal parameters"
    
    # Total params should equal shared + local (no orphans)
    assert len(shared_names) + len(local_names) == len(list(model.parameters()))
