import torch
import math
from torch import Tensor
from src.models.eegconformer import PatchEmbedding, ClassificationHead, EEGConformer


def test_output_shape_default() -> None:
    """Test that the default PatchEmbedding produces the expected shape."""
    model = PatchEmbedding()
    x: Tensor = torch.randn(4, 22, 1000)  # (B, C, T)
    out: Tensor = model(x)

    # Compute expected L based on the formula
    T = 1000
    temporal_kernel = 25
    pool_kernel = 75
    pool_stride = 15

    L = math.floor((T - temporal_kernel + 1 - pool_kernel) / pool_stride + 1)
    E = 40  # default emb_size

    assert out.shape == (4, L, E)


def test_embedding_dim_custom() -> None:
    """Test that changing emb_size affects the output dimension."""
    model = PatchEmbedding(emb_size=64)
    x: Tensor = torch.randn(2, 22, 500)
    out: Tensor = model(x)

    T = 500
    temporal_kernel = 25
    pool_kernel = 75
    pool_stride = 15
    L = math.floor((T - temporal_kernel + 1 - pool_kernel) / pool_stride + 1)
    E = 64

    assert out.shape == (2, L, E)


def test_num_channels_effect() -> None:
    """Test that changing the number of channels runs without shape errors."""
    model = PatchEmbedding(num_channels=16)
    x: Tensor = torch.randn(3, 16, 800)
    out: Tensor = model(x)

    T = 800
    temporal_kernel = 25
    pool_kernel = 75
    pool_stride = 15
    L = math.floor((T - temporal_kernel + 1 - pool_kernel) / pool_stride + 1)
    E = 40

    assert out.shape == (3, L, E)


def test_classification_head_output_shape() -> None:
    """Test that ClassificationHead maps (B, L, E) -> (B, K) as expected."""
    B, L, E, K = 4, 10, 40, 5
    model = ClassificationHead(dropout=0.5, num_classes=K)

    x = torch.randn(B, L, E)
    out = model(x)

    assert out.shape == (B, K)


def test_classification_head_robust_to_sequence_length() -> None:
    """Test that the head works for varying sequence lengths L."""
    B, E, K = 2, 64, 3
    for L in [5, 20, 50]:
        model = ClassificationHead(dropout=0.3, num_classes=K)  # fresh instance per L
        x = torch.randn(B, L, E)
        out = model(x)
        assert out.shape == (B, K)


def test_eegconformer_forward_shape_default() -> None:
    """Test that EEGConformer produces correct output shape with defaults."""
    model = EEGConformer()
    x: Tensor = torch.randn(4, 22, 1000)
    out: Tensor = model(x)

    K = 4  # num_classes

    assert out.shape == (4, K)


def test_eegconformer_custom_params() -> None:
    """Test EEGConformer with custom parameters changes final dim correctly."""
    model = EEGConformer(
        emb_size=64,
        num_channels=16,
        num_classes=6,
        num_heads=8,
        num_layers=3,
        dropout=0.4,
    )
    x: Tensor = torch.randn(2, 16, 800)
    out: Tensor = model(x)

    assert out.shape == (2, 6)


def test_eegconformer_consistency_with_patch() -> None:
    """Verify internal PatchEmbedding consistency within EEGConformer."""
    model = EEGConformer()
    x = torch.randn(1, 22, 500)
    with torch.no_grad():
        patch_out = model.patch_embeding(x)  # (B, L, E)
        enc_out = model.encoder(patch_out)  # (B, L, E)
        head_out = model.head(enc_out)  # (B, K)

    assert head_out.shape[-1] == 4
    assert enc_out.shape[:-1] == patch_out.shape[:-1]
    assert enc_out.shape[-1] == patch_out.shape[-1]
