"""Loss functions."""

from .classification import classification_loss
from .alignment import info_nce_loss, cosine_alignment_loss
from .latent_consistency import latent_consistency_loss
from .combined_loss import CombinedLoss

__all__ = [
    "classification_loss",
    "info_nce_loss",
    "cosine_alignment_loss",
    "latent_consistency_loss",
    "CombinedLoss",
]
