"""Custom model/loss/assigner patches for Phase 1 ablations (Mod A, B, C1, C2)."""

from .hooks import build_model_with_switches

__all__ = ["build_model_with_switches"]
