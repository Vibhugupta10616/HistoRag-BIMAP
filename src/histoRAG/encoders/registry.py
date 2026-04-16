"""Encoder registry: maps config name strings to Encoder subclasses.

Adding a new encoder in Phase 1+:
    1. Implement the Encoder ABC in a new module under histoRAG/encoders/.
    2. Import the class here and add it to ENCODERS.
    3. Reference it by name in the YAML config: encoder.name: <your_name>
"""
from __future__ import annotations

from histoRAG.encoders.clip import ClipEncoder
from histoRAG.encoders.uni import UNIEncoder

ENCODERS: dict[str, type] = {
    "clip-vitb16": ClipEncoder,
    "uni2h": UNIEncoder,
}


def get_encoder(name: str, **kwargs):
    """Instantiate and return the encoder registered under *name*.

    Parameters
    ----------
    name:
        Key in ENCODERS (e.g. 'clip-vitb16', 'uni2h').
    **kwargs:
        Forwarded to the encoder's constructor (e.g. device='cpu').

    Raises
    ------
    KeyError if *name* is not registered.
    """
    if name not in ENCODERS:
        raise KeyError(f"Unknown encoder '{name}'. Available: {sorted(ENCODERS)}")
    return ENCODERS[name](**kwargs)
