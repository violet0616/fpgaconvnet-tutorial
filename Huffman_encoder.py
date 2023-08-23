from dataclasses import dataclass, field
import numpy as np


@dataclass
class Huffman_encoder:
    B: int = 0
    D: int = 0
    F: int = 0
    L: float = 0
    Rate: int = 1
    II: int = 0
    Latency: int = 0