"""
Sequence Augmentation Functions
Maintains strict order: Deletion → Insertion → Translocation → Mutation
"""

import numpy as np
from typing import Optional


class SequenceAugmenter:
    """
    Augments protein sequences with strict ordering.
    
    Order: Deletion → Insertion → Translocation → Mutation

    Key parameters:
        mutate_frac (float): Fraction of sequence to mutate. Default 0.005 (≤0.5%).
        mutate_randomly (bool): 
            - False (default): Fixed strength — always mutate exactly floor(len * mutate_frac) positions.
            - True: At-most mode — randomly mutate between 1 and floor(len * mutate_frac) positions.
              Use this for the controlled study (Exp 1–3) to avoid a hard fixed mutation count.
    """
    
    def __init__(
        self,
        insert_max: int = 20,
        mutate_frac: float = 0.005,
        mutate_randomly: bool = True,
        shift_max: Optional[int] = None,
        shift_frac: float = 0.10,
        use_deletion: bool = True,
        use_insertion: bool = True,
        use_translocation: bool = True,
        use_mutation: bool = True,
        seed: Optional[int] = None,
    ):
        self.insert_max = insert_max
        self.mutate_frac = mutate_frac
        self.mutate_randomly = mutate_randomly  # If True, sample [1, max_mutations] instead of fixed max
        self.shift_max = shift_max
        self.shift_frac = shift_frac
        
        # Control which augmentations to use (for ablations)
        self.use_deletion = use_deletion
        self.use_insertion = use_insertion
        self.use_translocation = use_translocation
        self.use_mutation = use_mutation
        
        self.rng = np.random.RandomState(seed)
        
        # Valid amino acids
        self.valid_aa = list("ACDEFGHIKLMNPQRSTVWY")
    
    def augment(self, sequence: str, seed: Optional[int] = None) -> str:
        """
        Augment a single sequence.
        
        STRICT ORDER: Deletion → Insertion → Translocation → Mutation
        
        Args:
            sequence: Protein sequence string
            seed: Optional seed for reproducibility
        
        Returns:
            Augmented sequence
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = self.rng
        
        seq = sequence.upper()
        
        # Step 1: Deletion (if enabled)
        if self.use_deletion:
            seq = self._delete(seq, rng)
        
        # Step 2: Insertion (if enabled)
        if self.use_insertion:
            seq = self._insert(seq, rng)
        
        # Step 3: Translocation (if enabled)
        if self.use_translocation:
            seq = self._translocate(seq, rng)
        
        # Step 4: Mutation (if enabled)
        if self.use_mutation:
            seq = self._mutate(seq, rng)
        
        return seq
    
    def _delete(self, seq: str, rng: np.random.RandomState) -> str:
        """Delete a random segment."""
        if len(seq) < 2:
            return seq
        
        # Determine delete length (up to insert_max)
        max_del = min(self.insert_max, len(seq) - 1)
        del_len = rng.randint(1, max_del + 1)
        
        # Random start position
        start = rng.randint(0, len(seq) - del_len + 1)
        
        # Delete segment
        seq = seq[:start] + seq[start + del_len:]
        
        return seq
    
    def _insert(self, seq: str, rng: np.random.RandomState) -> str:
        """Insert a random segment."""
        # Determine insert length
        insert_len = rng.randint(1, self.insert_max + 1)
        
        # Random position
        pos = rng.randint(0, len(seq) + 1)
        
        # Generate random amino acids
        insert_seq = ''.join(rng.choice(self.valid_aa, size=insert_len))
        
        # Insert
        seq = seq[:pos] + insert_seq + seq[pos:]
        
        return seq
    
    def _translocate(self, seq: str, rng: np.random.RandomState) -> str:
        """Move a segment to a different position."""
        if len(seq) < 3:
            return seq
        
        # Determine shift length
        if self.shift_max is not None:
            max_shift = min(self.shift_max, len(seq))
        else:
            max_shift = max(1, int(len(seq) * self.shift_frac))
        
        shift_len = rng.randint(1, max_shift + 1)
        
        if shift_len >= len(seq):
            return seq
        
        # Random source position
        src_start = rng.randint(0, len(seq) - shift_len + 1)
        src_end = src_start + shift_len
        
        # Extract segment
        segment = seq[src_start:src_end]
        remaining = seq[:src_start] + seq[src_end:]
        
        # Random destination
        dst_pos = rng.randint(0, len(remaining) + 1)
        
        # Insert at new position
        seq = remaining[:dst_pos] + segment + remaining[dst_pos:]
        
        return seq
    
    def _mutate(self, seq: str, rng: np.random.RandomState) -> str:
        """Point mutations.
        
        If mutate_randomly=False: mutate exactly floor(len * mutate_frac) positions
            (fixed strength).
        If mutate_randomly=True (default): mutate a random number of positions in [1, max_mutations],
            i.e. "at most" mutate_frac of the sequence (controlled/gentle mode).
        """
        
        """
        Mutate only positions that are standard amino acids (20 AA).
        Non-standard symbols (e.g., '-', 'X') are left unchanged.
        """
        if len(seq) == 0:
            return seq

        # Candidate positions: only standard AA tokens
        candidates = [i for i, ch in enumerate(seq) if ch in self.valid_aa]
        if len(candidates) == 0:
            return seq
        
        # Force at least 1 mutation when candidates exist
        max_mutations = max(1, int(len(candidates) * self.mutate_frac))
        
        if self.mutate_randomly:
            # Sample anywhere from 1 up to max_mutations (inclusive)
            num_mutate = rng.randint(1, max_mutations + 1)
        else:
            # Fixed: always mutate exactly max_mutations positions
            num_mutate = max_mutations
        
        # Random positions
        positions = rng.choice(candidates, size=num_mutate, replace=False)
        
        # Mutate
        seq_list = list(seq)
        for pos in positions:
            # Choose different amino acid
            current_aa = seq_list[pos]
            possible_aa = [aa for aa in self.valid_aa if aa != current_aa]
            seq_list[pos] = rng.choice(possible_aa)
        
        return ''.join(seq_list)


def encode_aa_one_hot(sequence: str, max_len: Optional[int] = None) -> np.ndarray:
    """
    One-hot encode amino acid sequence.
    
    Args:
        sequence: Amino acid sequence
        max_len: Maximum length (will pad with zeros if needed)
    
    Returns:
        One-hot encoded array of shape (20, L) where L = max_len or len(sequence)
    """
    aa_to_idx = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    
    seq = sequence.upper()
    seq_len = len(seq)
    
    # Determine output length
    if max_len is not None:
        out_len = max_len
    else:
        out_len = seq_len
    
    # Initialize with zeros (padding)
    encoded = np.zeros((20, out_len), dtype=np.float32)
    
    # Encode sequence
    for i, aa in enumerate(seq):
        if i >= out_len:
            break
        if aa in aa_to_idx:
            encoded[aa_to_idx[aa], i] = 1.0
    
    return encoded