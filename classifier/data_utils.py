"""Utilities for preparing protein sequences for ESM classifiers."""

from __future__ import annotations

from dataclasses import dataclass
import re

# ESM's fair-esm alphabet accepts the 20 canonical amino acids plus common
# ambiguous/special amino-acid tokens and gap characters. It does not accept
# stop codons ("*"), so those need explicit handling before tokenization.
ESM_ALLOWED_TOKENS = set("ACDEFGHIKLMNPQRSTVWYBXZUO.-")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class CleanedSequence:
    """A sequence after ESM-compatible cleanup plus auditable metadata."""

    sequence: str
    original_length: int
    cleaned_length: int
    stops_removed: int = 0
    stops_replaced: int = 0
    invalid_removed: int = 0
    invalid_replaced: int = 0

    @property
    def changed(self) -> bool:
        """Whether cleanup changed the input sequence."""
        return (
            self.original_length != self.cleaned_length
            or self.stops_removed > 0
            or self.stops_replaced > 0
            or self.invalid_removed > 0
            or self.invalid_replaced > 0
        )


def label_from_identifier(identifier: str) -> int:
    """Return the project label convention: cruci-like IDs are positive."""
    return 1 if "cruci" in str(identifier).lower() else 0


def clean_protein_sequence(
    sequence: str,
    *,
    stop_action: str = "remove",
    invalid_action: str = "replace_x",
) -> CleanedSequence:
    """Normalize a protein sequence so fair-esm can tokenize it.

    Args:
        sequence: Raw amino-acid sequence.
        stop_action: How to handle ``*`` stop characters:
            ``remove`` drops them, ``replace_x`` maps them to ``X``, and
            ``error`` raises if any are present.
        invalid_action: How to handle non-ESM characters after stop handling:
            ``replace_x`` maps them to ``X``, ``remove`` drops them, and
            ``error`` raises.

    Returns:
        CleanedSequence with the cleaned sequence and cleanup counts.
    """
    if sequence is None:
        raise ValueError("sequence cannot be None")
    if stop_action not in {"remove", "replace_x", "error"}:
        raise ValueError(f"unknown stop_action: {stop_action}")
    if invalid_action not in {"replace_x", "remove", "error"}:
        raise ValueError(f"unknown invalid_action: {invalid_action}")

    raw = _WHITESPACE_RE.sub("", str(sequence)).upper()
    original_length = len(raw)

    stop_count = raw.count("*")
    stops_removed = 0
    stops_replaced = 0
    if stop_count:
        if stop_action == "error":
            raise ValueError("sequence contains '*' stop characters")
        if stop_action == "remove":
            raw = raw.replace("*", "")
            stops_removed = stop_count
        else:
            raw = raw.replace("*", "X")
            stops_replaced = stop_count

    cleaned_chars: list[str] = []
    invalid_removed = 0
    invalid_replaced = 0
    for char in raw:
        if char in ESM_ALLOWED_TOKENS:
            cleaned_chars.append(char)
        elif invalid_action == "error":
            raise ValueError(f"sequence contains unsupported ESM token: {char!r}")
        elif invalid_action == "remove":
            invalid_removed += 1
        else:
            cleaned_chars.append("X")
            invalid_replaced += 1

    cleaned = "".join(cleaned_chars)
    return CleanedSequence(
        sequence=cleaned,
        original_length=original_length,
        cleaned_length=len(cleaned),
        stops_removed=stops_removed,
        stops_replaced=stops_replaced,
        invalid_removed=invalid_removed,
        invalid_replaced=invalid_replaced,
    )
