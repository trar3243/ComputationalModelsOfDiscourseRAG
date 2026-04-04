"""
Chunking evaluation metrics for coreference-aware RAG analysis.

All three metric functions operate on a single generic input:

  mention_chunks : List[List[int]]
      Outer list  — one entry per chain.
      Inner list  — one chunk-id per mention in that chain, in mention order.

This format is dataset-agnostic.  Use the helper below to produce it from
LitBank's tokenised sentences + [sent_id, start_tok, end_tok] mention format.
For other datasets, write a comparable builder that maps each mention to a
character offset and divides by chunk_size.

Metrics
-------
cluster_break_rate   : fraction of chains whose mentions span more than one chunk.
edge_cut_rate        : fraction of consecutive mention-pairs (edges) crossing a boundary.
entity_concentration : average per-chunk dominance of the most-mentioned entity.
"""

from collections import defaultdict


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def cluster_break_rate(mention_chunks: list[list[int]]) -> float:
    """
    Cluster Break Rate = # chains spanning multiple chunks / # total chains.

    A chain "breaks" when at least two of its mentions fall in different chunks.
    Singleton chains count in the denominator but never break.
    """
    if not mention_chunks:
        return 0.0

    broken = sum(1 for chunks in mention_chunks if len(set(chunks)) > 1)
    return broken / len(mention_chunks)


def edge_cut_rate(mention_chunks: list[list[int]]) -> float:
    """
    Edge Cut Rate = # edges crossing a chunk boundary / # total edges.

    An "edge" is a consecutive mention-pair within the same chain.
    Singleton chains contribute no edges and are excluded from both counts.
    """
    total = 0
    cut   = 0
    for chunks in mention_chunks:
        for a, b in zip(chunks, chunks[1:]):
            total += 1
            if a != b:
                cut += 1
    return cut / total if total > 0 else 0.0


def entity_concentration(mention_chunks: list[list[int]], total_chunks: int) -> float:
    """
    Entity Concentration = (1 / total_chunks) * Σ_k (max_mentions_k / total_mentions_k)

    For each chunk k:
      max_mentions_k   — count of the most-mentioned chain in chunk k.
      total_mentions_k — total mention instances in chunk k across all chains.
    Chunks with zero mentions contribute 0 to the sum.

    Higher values mean each chunk is dominated by a single entity (concentrated);
    lower values mean mentions are spread evenly across many entities per chunk.
    """
    # chunk_id -> {chain_idx -> count}
    chunk_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for chain_idx, chunks in enumerate(mention_chunks):
        for chunk_id in chunks:
            chunk_counts[chunk_id][chain_idx] += 1

    concentration_sum = 0.0
    for chunk_id in range(total_chunks):
        counts = chunk_counts.get(chunk_id)
        if not counts:
            continue
        total = sum(counts.values())
        concentration_sum += max(counts.values()) / total

    return concentration_sum / total_chunks if total_chunks > 0 else 0.0


# ---------------------------------------------------------------------------
# Character-based chunking helpers (LitBank / any tokenised-sentence corpus)
# ---------------------------------------------------------------------------

def _build_offsets(sentences: list[list[str]]) -> tuple[dict[tuple[int, int], int], str]:
    """
    Flatten tokenised sentences into a single space-joined string and record
    the character start position of every token.

    Returns
    -------
    token_char_start : dict mapping (sent_id, tok_idx) -> char position in flat_text
    flat_text        : the full document as a single string
    """
    token_char_start: dict[tuple[int, int], int] = {}
    pos = 0
    for sent_id, sent in enumerate(sentences):
        for tok_idx, token in enumerate(sent):
            token_char_start[(sent_id, tok_idx)] = pos
            pos += len(token) + 1  # +1 for the space separator
    flat_text = " ".join(token for sent in sentences for token in sent)
    return token_char_start, flat_text


def chunk_text(sentences: list[list[str]], chunk_size: int = 100) -> list[str]:
    """
    Split the document into fixed-size character chunks.

    Parameters
    ----------
    sentences  : List[List[str]] — tokenised sentences
    chunk_size : number of characters per chunk

    Returns
    -------
    List[str] — the text of each chunk in order
    """
    _, flat_text = _build_offsets(sentences)
    return [flat_text[i : i + chunk_size] for i in range(0, max(len(flat_text), 1), chunk_size)]


def build_mention_chunks_char(
    coref_chains: list[list[list[int]]],
    sentences: list[list[str]],
    chunk_size: int = 100,
) -> tuple[list[list[int]], int]:
    """
    Assign each mention to the chunk that contains its first character.

    Parameters
    ----------
    coref_chains : LitBank-style chains — List[List[[sent_id, start_tok, end_tok]]]
    sentences    : List[List[str]] — tokenised sentences from the same document
    chunk_size   : number of characters per chunk

    Returns
    -------
    mention_chunks : List[List[int]] — chunk-id per mention, ready for the metrics
    total_chunks   : int
    """
    token_char_start, flat_text = _build_offsets(sentences)
    total_chunks = (max(len(flat_text), 1) + chunk_size - 1) // chunk_size

    mention_chunks = []
    for chain in coref_chains:
        chain_chunk_ids = []
        for mention in chain:
            sent_id, start_tok = mention[0], mention[1]
            char_start = token_char_start.get((sent_id, start_tok), 0)
            chain_chunk_ids.append(char_start // chunk_size)
        mention_chunks.append(chain_chunk_ids)

    return mention_chunks, total_chunks
