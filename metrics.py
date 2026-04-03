"""
Chunking evaluation metrics for coreference-aware RAG analysis.

All three metrics operate on:
  - coref_chains   : List[List[List[int]]]
                     Each chain is a list of mentions.
                     Each mention is [sentence_id, start_token, end_token] (0-based, inclusive).
  - sent_to_chunk  : dict[int, int]
                     Maps each sentence index to a chunk index.

Metrics
-------
cluster_break_rate   : fraction of chains whose mentions span more than one chunk.
edge_cut_rate        : fraction of consecutive mention-pairs (edges) that cross a chunk boundary.
entity_concentration : average per-chunk dominance of the most-mentioned entity.
"""

from collections import defaultdict


def cluster_break_rate(
    coref_chains: list[list[list[int]]],
    sent_to_chunk: dict[int, int],
) -> float:
    """
    Cluster Break Rate = # chains spanning multiple chunks / # total chains.

    A chain "breaks" if at least two of its mentions fall in different chunks.
    Singleton chains (single mention) are counted in the denominator but never break.
    """
    if not coref_chains:
        return 0.0

    broken = 0
    for chain in coref_chains:
        chunks_hit = {sent_to_chunk[m[0]] for m in chain if m[0] in sent_to_chunk}
        if len(chunks_hit) > 1:
            broken += 1

    return broken / len(coref_chains)


def edge_cut_rate(
    coref_chains: list[list[list[int]]],
    sent_to_chunk: dict[int, int],
) -> float:
    """
    Edge Cut Rate = # coreference edges crossing a chunk boundary / # total edges.

    An "edge" is a pair of consecutive mentions within the same chain
    (i.e., chain[i] -> chain[i+1]).  Chains with only one mention contribute
    no edges and are excluded from both numerator and denominator.
    """
    total_edges = 0
    cut_edges = 0

    for chain in coref_chains:
        if len(chain) < 2:
            continue
        for a, b in zip(chain, chain[1:]):
            chunk_a = sent_to_chunk.get(a[0])
            chunk_b = sent_to_chunk.get(b[0])
            if chunk_a is None or chunk_b is None:
                continue
            total_edges += 1
            if chunk_a != chunk_b:
                cut_edges += 1

    return cut_edges / total_edges if total_edges > 0 else 0.0


def entity_concentration(
    coref_chains: list[list[list[int]]],
    sent_to_chunk: dict[int, int],
    total_chunks: int,
) -> float:
    """
    Entity Concentration = (1 / total_chunks) * Σ_k (max_mentions_k / total_mentions_k)

    For each chunk k:
      - Count how many times each chain is mentioned (i.e., has a mention in chunk k).
      - max_mentions_k  = count of the most-mentioned chain in chunk k.
      - total_mentions_k = total mention instances across all chains in chunk k.
    Chunks with zero mentions contribute 0 to the sum.

    Higher values indicate that each chunk is dominated by a single entity
    (good concentration); lower values indicate mentions are spread evenly
    across many entities within chunks.
    """
    # chunk_index -> {chain_index -> mention_count}
    chunk_chain_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for chain_idx, chain in enumerate(coref_chains):
        for mention in chain:
            chunk = sent_to_chunk.get(mention[0])
            if chunk is None:
                continue
            chunk_chain_counts[chunk][chain_idx] += 1

    concentration_sum = 0.0
    for chunk_idx in range(total_chunks):
        counts = chunk_chain_counts.get(chunk_idx, {})
        if not counts:
            continue
        total = sum(counts.values())
        top   = max(counts.values())
        concentration_sum += top / total

    return concentration_sum / total_chunks if total_chunks > 0 else 0.0
