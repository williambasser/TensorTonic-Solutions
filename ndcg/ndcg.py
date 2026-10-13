import math
def ndcg(relevance_scores: list, k: int) -> float:
    """
    Returns NDCG as a float.
    """
    # Write code here
    # Top k scores

    dcg = 0
    for i, r in enumerate(relevance_scores[:k]):
        dcg += (2**r - 1) / (math.log2(i+2))

    sorted_scores = sorted(relevance_scores, reverse=True)
    idcg = 0
    for i, r in enumerate(sorted_scores[:k]):
        idcg += (2**r - 1) / (math.log2(i+2))
    if idcg == 0:
        return 0
    return dcg/idcg