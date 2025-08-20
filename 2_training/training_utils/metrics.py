def compute_pairwise_jaccard_similarity(s0, s1):
    # quantify how much new set s1 changed from old set s0
    if isinstance(s0, list):
        s0 = set(s0)
    if isinstance(s1, list):
        s1 = set(s1)

    intersection = len(s0 & s1)
    union = len(s0 | s1)
    jaccard = intersection / union

    return jaccard