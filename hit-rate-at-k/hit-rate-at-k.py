def hit_rate_at_k(recommendations: list, ground_truth: list, k: int) -> float:
    """
    Returns the fraction of users with a relevant item in their first k recommendations.
    """
    # Write code here
    hits = 0
    users = len(recommendations)
    if not users:
        return 0
    for user_recs, user_gt in zip(recommendations, ground_truth):
        ground_truth_set = set(user_gt) 
        user_recommendations_at_k = set(user_recs[:k])
        if user_recommendations_at_k & ground_truth_set:
            hits+=1

    return float(hits) / float(users)