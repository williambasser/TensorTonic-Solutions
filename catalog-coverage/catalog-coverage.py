def catalog_coverage(recommendations: list, n_items: int) -> float:
    """
    Returns the fraction of catalog items that were recommended.
    """
    if n_items == 0:
        return 0.0
    items = set()
    for user in recommendations:
        for recommendation in user:
            items.add(recommendation)
    
    return len(items) / n_items