def double_constrained_knapsack(values, weights1, weights2, max_weight1, max_weight2):
    """
    Solve the double-constrained knapsack problem.

    :param values: List of item values.
    :param weights1: List of item weights for the first constraint.
    :param weights2: List of item weights for the second constraint.
    :param max_weight1: Maximum weight for the first constraint.
    :param max_weight2: Maximum weight for the second constraint.
    :return: Maximum value that fits within the constraints.
    """
    n = len(values)

    # Initialize DP table with zeros
    dp = [[[0 for _ in range(max_weight2 + 1)] for _ in range(max_weight1 + 1)] for _ in range(n + 1)]

    # Iterate over items
    for i in range(1, n + 1):
        value = values[i - 1]
        weight1 = weights1[i - 1]
        weight2 = weights2[i - 1]

        # Iterate over each possible weight in both constraints
        for w1 in range(max_weight1 + 1):
            for w2 in range(max_weight2 + 1):
                # If the item can be included based on its weights
                if weight1 <= w1 and weight2 <= w2:
                    dp[i][w1][w2] = max(dp[i - 1][w1][w2], dp[i - 1][w1 - weight1][w2 - weight2] + value)
                else:
                    dp[i][w1][w2] = dp[i - 1][w1][w2]

    # The maximum value with the given constraints
    return dp[n][max_weight1][max_weight2]


# Example usage
values = [60, 100, 120]
weights1 = [10, 20, 30]
weights2 = [5, 10, 15]
max_weight1 = 50
max_weight2 = 30

max_value = double_constrained_knapsack(values, weights1, weights2, max_weight1, max_weight2)
print("Maximum value:", max_value)
