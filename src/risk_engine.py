def get_risk_level(churn_probability):
    """
    Convert churn probability into a business risk level.
    """

    if churn_probability < 0.25:
        return "Low"

    elif churn_probability < 0.50:
        return "Medium"

    elif churn_probability < 0.75:
        return "High"

    else:
        return "Very High"


def get_retention_priority(risk_level):
    """
    Convert risk level into retention priority.
    """

    priority_map = {
        "Low": "Normal",
        "Medium": "Monitor",
        "High": "Prioritize",
        "Very High": "Immediate"
    }

    return priority_map[risk_level]



if __name__ == "__main__":

    test_probabilities = [
        0.10,
        0.30,
        0.60,
        0.85
    ]

    for probability in test_probabilities:

        risk = get_risk_level(
            probability
        )

        priority = get_retention_priority(
            risk
        )

        print(
            f"Probability: {probability:.0%} | "
            f"Risk: {risk} | "
            f"Priority: {priority}"
        )