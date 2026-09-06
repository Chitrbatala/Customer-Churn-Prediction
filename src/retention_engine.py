def get_retention_recommendations(customer):
    """
    Generate retention recommendations based on
    customer characteristics.

    This is rule-based business logic.
    It does not claim causal relationships.
    """

    recommendations = []


    # --------------------------------------------------
    # Contract
    # --------------------------------------------------

    if customer["Contract"] == "Month-to-month":

        recommendations.append(
            "Offer an incentive to move to a longer-term contract."
        )


    # --------------------------------------------------
    # Technical support
    # --------------------------------------------------

    if customer["TechSupport"] == "No":

        recommendations.append(
            "Offer technical support or a support-service promotion."
        )


    # --------------------------------------------------
    # Online security
    # --------------------------------------------------

    if customer["OnlineSecurity"] == "No":

        recommendations.append(
            "Consider offering an online security package."
        )


    # --------------------------------------------------
    # Online backup
    # --------------------------------------------------

    if customer["OnlineBackup"] == "No":

        recommendations.append(
            "Consider offering an online backup package."
        )


    # --------------------------------------------------
    # Monthly charges
    # --------------------------------------------------

    if customer["MonthlyCharges"] >= 80:

        recommendations.append(
            "Review pricing and plan suitability with the customer."
        )


    # --------------------------------------------------
    # Payment method
    # --------------------------------------------------

    if customer["PaymentMethod"] == "Electronic check":

        recommendations.append(
            "Consider offering an automatic payment option."
        )


    # --------------------------------------------------
    # Tenure
    # --------------------------------------------------

    if customer["tenure"] <= 12:

        recommendations.append(
            "Provide an early-lifecycle retention offer or onboarding support."
        )


    # --------------------------------------------------
    # Fallback
    # --------------------------------------------------

    if not recommendations:

        recommendations.append(
            "Continue normal customer engagement and monitor churn risk."
        )


    return recommendations