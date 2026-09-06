// ============================================
// CUSTOMER CHURN INTELLIGENCE DASHBOARD
// ============================================

// --------------------------------------------
// DOM ELEMENTS
// --------------------------------------------

const form = document.getElementById("customerForm");
const predictButton = document.getElementById("predictButton");

const resultsSection = document.getElementById("results");

const predictionBox = document.getElementById("predictionBox");
const riskBox = document.getElementById("riskBox");

const churnProbability = document.getElementById("churnProbability");
const probabilityFill = document.getElementById("probabilityFill");

const prediction = document.getElementById("prediction");
const riskLevel = document.getElementById("riskLevel");
const retentionPriority = document.getElementById("retentionPriority");

const riskFactors = document.getElementById("riskFactors");
const protectiveFactors = document.getElementById("protectiveFactors");
const recommendations = document.getElementById("recommendations");


// ============================================
// FORM SUBMISSION
// ============================================

form.addEventListener("submit", async function (event) {

    event.preventDefault();

    // ----------------------------------------
    // Loading state
    // ----------------------------------------

    const originalButtonText = predictButton.innerHTML;

    predictButton.disabled = true;

    predictButton.innerHTML = `
        <span class="loading-spinner"></span>
        Analyzing Customer...
    `;


    try {

        // ====================================
        // COLLECT CUSTOMER DATA
        // ====================================

        const customer = {

            gender:
                document.getElementById("gender").value,

            SeniorCitizen:
                Number(
                    document.getElementById("SeniorCitizen").value
                ),

            Partner:
                document.getElementById("Partner").value,

            Dependents:
                document.getElementById("Dependents").value,

            tenure:
                Number(
                    document.getElementById("tenure").value
                ),

            PhoneService:
                document.getElementById("PhoneService").value,

            MultipleLines:
                document.getElementById("MultipleLines").value,

            InternetService:
                document.getElementById("InternetService").value,

            OnlineSecurity:
                document.getElementById("OnlineSecurity").value,

            OnlineBackup:
                document.getElementById("OnlineBackup").value,

            DeviceProtection:
                document.getElementById("DeviceProtection").value,

            TechSupport:
                document.getElementById("TechSupport").value,

            StreamingTV:
                document.getElementById("StreamingTV").value,

            StreamingMovies:
                document.getElementById("StreamingMovies").value,

            Contract:
                document.getElementById("Contract").value,

            PaperlessBilling:
                document.getElementById("PaperlessBilling").value,

            PaymentMethod:
                document.getElementById("PaymentMethod").value,

            MonthlyCharges:
                Number(
                    document.getElementById("MonthlyCharges").value
                ),

            TotalCharges:
                Number(
                    document.getElementById("TotalCharges").value
                )
        };


        // ====================================
        // DEBUG
        // ====================================

        console.log(
            "Customer data being sent:",
            customer
        );


        // ====================================
        // SEND DATA TO FASTAPI
        // ====================================

        const response = await fetch(
            "/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(customer)
            }
        );


        // ====================================
        // HANDLE API ERROR
        // ====================================

        if (!response.ok) {

            let errorMessage;

            try {

                const errorData =
                    await response.json();

                console.error(
                    "FastAPI validation error:",
                    errorData
                );

                errorMessage =
                    JSON.stringify(
                        errorData,
                        null,
                        2
                    );

            } catch {

                errorMessage =
                    `API Error: ${response.status}`;
            }


            throw new Error(errorMessage);
        }


        // ====================================
        // GET RESPONSE
        // ====================================

        const result =
            await response.json();


        console.log(
            "Prediction result:",
            result
        );


        // ====================================
        // DISPLAY RESULT
        // ====================================

        displayPrediction(result);


        // ====================================
        // SHOW RESULTS SECTION
        // ====================================

        if (resultsSection) {

            resultsSection.classList.remove("hidden");

            resultsSection.classList.add("show");

            // Smoothly move to results
            setTimeout(() => {

                resultsSection.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });

            }, 100);
        }

    }


    // ========================================
    // ERROR HANDLING
    // ========================================

    catch (error) {

        console.error(
            "Prediction error:",
            error
        );


        alert(
            "Prediction failed.\n\n" +
            error.message
        );
    }


    // ========================================
    // RESTORE BUTTON
    // ========================================

    finally {

        predictButton.disabled = false;

        predictButton.innerHTML =
            originalButtonText;
    }

});


// ============================================
// DISPLAY PREDICTION
// ============================================

function displayPrediction(result) {

    // ========================================
    // CHURN PROBABILITY
    // ========================================

    const probability =
        Number(
            result.churn_probability_percentage
        );

    const scoreRing = document.querySelector(".score-ring");

    if (scoreRing) {
        const angle = Math.min(Math.max(probability, 0), 100) * 3.6;

        scoreRing.style.setProperty(
            "--risk-angle",
            `${angle}deg`
        );
    }


    if (churnProbability) {

        churnProbability.textContent =
            `${probability.toFixed(2)}%`;
    }


    if (probabilityFill) {

        probabilityFill.style.width =
            `${probability}%`;
    }


    // ========================================
    // PREDICTION
    // ========================================

    if (prediction) {

        prediction.textContent =
            result.churn_prediction || "--";
    }


    // ========================================
    // PREDICTION BOX STYLE
    // ========================================

    if (predictionBox) {

        predictionBox.classList.remove(
            "prediction-churn",
            "prediction-stay"
        );


        if (
            result.churn_prediction ===
            "Likely to Churn"
        ) {

            predictionBox.classList.add(
                "prediction-churn"
            );

        } else {

            predictionBox.classList.add(
                "prediction-stay"
            );
        }
    }


    // ========================================
    // RISK LEVEL
    // ========================================

    if (riskLevel) {

        riskLevel.textContent =
            result.risk_level || "--";
    }


    // ========================================
    // RISK BOX STYLE
    // ========================================

    if (riskBox) {

        riskBox.classList.remove(
            "risk-low",
            "risk-medium",
            "risk-high",
            "risk-very-high"
        );


        const riskClass =
            String(
                result.risk_level || ""
            )
            .toLowerCase()
            .replace(/\s+/g, "-");


        if (riskClass) {

            riskBox.classList.add(
                `risk-${riskClass}`
            );
        }
    }


    // ========================================
    // RETENTION PRIORITY
    // ========================================

    if (retentionPriority) {

        retentionPriority.textContent =
            result.retention_priority || "--";
    }


    // ========================================
    // SHAP RISK FACTORS
    // ========================================

    renderRiskFactors(
        result.risk_factors || []
    );


    // ========================================
    // SHAP PROTECTIVE FACTORS
    // ========================================

    renderProtectiveFactors(
        result.protective_factors || []
    );


    // ========================================
    // RETENTION RECOMMENDATIONS
    // ========================================

    renderRecommendations(
        result.retention_recommendations || []
    );
}


// ============================================
// RISK FACTORS
// ============================================

function renderRiskFactors(factors) {

    if (!riskFactors) {
        return;
    }


    riskFactors.innerHTML = "";


    // ----------------------------------------
    // Empty state
    // ----------------------------------------

    if (factors.length === 0) {

        const li =
            document.createElement("li");

        li.className =
            "factor-item";

        li.textContent =
            "No major risk factors identified.";

        riskFactors.appendChild(li);

        return;
    }


    // ----------------------------------------
    // Find maximum SHAP impact
    // ----------------------------------------

    const maxRiskShap =
        Math.max(
            ...factors.map(
                item =>
                    Math.abs(
                        Number(
                            item.shap_value
                        ) || 0
                    )
            ),
            1
        );


    // ----------------------------------------
    // Create factor cards
    // ----------------------------------------

    factors.forEach(item => {

        const li =
            document.createElement("li");

        li.className =
            "factor-item risk-factor";


        const content =
            document.createElement("div");

        content.className =
            "factor-content";


        // Feature name
        const title =
            document.createElement("div");

        title.className =
            "factor-title";

        title.textContent =
            item.feature;


        // Effect
        const effect =
            document.createElement("div");

        effect.className =
            "factor-effect";

        effect.textContent =
            item.effect;


        // Bar container
        const barContainer =
            document.createElement("div");

        barContainer.className =
            "factor-bar";


        // Bar
        const bar =
            document.createElement("div");

        bar.className =
            "factor-bar-fill risk-bar";


        const shapValue =
            Math.abs(
                Number(
                    item.shap_value
                ) || 0
            );


        const percentage =
            (shapValue / maxRiskShap) * 100;


        bar.style.width =
            `${percentage}%`;


        // Build
        barContainer.appendChild(bar);

        content.appendChild(title);

        content.appendChild(effect);

        content.appendChild(barContainer);

        li.appendChild(content);

        riskFactors.appendChild(li);

    });
}


// ============================================
// PROTECTIVE FACTORS
// ============================================

function renderProtectiveFactors(factors) {

    if (!protectiveFactors) {
        return;
    }


    protectiveFactors.innerHTML = "";


    // ----------------------------------------
    // Empty state
    // ----------------------------------------

    if (factors.length === 0) {

        const li =
            document.createElement("li");

        li.className =
            "factor-item";

        li.textContent =
            "No major protective factors identified.";

        protectiveFactors.appendChild(li);

        return;
    }


    // ----------------------------------------
    // Maximum SHAP impact
    // ----------------------------------------

    const maxProtectiveShap =
        Math.max(
            ...factors.map(
                item =>
                    Math.abs(
                        Number(
                            item.shap_value
                        ) || 0
                    )
            ),
            1
        );


    // ----------------------------------------
    // Create cards
    // ----------------------------------------

    factors.forEach(item => {

        const li =
            document.createElement("li");

        li.className =
            "factor-item protective-factor";


        const content =
            document.createElement("div");

        content.className =
            "factor-content";


        // Feature
        const title =
            document.createElement("div");

        title.className =
            "factor-title";

        title.textContent =
            item.feature;


        // Effect
        const effect =
            document.createElement("div");

        effect.className =
            "factor-effect";

        effect.textContent =
            item.effect;


        // Bar container
        const barContainer =
            document.createElement("div");

        barContainer.className =
            "factor-bar";


        // Bar
        const bar =
            document.createElement("div");

        bar.className =
            "factor-bar-fill protective-bar";


        const shapValue =
            Math.abs(
                Number(
                    item.shap_value
                ) || 0
            );


        const percentage =
            (shapValue /
                maxProtectiveShap) * 100;


        bar.style.width =
            `${percentage}%`;


        // Build
        barContainer.appendChild(bar);

        content.appendChild(title);

        content.appendChild(effect);

        content.appendChild(barContainer);

        li.appendChild(content);

        protectiveFactors.appendChild(li);

    });
}


// ============================================
// RETENTION RECOMMENDATIONS
// ============================================

function renderRecommendations(items) {

    if (!recommendations) {
        return;
    }


    recommendations.innerHTML = "";


    // ----------------------------------------
    // Empty state
    // ----------------------------------------

    if (items.length === 0) {

        const li =
            document.createElement("li");

        li.className =
            "recommendation-item";


        const number =
            document.createElement("span");

        number.className =
            "recommendation-number";

        number.textContent =
            "01";


        const text =
            document.createElement("span");

        text.className =
            "recommendation-text";

        text.textContent =
            "Continue normal customer engagement and monitor churn risk.";


        li.appendChild(number);

        li.appendChild(text);

        recommendations.appendChild(li);

        return;
    }


    // ----------------------------------------
    // Create recommendation cards
    // ----------------------------------------

    items.forEach(
        (recommendation, index) => {

            const li =
                document.createElement("li");

            li.className =
                "recommendation-item";


            // Number
            const number =
                document.createElement("span");

            number.className =
                "recommendation-number";

            number.textContent =
                String(index + 1)
                    .padStart(2, "0");


            // Text
            const text =
                document.createElement("span");

            text.className =
                "recommendation-text";

            text.textContent =
                recommendation;


            // Build
            li.appendChild(number);

            li.appendChild(text);

            recommendations.appendChild(li);
        }
    );
}


// ============================================
// INITIAL STATE
// ============================================

if (resultsSection) {

    resultsSection.classList.add("hidden");

    resultsSection.classList.remove("show");
}