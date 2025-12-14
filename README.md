# Credit Risk Probability Model for Alternative Data

## Credit Scoring Business Understanding

### 1. Basel II and Model Interpretability

[cite_start]The Basel II Capital Accord requires financial institutions to hold capital commensurate with the risks they bear[cite: 1083]. This regulatory framework influences model development in several critical ways:

* **Auditability and Validation:** Basel II requires banks using internal risk models (Internal Ratings Based or IRB approach) to validate their models thoroughly. This necessitates that the model's logic be transparent and easy to audit to ensure the estimated risk metrics (like Probability of Default) are reliable for calculating regulatory capital.
* **Transparency and Documentation:** An interpretable model allows stakeholders—including regulators, senior management, and risk officers—to understand *why* a specific score or risk probability was assigned. [cite_start]Poorly documented or complex "black-box" models pose significant **model risk** and fail to meet the required standards for explanation and justification[cite: 1188].
* **Compliance:** The need for clear, documented rules ensures the bank can comply with legal and fairness requirements, particularly when justifying the denial of a loan.

### 2. The Necessity and Risks of a Proxy Variable

[cite_start]In this project, we are developing a credit scoring model using transactional data from an eCommerce platform, which inherently **lacks a direct, pre-labeled "default" column** (ground truth)[cite: 1085, 1244].

* **Necessity:** A supervised machine learning model requires a defined target variable (or label, Y) to train on. Since a direct default label is absent, a **proxy variable** must be engineered. [cite_start]This is achieved by transforming observable customer behavior (Recency, Frequency, Monetary/RFM patterns) into an estimated risk signal [cite: 1086, 1245, 1248-1249].
* **Potential Business Risks of using a Proxy:**
    * **Proxy Error (Measurement Error):** The primary risk is that the RFM proxy is an imperfect substitute for true default behavior. A customer classified as "high-risk" based on low engagement (low RFM) may still be creditworthy.
    * **Incorrect Decisions:** This error can lead to two types of costly errors:
        * **False Negatives (Type I Error):** Rejecting creditworthy customers (labeled high-risk by proxy) and losing profitable business opportunities.
        * **False Positives (Type II Error):** Approving high-risk customers (labeled low-risk by proxy) who ultimately default, leading to financial losses for the bank.
    * **Model Miscalibration:** If the proxy poorly reflects actual risk, the resulting probability scores will be inaccurate when applied to the real-world population.

### 3. Model Trade-offs in a Regulated Financial Context

The choice between a simple, interpretable model and a complex, high-performance model involves balancing regulatory necessity with predictive power:

| Model Type | Primary Advantage | Primary Disadvantage | Regulatory Acceptance (Basel II) |
| :--- | :--- | :--- | :--- |
| **Simple (Logistic Regression + WoE)** | High **Interpretability** (clear, linear rules, scorecards) and stability. [cite_start]WoE provides a statistically robust, traditional feature transformation[cite: 1241]. | [cite_start]Lower predictive performance (e.g., lower AUC) compared to complex models[cite: 1191]. | **High.** Preferred for ease of audit, validation, and justification of decisions. |
| **Complex (Gradient Boosting / XGBoost)** | [cite_start]High **Predictive Accuracy** (AUC) by capturing non-linear relationships and complex feature interactions[cite: 1191]. | Low **Interpretability** (black-box). Difficult to explain why a specific customer received a specific score. | **Lower.** Requires significant extra work (e.g., SHAP, LIME) to satisfy regulatory demands for transparency and explanation. |

[cite_start]In a regulated financial context, the trade-off often favors **interpretability** and **auditability** (Logistic Regression/Scorecards) over raw predictive performance, prioritizing compliance and robust risk management required by frameworks like Basel II[cite: 1188].
