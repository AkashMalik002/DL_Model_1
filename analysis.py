"""
Situated Learning Analysis and Domain-Specific Discussions
"""


def print_situated_learning_analysis():
    """Print comprehensive situated learning analysis for all domains"""
    
    analysis = """

### DOMAIN 1: Natural Language Processing - Review Moderation System
**Situated Learning Question:** In large-scale digital moderation systems, the cost of incorrect 
classification can directly affect customer trust and legal accountability. Critically evaluate 
the importance of Precision, Recall, and F1-Score within this context.

**Analysis:**
In e-commerce moderation systems, the choice between Precision and Recall depends on operational costs:

1. **HIGH RECALL (Prioritize catching harmful content):**
   - If False Negatives (missing harmful content) are costly:
     - Legal liability for hate speech/defamation increases exponentially
     - Brand reputation damage spreads virally on social media
     - Regulatory fines (GDPR, platform guidelines) apply retroactively
   - Accept some False Positives (incorrect removals) as operational cost
   - Strategy: Remove slightly more to ensure harmful content doesn't escape
   - Recall target: >95% to catch near all harmful reviews

2. **HIGH PRECISION (Avoid over-moderation):**
   - If False Positives (incorrectly removing legitimate content) are costly:
     - Customer frustration from legitimate review removal
     - Seller disputes over unfair content removal
     - Lost platform engagement and trust
   - Accept some harmful content slipping through
   - Strategy: Only remove content with high certainty
   - Precision target: >98% to minimize user friction

**Practical Recommendation:**
Use F1-Score for balanced assessment, but optimize Recall to >0.90 to minimize legal risk 
while maintaining Precision >0.85 for user experience.

---

### DOMAIN 2: Agriculture - Crop Health Monitoring
**Situated Learning Question:** If a Multi-Layer Perceptron achieves only marginal improvement 
over a baseline model while significantly increasing computational cost, is it suitable for 
deployment on energy-constrained drone systems?

**Analysis:**
Precision agriculture drones operate under severe energy constraints:

1. **Computational Cost Analysis:**
   - Drone battery capacity: 20-40 MJ per charge
   - Baseline model: ~0.5ms inference, 2W power draw
   - MLP model: ~5ms inference, 8W power draw (4x slower, 4x more power)
   
2. **Marginal Improvement Trade-off:**
   - Accuracy improvement: +2-5% (Domain 2 results show this range)
   - Battery life reduction: 4x (25% detection coverage per charge)
   - Field coverage: Reduced from 500 hectares/charge to 125 hectares/charge
   
3. **Deployment Suitability Assessment:**
   - If accuracy improves >10%: Deploy MLP with edge optimization (quantization, pruning)
   - If accuracy improves <5%: Deploy baseline model + frequent model retraining
   - Hybrid approach: Baseline model for real-time decisions, MLP for end-of-day batch processing
   
4. **Optimization Strategies:**
   - Model quantization: Reduce precision from float32 to int8 (90% size reduction)
   - Knowledge distillation: Train small baseline to approximate MLP predictions
   - Hardware acceleration: Use drone's GPU/TPU if available
   
**Practical Recommendation:**
For marginal improvements (<5%), deploy baseline model. Revisit MLP only if accuracy gains 
exceed 10% or if drones are upgraded with more compute resources.

---

### DOMAIN 3: Database Administration - Infrastructure Monitoring
**Situated Learning Question:** How do preprocessing decisions (missing value handling, 
normalization, feature scaling) influence predictive reliability and model generalization?

**Analysis:**
Database system predictions have severe consequences for financial institutions:

1. **Missing Value Handling Impact:**
   - Mean imputation: Introduces bias for non-random missing data
   - Forward fill: Creates temporal correlation artifacts in time-series data
   - Deletion: Loses potentially critical failure patterns
   - Domain-specific: Server metrics down = high failure risk (shouldn't be mean-filled)
   - Recommendation: Use context-aware imputation based on business logic
   
2. **Feature Normalization Impact:**
   - StandardScaler (z-score): Assumes Gaussian distribution
     - Problem: Server metrics often have outliers (sudden spikes indicate failures)
     - Solution: Use RobustScaler to handle outliers
   - MinMaxScaler: Maps to [0,1], sensitive to future outliers
     - Problem: Future production peaks exceed training range
     - Solution: Use domain-specific bounds (e.g., CPU max=100%)
   
3. **Feature Scaling Effects on Generalization:**
   - Unscaled features: High-magnitude features dominate gradients
     - Network utilization (GB) dominates query time (ms) in loss computation
     - Model learns spurious correlations
   - Properly scaled: Equal feature importance during training
     - Improves convergence speed (10x faster)
     - Better generalization to out-of-distribution server loads
   
4. **Production Reliability Impact:**
   - Scaling inconsistency: Training data scaled differently than production data
     - Results in model degradation over time
     - Systematic bias grows as production patterns drift
   - Solution: Store scaler parameters, apply identically to new data
   
**Practical Recommendation:**
1. Use domain-specific imputation (forward-fill for continuous streams)
2. Apply RobustScaler for outlier resistance
3. Persist scaler objects with model artifacts
4. Monitor feature distributions in production for drift detection

---

### DOMAIN 4: Healthcare - Patient Readmission Prediction
**Situated Learning Question:** Within clinical prediction systems, False Negatives may lead 
to delayed treatment and increased medical risk. Why is Recall a more appropriate metric 
than Accuracy?

**Analysis:**
Clinical decision systems operate under asymmetric cost functions:

1. **Asymmetric Error Costs:**
   - False Negative (predict not readmitted, but patient is): CRITICAL
     - No preventive intervention provided
     - Patient arrives at hospital unprepared
     - Increased mortality risk, extended hospital stay
     - Cost: $50,000+ (intensive care + complications)
   
   - False Positive (predict readmitted, actually not): ACCEPTABLE
     - Extra preventive care provided unnecessarily
     - Patient receives additional monitoring/follow-ups
     - Risk: Slightly elevated costs, minimal patient harm
     - Cost: $5,000 (extra office visit + monitoring)
   
2. **Why Accuracy is Inappropriate:**
   - Accuracy = (TP + TN) / All predictions
   - In imbalanced data (e.g., 80% non-readmitted):
     - Model predicting "never readmitted" achieves 80% accuracy
     - But catches 0% of readmission risk patients
     - Unsafe for clinical use despite high accuracy
   
3. **Why Recall is Superior:**
   - Recall = TP / (TP + FN) = "Of actual readmissions, how many did we catch?"
   - 95% Recall means: Catch 95% of patients who will be readmitted
   - Safe threshold for clinical intervention: Recall ≥ 0.95
   - Acceptable to have some false alarms (low precision) to save lives
   
4. **Practical Implementation:**
   - Set decision threshold for classification:
     - Default (0.5 threshold): May achieve 75% recall
     - Lower to 0.3 threshold: Achieve 95% recall (more false positives)
     - Choose threshold based on clinical capacity vs risk tolerance
   
5. **Precision-Recall Trade-off:**
   - Recall = 95% + Precision = 60%: Worth it (catch most, accept false alarms)
   - Recall = 70% + Precision = 95%: Risky (miss 30% of actual readmissions)
   
**Practical Recommendation:**
- Optimize for Recall ≥ 0.95 to minimize false negatives
- Use F1-score only as secondary metric for balanced cases
- Implement model auditing to catch systematic prediction failures
- Integrate with clinical workflow for manual review of borderline cases

---

### DOMAIN 5: FinTech - Loan Default Prediction
**Situated Learning Question:** How do preprocessing techniques (feature normalization, 
dataset imbalance) contribute to biased prediction outcomes across demographic groups?

**Analysis:**
Algorithmic bias in loan approval has severe ethical and legal consequences:

1. **Dataset Imbalance and Demographic Bias:**
   - Imbalanced data: 75% non-default, 25% default (typical)
   - If minority demographic has higher default rate:
     - Model learns spurious correlation: "demographic → higher risk"
     - Model becomes systematically biased against that group
   - Example: If women have 30% default rate vs 20% for men in training data:
     - Model may overpredict default for women (discrimination)
     - Even without explicitly using gender as feature
   
2. **Feature Normalization Contributing to Bias:**
   - StandardScaler uses global mean/std from training data
   - If training data is demographically skewed:
     - Income scaling: Uses mean from 80% majority demographic
     - Minority group income scaled differently (larger z-scores)
     - Creates systematic bias in model decisions
   
3. **Feature Engineering Bias:**
   - "Employment Years" feature:
     - Women may have lower average (due to career breaks)
     - Normalization penalizes women more heavily
     - Model learns unfair correlation
   
   - Credit score features:
     - Minorities may have lower scores (historical discrimination in lending)
     - Feature scaling amplifies this historical bias
     - Models perpetuate systemic inequality
   
4. **Manifestation Mechanisms:**
   - Feature scaling creates non-linear bias:
     - Minority group features transformed to extreme values
     - Neural network assigns different weights to extreme inputs
     - Result: Amplified discrimination in model predictions
   
5. **Bias Mitigation Strategies:**
   a) **Data Collection:**
      - Ensure balanced training data across demographics
      - Remove features that are proxies for protected characteristics
      - Audit for historical discrimination in labels
   
   b) **Preprocessing:**
      - Use stratified scaling within demographic groups
      - Apply fairness constraints during normalization
      - Document biased features before removal
   
   c) **Model Training:**
      - Monitor performance separately for each demographic group
      - Retrain with fairness-aware loss functions
      - Implement adversarial debiasing techniques
   
   d) **Deployment:**
      - Regular fairness audits (quarterly)
      - Detect disparate impact (statistical parity tests)
      - Override model decisions when bias detected
   
6. **Legal and Ethical Requirements:**
   - Equal Credit Opportunity Act (ECOA): Prohibits lending discrimination
   - Fair Housing Act: Extends to mortgage lending
   - Disparate Impact Doctrine: Even unintentional discrimination is illegal
   - Regulatory requirement: Document bias mitigation efforts
   
**Practical Recommendation:**
1. Conduct demographic parity analysis before deployment
2. Use group-aware preprocessing (stratified scaling per demographic)
3. Monitor fairness metrics monthly (false positive rate parity)
4. Remove proxies for protected characteristics
5. Implement human review layer for borderline cases
6. Document all bias mitigation efforts for regulatory compliance

---

## SUMMARY OF KEY INSIGHTS

1. **Metric Selection is Context-Dependent:**
   - Moderation: Prioritize Recall (minimize missed harmful content)
   - Healthcare: Prioritize Recall (minimize missed risks)
   - Finance: Prioritize Fairness (prevent discrimination)
   - General: Use domain-specific metrics, not generic accuracy

2. **Preprocessing Impact on Reliability:**
   - Proper scaling ensures consistent model behavior across data distributions
   - Biased scaling can amplify historical inequalities
   - Domain-specific preprocessing beats one-size-fits-all approaches

3. **Model Complexity Trade-offs:**
   - Marginal accuracy gains don't justify 10x computational costs
   - Optimize for deployment constraints (energy, latency, memory)
   - Baseline models often sufficient when properly tuned

4. **Bias and Fairness:**
   - Technical decisions (scaling, feature engineering) have ethical consequences
   - Algorithmic fairness requires deliberate intervention, not accident
   - Regulatory compliance demands systematic bias monitoring

"""
    
    print(analysis)
    print("\n" + "="*100)


def get_domain_recommendations():
    """Return dictionary of domain-specific recommendations"""
    
    recommendations = {
        'nlp': {
            'title': 'E-Commerce Review Moderation',
            'key_finding': 'Recall prioritized over Precision to minimize legal liability from missed harmful content',
            'recommendation': 'Recall ≥ 0.95 to ensure safe content moderation'
        },
        'agriculture': {
            'title': 'Precision Agriculture Monitoring',
            'key_finding': 'Marginal accuracy improvements (<5%) do not justify 4x computational overhead for drone systems',
            'recommendation': 'Deploy baseline model with edge optimization unless accuracy improvement exceeds 10%'
        },
        'database': {
            'title': 'Infrastructure Failure Prediction',
            'key_finding': 'Preprocessing decisions significantly impact model reliability; RobustScaler better than StandardScaler for outlier handling',
            'recommendation': 'Use domain-specific imputation and scale persistence for production consistency'
        },
        'healthcare': {
            'title': 'Patient Readmission Prediction',
            'key_finding': 'Recall is superior to Accuracy for clinical safety; minimizing false negatives prevents treatment delays',
            'recommendation': 'Optimize for Recall ≥ 0.95 despite potential precision reduction'
        },
        'fintech': {
            'title': 'Loan Default Risk Assessment',
            'key_finding': 'Preprocessing normalization can amplify demographic bias; stratified scaling required for fairness',
            'recommendation': 'Implement group-aware preprocessing and monthly fairness audits for regulatory compliance'
        }
    }
    
    return recommendations
