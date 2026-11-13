# Retail Repeat-Purchase Prediction — measured results

Selected model: **random_forest**, chosen using validation months only.

Raw transaction lines: 1,067,371; cleaned positive-sale lines: 779,425.

At 20% monthly contact capacity, held-out mean precision is **65.16%**, recall **38.39%**, and lift over random selection **1.92×**.

These are historical buyer-identification results, not demonstrated campaign-driven sales or profit. See budget_summary.csv for comparisons against simple baselines; an ML win is not assumed.

Training cutoffs: June 2010–June 2011. Validation: July–August 2011. Test: September–November 2011. Each label spans 30 days. Final models are refit on train + validation; test is never used for model selection. Customers may recur across dates because the task is predicting future behaviour of existing customers. Monthly scores are macro-averaged; no independent-customer confidence intervals are claimed.

Eligibility: a known customer with a positive purchase in the preceding 180 days. Excludes cancellations, nonpositive prices/quantities, missing required fields and exact duplicate lines. Returns are not netted against sales; spend is positive-sale value, not profit. A purchase is any positive qualifying invoice within the next 30 days. Dormant customers outside the lookback and anonymous customers are outside scope.

## Strongest simple baseline

Frequency-only selection reaches 64.68% mean precision at 20% capacity, versus 65.16% for random forest: approximately 0.48 percentage points difference. Statistical superiority is not established. This is a substantive management finding: a simple purchase-frequency rule is highly competitive.
