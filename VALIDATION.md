# Validation and reproducibility

## Automated checks

Run `python -m unittest discover -s tests -v` from the project folder.

Six checks cover future-data invariance, exclusive label endpoints, incomplete observation windows, cancellation filtering, budget metric arithmetic, and exclusion of customers with no past history.

## Evaluation protocol

Model choice is based exclusively on July–August 2011 validation precision at the predeclared 20% contact capacity. The September–November 2011 holdout is used once for final reporting. The saved estimator has been refit on train + validation, not on the holdout. Feature scaling is fit inside the logistic regression pipeline on training data only. Customer IDs, future transactions and targets are excluded from predictors.

Customer overlap between periods is intentional for an existing-customer repeat-purchase task. Results do not measure acquisition of previously unseen customers. Features cover only the preceding 180 days; frequency is distinct invoice count rather than invoice-line count. Every eligible customer receives a label; customers without a later qualifying purchase are negatives, including customers who leave the retailer.

Monthly contact counts are rounded upward. Metrics are averaged equally across test months. Random expected precision equals each month's buyer prevalence. Random realised results use seed 42. Tie-breaking is stable customer-ID order. These definitions make the business comparisons reproducible.

## Interpretation

The 20% contact-capacity scenario is an analytical assumption, not a retailer-supplied budget. The output identifies likely buyers; causal uplift and monetary ROI cannot be measured from these transactions. Test months cover only one autumn period. Robustness across industries, countries and later years is not established. No claimed improvement should be reported without consulting the measured baseline results.
