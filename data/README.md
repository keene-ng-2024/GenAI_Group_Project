# Dataset Preparation and Reviewer Agreement Metrics

## Overview
This directory contains the prepared dataset splits and baseline metrics for the AI Research Paper Critique project, fulfilling the Phase 1 responsibilities of the Data & Dataset Lead.

## Dataset Splits
* **Development Set (`dev_split.jsonl`):** Contains 5 papers used for active system development and prompt tuning.
* **Evaluation Set (`eval_split.jsonl`):** Contains 15 papers reserved for the final, unbiased evaluation of the multi-agent system.

## Reviewer Agreement Statistics
To establish a quantitative baseline for human variance, we computed the reviewer agreement based on the "Recommendation" score provided by multiple human reviewers per paper.

* **Average Reviewer Score Variance:** 2.04
* **Standard Deviation:** 1.43 points

### Formulas
The sample variance for each paper's scores was calculated using the following mathematical formulation:
$$s^2 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n - 1}$$
Where $x_i$ is each reviewer's score, $\bar{x}$ is the average score for that paper, and $n$ is the number of reviewers.

The standard deviation is the square root of the variance:
$$s = \sqrt{s^2}$$

### Interpretation and Application
A variance of 2.04 translates to a standard deviation of approximately 1.43. This indicates that human reviewers evaluating the exact same research paper typically disagree by about 1.43 points on their final recommendation. 

During Phase 3 (Evaluation & Experiments), when the multi-agent system generates its outputs, the system will calculate a reviewer similarity score. This 1.43-point margin establishes the statistical threshold for human-level consensus. If the AI's scores fall within this standard deviation from the human average, the system's performance is demonstrably aligned with expert reviewers.
