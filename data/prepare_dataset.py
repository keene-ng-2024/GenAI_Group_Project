import json
import random
import statistics
import math

def generate_readme(variance, std_dev, dev_count, eval_count):
    """Generates a README.md file explaining the dataset and metrics."""
    formula_variance = "$$s^2 = \\frac{\\sum_{i=1}^{n} (x_i - \\bar{x})^2}{n - 1}$$"
    formula_stddev = "$$s = \\sqrt{s^2}$$"
    
    readme_content = f"""# Dataset Preparation and Reviewer Agreement Metrics

## Overview
This directory contains the prepared dataset splits and baseline metrics for the AI Research Paper Critique project, fulfilling the Phase 1 responsibilities of the Data & Dataset Lead.

## Dataset Splits
* **Development Set (`dev_split.jsonl`):** Contains {dev_count} papers used for active system development and prompt tuning.
* **Evaluation Set (`eval_split.jsonl`):** Contains {eval_count} papers reserved for the final, unbiased evaluation of the multi-agent system.

## Reviewer Agreement Statistics
To establish a quantitative baseline for human variance, we computed the reviewer agreement based on the "Recommendation" score provided by multiple human reviewers per paper.

* **Average Reviewer Score Variance:** {variance:.2f}
* **Standard Deviation:** {std_dev:.2f} points

### Formulas
The sample variance for each paper's scores was calculated using the following mathematical formulation:
{formula_variance}
Where $x_i$ is each reviewer's score, $\\bar{{x}}$ is the average score for that paper, and $n$ is the number of reviewers.

The standard deviation is the square root of the variance:
{formula_stddev}

### Interpretation and Application
A variance of {variance:.2f} translates to a standard deviation of approximately {std_dev:.2f}. This indicates that human reviewers evaluating the exact same research paper typically disagree by about {std_dev:.2f} points on their final recommendation. 

During Phase 3 (Evaluation & Experiments), when the multi-agent system generates its outputs, the system will calculate a reviewer similarity score. This {std_dev:.2f}-point margin establishes the statistical threshold for human-level consensus. If the AI's scores fall within this standard deviation from the human average, the system's performance is demonstrably aligned with expert reviewers.
"""
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print("README.md successfully generated.")

def process_jsonl_dataset(input_filepath):
    """Parses the JSONL dataset, structures reviews, creates splits, and generates docs."""
    data = []
    
    with open(input_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                raw_record = json.loads(line)
                structured_record = {
                    "title": raw_record.get("title", ""),
                    "decision": raw_record.get("decision", ""),
                    "paper_text": raw_record.get("body_text", ""), 
                    "reviews": []
                }
                for key, value in raw_record.items():
                    if key.startswith('review#'):
                        structured_record["reviews"].append(value)
                data.append(structured_record)
    
    agreement_scores = []
    for paper in data:
        scores = []
        for review_container in paper['reviews']:
            if isinstance(review_container, dict) and 'score' in review_container:
                score_dict = review_container['score']
                if 'Recommendation' in score_dict:
                    try:
                        scores.append(float(score_dict['Recommendation'].strip()))
                    except ValueError:
                        pass
        if len(scores) > 1:
            agreement_scores.append(statistics.variance(scores))
    
    if agreement_scores:
        avg_variance = sum(agreement_scores) / len(agreement_scores)
        std_dev = math.sqrt(avg_variance)
        print(f"Average Reviewer Score Variance: {avg_variance:.2f}")
        print(f"Standard Deviation: {std_dev:.2f}")
    else:
        avg_variance, std_dev = 0, 0
        print("Could not compute numerical reviewer agreement.")

    random.shuffle(data)
    dev_split = data[:5]
    eval_split = data[5:20] 
    
    with open('dev_split.jsonl', 'w', encoding='utf-8') as f:
        for item in dev_split:
            f.write(json.dumps(item) + '\n')
            
    with open('eval_split.jsonl', 'w', encoding='utf-8') as f:
        for item in eval_split:
            f.write(json.dumps(item) + '\n')
            
    generate_readme(avg_variance, std_dev, len(dev_split), len(eval_split))

if __name__ == "__main__":
    process_jsonl_dataset('ReviewCritique.jsonl')