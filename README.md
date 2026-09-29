# Reasoning Externalization for Faithful LLM Narratives of Stock Return Predictions

Official code repository for:

**Reasoning Externalization for Faithful Large Language Model Narratives of Stock Return Predictions**  
*LLM4XAI @ CIKM 2026*

## Overview

This study investigates how externalizing numerical and relational reasoning from Large Language Models (LLMs) can improve the faithfulness of natural-language explanations for stock return predictions.

Our framework combines:

- **XGBoost** for cross-sectional stock return prediction
- **TreeSHAP** for feature attribution
- **Temporal XAI evidence** based on six-month SHAP importance trajectories
- **Deterministic temporal descriptors** such as attribution changes, slopes, and trend directions
- **Feature-level relational evidence** such as OVERTAKES relations
- **Historical analog retrieval** based on similar SHAP importance dynamics
- **Evidence-grounded LLM narratives** with provenance-based verification

## Framework

The reasoning burden is progressively externalized through five experimental conditions:

| Condition | Evidence |
|---|---|
| C0 | Prediction, local SHAP, current global attribution |
| C1 | C0 + raw 6-month attribution sequence |
| C2 | C1 + temporal change, slope, and trend |
| C3 | C2 + deterministic feature relations |
| C3-Analog | C3 + historical analog evidence |

## Key Results

For **Qwen3-32B**:

| Condition | Evidence Faithfulness | Temporal Accuracy | Relational Accuracy |
|---|---:|---:|---:|
| C1-Raw | 0.696 | 0.340 | 0.525 |
| C2-Numerical | 0.903 | 0.990 | 0.573 |
| C3-Relational | 0.996 | 0.931 | 1.000 |
| C3-Analog | 0.989 | 0.900 | 1.000 |

The results show that externalizing error-prone numerical and relational reasoning into verifiable structured evidence substantially improves the reliability of LLM-generated financial XAI narratives.

## Data

The experiments use monthly stock data from the **Center for Research in Security Prices (CRSP)** accessed through **WRDS**.

The prediction task uses 30 stock-level characteristics to predict next-month stock returns.

## Environment

Main experimental libraries:

- Python
- XGBoost 3.2.0
- SHAP 0.51.0
- vLLM 0.11.2

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{kim2026reasoning,
  title={Reasoning Externalization for Faithful Large Language Model Narratives of Stock Return Predictions},
  author={Kim, Sujung and Cho, Seung Hwan and Park, Sangjin and Kim, Young-Min},
  booktitle={LLM4XAI @ CIKM 2026},
  year={2026}
}
