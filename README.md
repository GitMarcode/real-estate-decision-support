# Real Estate Investment Decision Support (Paris) — ELECTRE III + Pareto

Multi-criteria decision analysis (MCDA) tool to identify **robust rental property investments** in the Paris region using **ELECTRE III** and **Pareto optimization** on **11,565 properties**.

## Problem
How to rank and select real-estate investment opportunities when profitability depends on multiple conflicting criteria (ROI, cash flow, price/m², yield, etc.), not a single metric.

## Method
- **Data ingestion** from French open data:
  - DVF (land registry / transactions)
  - DHUP (rent data)
- **Preprocessing & feature engineering** for investment metrics (ROI, cash flow, yield, price/m², …)
- **Pareto filtering** to remove clearly dominated options
- **ELECTRE III** (outranking) to obtain a robust core of best candidates under preference/threshold modeling

## Data
- **Scale**: 11,565 properties
- **Criteria**: 8 financial metrics (ROI, cash flow, price/m², rental yield, etc.)
- **Sources**: French government open data (DVF + DHUP)

> Note: if the dataset is not included in the repository, the notebook reproduces the pipeline from the APIs.

## Results (high level)
Identified a robust core of **3 optimal properties** with:
- **ROI**: > 24%
- **Annual cash flow**: > €33,000
- **Price point**: < €1,600/m²

(See the notebook for full details, plots and intermediate outputs.)

## Reproduce
### 1) Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
