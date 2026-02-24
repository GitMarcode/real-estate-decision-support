# Project Summary — Real Estate Investment Decision Support (Paris)

## Objective
The objective of this project is to identify the most promising rental real estate investment opportunities in the Paris region using a multi-criteria decision-making approach.  
Investment profitability cannot be evaluated using a single metric, so we combine several financial indicators.

## Methodology

1. Data collection from French open data:
   - DVF (property transaction data)
   - DHUP (rental data)

2. Data preprocessing and feature engineering:
   - ROI (Return on Investment)
   - Annual cash flow
   - Price per square meter
   - Rental yield
   - Other financial indicators

3. Pareto optimization:
   - Eliminate dominated properties
   - Keep only efficient investment candidates

4. ELECTRE III outranking method:
   - Model preferences with thresholds
   - Identify a robust core of top-performing properties

## Dataset

- 11,565 properties analyzed
- 8 financial decision criteria
- Focus: Paris region

## Key Results

The analysis identified a robust core of 3 optimal properties with:

- ROI > 24%
- Annual cash flow > €33,000
- Price < €1,600 per m²

## Tools & Technologies

- Python
- Jupyter Notebook
- NumPy / Pandas
- Multi-criteria decision analysis (MCDA)

## Next Steps

- Sensitivity analysis on ELECTRE thresholds and weights
- Uncertainty modeling (rent variability, vacancy rate, renovation costs)
- Development of a Streamlit web app for interactive exploration

## Author

Marwane Bennat  & Alexis Bessy
Master 1 Data Science  
Université Paris Dauphine–PSL (2025–2026)
