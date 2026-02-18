# 🏠 Real Estate Investment Decision Support System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Multi-criteria decision analysis tool for identifying optimal rental property investments in the Paris region using ELECTRE III methodology.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Data Sources](#data-sources)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies](#technologies)
- [Author](#author)
- [License](#license)

---

## 🎯 Overview

This project analyzes **11,565 properties** in Île-de-France to identify optimal investment opportunities for individual real estate investors. Using multi-criteria decision analysis (ELECTRE III) combined with Pareto optimization, the system evaluates properties across 8 financial criteria and recommends the most robust investment choices.

**Academic Context**: Master 1 Data Science project, Université Paris Dauphine-PSL (2025-2026)

---

## ✨ Features

- 📊 **Large-scale analysis**: 11,565 properties analyzed
- 🔍 **Multi-criteria evaluation**: 8 financial metrics
- 🎯 **ELECTRE III algorithm**: Robust outranking method
- 📈 **Pareto filtering**: Identification of non-dominated solutions
- 🗺️ **Geographic focus**: Île-de-France region
- 💰 **Financial optimization**: ROI, cash flow, rental yield
- 🔗 **API integration**: Automated data extraction (DVF, DHUP)

---

## 📂 Data Sources

1. **DVF (Demandes de Valeurs Foncières)**
   - French notary transaction data
   - Property sales records
   - Source: data.gouv.fr

2. **DHUP (Direction de l'Habitat, de l'Urbanisme et des Paysages)**
   - Rental price data
   - Geographic rental market information

---

## 🧮 Methodology

### 1. Data Collection & Preprocessing
- API extraction from government databases
- Data cleaning and normalization
- Feature engineering (price/m², rental yield calculation)

### 2. Criteria Definition
Evaluated across 8 financial metrics:
- Purchase price
- Price per m²
- Estimated monthly rent
- Rental yield (%)
- Gross annual cash flow
- Annual ROI (%)
- Property size
- Location score

### 3. ELECTRE III Analysis
- Pairwise comparison of alternatives
- Concordance and discordance indices
- Outranking relations
- Sensitivity analysis

### 4. Pareto Optimization
- Identification of Pareto-efficient properties
- Filtering of dominated solutions
- Robust core selection

---

## 📊 Results

### Optimal Investment Opportunities

The analysis identified **3 optimal properties** in the robust core:

| Metric | Property A | Property B | Property C |
|--------|-----------|-----------|-----------|
| **Price** | €XXX,XXX | €XXX,XXX | €XXX,XXX |
| **ROI** | >24% | >24% | >24% |
| **Cash Flow** | >€33k/year | >€33k/year | >€33k/year |
| **Price/m²** | <€1,600 | <€1,600 | <€1,600 |

*(Exact values available in analysis notebook)*

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone repository
git clone https://github.com/GitMarcode/real-estate-decision-support.git
cd real-estate-decision-support

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage

### Running the Analysis

```bash
# Run complete analysis pipeline
python src/main.py

# Run specific modules
python src/data_collection.py  # Data extraction
python src/electre.py          # ELECTRE III analysis
python src/pareto.py           # Pareto optimization
```

### Jupyter Notebook

```bash
# Launch interactive analysis
jupyter notebook notebooks/analysis.ipynb
```

---

## 📁 Project Structure

```
real-estate-decision-support/
├── src/
│   ├── __init__.py
│   ├── main.py              # Main pipeline
│   ├── data_collection.py   # API data extraction
│   ├── preprocessing.py     # Data cleaning
│   ├── electre.py           # ELECTRE III implementation
│   ├── pareto.py            # Pareto optimization
│   └── utils.py             # Helper functions
├── notebooks/
│   └── analysis.ipynb       # Interactive analysis
├── data/
│   ├── raw/                 # Raw data
│   └── processed/           # Processed data
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🛠️ Technologies

- **Python 3.8+**: Core language
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **matplotlib/seaborn**: Data visualization
- **requests**: API integration
- **jupyter**: Interactive analysis

---

## 👤 Author

**GitMarcode**
- GitHub: [@GitMarcode](https://github.com/GitMarcode)
- Academic: Université Paris Dauphine-PSL

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
