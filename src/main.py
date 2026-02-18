"""
Main pipeline for the Real Estate Investment Decision Support System.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_collection import collect_data
from src.preprocessing import PropertyDataPreprocessor
from src.electre import ELECTREIII
from src.pareto import ParetoOptimizer


def create_sample_data() -> pd.DataFrame:
    """
    Create sample property data for demonstration.
    
    Returns:
        DataFrame with sample property data
    """
    np.random.seed(42)
    n_properties = 100
    
    data = {
        'id': range(1, n_properties + 1),
        'price': np.random.uniform(150000, 500000, n_properties),
        'surface': np.random.uniform(30, 120, n_properties),
        'monthly_rent': np.random.uniform(800, 2500, n_properties),
        'municipality': np.random.choice(['Paris', 'Boulogne', 'Neuilly', 'Versailles'], n_properties),
        'department': ['75'] * n_properties
    }
    
    return pd.DataFrame(data)


def run_pipeline(use_sample_data: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Run the complete analysis pipeline.
    
    Args:
        use_sample_data: If True, use sample data; if False, collect from APIs
        
    Returns:
        Dictionary with analysis results
    """
    print("=" * 60)
    print("Real Estate Investment Decision Support System")
    print("=" * 60)
    print()
    
    # Step 1: Data Collection
    print("Step 1: Data Collection")
    print("-" * 60)
    
    if use_sample_data:
        print("Using sample data for demonstration...")
        df = create_sample_data()
    else:
        df = collect_data(save_path="data/raw/properties.csv")
    
    print(f"Loaded {len(df)} properties")
    print()
    
    # Step 2: Preprocessing
    print("Step 2: Data Preprocessing")
    print("-" * 60)
    
    preprocessor = PropertyDataPreprocessor()
    df_processed = preprocessor.preprocess(df)
    
    # Save processed data
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df_processed.to_csv("data/processed/properties_processed.csv", index=False)
    print()
    
    # Step 3: Define criteria and weights
    print("Step 3: Defining Criteria")
    print("-" * 60)
    
    criteria = [
        'price',
        'price_per_sqm',
        'monthly_rent',
        'rental_yield',
        'annual_cash_flow',
        'roi',
        'surface'
    ]
    
    # Filter to only properties with all criteria
    df_analysis = df_processed.dropna(subset=criteria).copy()
    print(f"Properties with complete data: {len(df_analysis)}")
    
    weights = {
        'price': 0.15,              # Lower price is better
        'price_per_sqm': 0.10,      # Lower price/m² is better
        'monthly_rent': 0.15,        # Higher rent is better
        'rental_yield': 0.20,        # Higher yield is better
        'annual_cash_flow': 0.20,    # Higher cash flow is better
        'roi': 0.15,                 # Higher ROI is better
        'surface': 0.05              # Larger surface is better
    }
    
    maximize = {
        'price': False,
        'price_per_sqm': False,
        'monthly_rent': True,
        'rental_yield': True,
        'annual_cash_flow': True,
        'roi': True,
        'surface': True
    }
    
    print(f"Criteria: {', '.join(criteria)}")
    print()
    
    # Step 4: ELECTRE III Analysis
    print("Step 4: ELECTRE III Analysis")
    print("-" * 60)
    
    electre = ELECTREIII(
        criteria=criteria,
        weights=weights,
        maximize=maximize
    )
    
    df_ranked = electre.rank_alternatives(df_analysis)
    df_ranked.to_csv("data/processed/electre_ranked.csv", index=False)
    print()
    
    # Step 5: Pareto Optimization
    print("Step 5: Pareto Optimization")
    print("-" * 60)
    
    maximize_list = [maximize[c] for c in criteria]
    
    optimizer = ParetoOptimizer(
        criteria=criteria,
        maximize=maximize_list
    )
    
    results = optimizer.optimize(df_ranked)
    
    # Save results
    results['pareto_front'].to_csv("data/processed/pareto_front.csv", index=False)
    results['robust_core'].to_csv("data/processed/robust_core.csv", index=False)
    print()
    
    # Step 6: Display Results
    print("Step 6: Results Summary")
    print("-" * 60)
    print()
    
    print(f"Total properties analyzed: {len(df)}")
    print(f"Properties after preprocessing: {len(df_processed)}")
    print(f"Properties in analysis: {len(df_analysis)}")
    print(f"Pareto-efficient properties: {len(results['pareto_front'])}")
    print(f"Robust core properties: {len(results['robust_core'])}")
    print()
    
    if len(results['robust_core']) > 0:
        print("Top 3 Investment Opportunities:")
        print("-" * 60)
        
        top_properties = results['robust_core'].head(3)
        
        for idx, (i, row) in enumerate(top_properties.iterrows(), 1):
            print(f"\nProperty {idx}:")
            print(f"  Price: €{row['price']:,.0f}")
            print(f"  Surface: {row['surface']:.1f} m²")
            print(f"  Price/m²: €{row['price_per_sqm']:,.0f}")
            print(f"  Monthly Rent: €{row['monthly_rent']:,.0f}")
            print(f"  Rental Yield: {row['rental_yield']:.2f}%")
            print(f"  Annual Cash Flow: €{row['annual_cash_flow']:,.0f}")
            print(f"  ROI: {row['roi']:.2f}%")
            print(f"  Aggregate Score: {row['aggregate_score']:.3f}")
    
    print()
    print("=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    # Run the complete pipeline
    results = run_pipeline(use_sample_data=True)
    
    print("\nResults saved to data/processed/")
    print("- electre_ranked.csv: All properties ranked by ELECTRE III")
    print("- pareto_front.csv: Pareto-efficient properties")
    print("- robust_core.csv: Optimal investment opportunities")
