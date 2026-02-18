"""
Utility functions for the real estate decision support system.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any


def calculate_rental_yield(price: float, monthly_rent: float) -> float:
    """
    Calculate rental yield percentage.
    
    Args:
        price: Property purchase price
        monthly_rent: Monthly rental income
        
    Returns:
        Rental yield as a percentage
    """
    if price == 0:
        return 0.0
    annual_rent = monthly_rent * 12
    return (annual_rent / price) * 100


def calculate_roi(price: float, annual_cash_flow: float) -> float:
    """
    Calculate Return on Investment (ROI).
    
    Args:
        price: Property purchase price
        annual_cash_flow: Annual cash flow
        
    Returns:
        ROI as a percentage
    """
    if price == 0:
        return 0.0
    return (annual_cash_flow / price) * 100


def calculate_price_per_sqm(price: float, surface: float) -> float:
    """
    Calculate price per square meter.
    
    Args:
        price: Property purchase price
        surface: Property surface area in m²
        
    Returns:
        Price per square meter
    """
    if surface == 0:
        return 0.0
    return price / surface


def normalize_criteria(df: pd.DataFrame, criteria: List[str]) -> pd.DataFrame:
    """
    Normalize criteria values to [0, 1] range.
    
    Args:
        df: DataFrame with criteria columns
        criteria: List of column names to normalize
        
    Returns:
        DataFrame with normalized criteria
    """
    df_normalized = df.copy()
    
    for criterion in criteria:
        if criterion in df.columns:
            min_val = df[criterion].min()
            max_val = df[criterion].max()
            
            if max_val - min_val > 0:
                df_normalized[criterion] = (df[criterion] - min_val) / (max_val - min_val)
            else:
                df_normalized[criterion] = 0.0
    
    return df_normalized


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with loaded data
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {filepath}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def save_data(df: pd.DataFrame, filepath: str) -> None:
    """
    Save DataFrame to CSV file.
    
    Args:
        df: DataFrame to save
        filepath: Path where to save the file
    """
    try:
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
    except Exception as e:
        raise Exception(f"Error saving data: {str(e)}")


def validate_property_data(df: pd.DataFrame) -> bool:
    """
    Validate property data has required columns.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_columns = ['price', 'surface', 'monthly_rent']
    
    for col in required_columns:
        if col not in df.columns:
            print(f"Missing required column: {col}")
            return False
    
    return True
