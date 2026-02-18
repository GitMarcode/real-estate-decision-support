"""
Data preprocessing module for cleaning and preparing property data.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from src.utils import calculate_rental_yield, calculate_roi, calculate_price_per_sqm


class PropertyDataPreprocessor:
    """
    Preprocessor for property data cleaning and feature engineering.
    """
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.feature_columns = []
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw property data.
        
        Args:
            df: Raw property DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        print("Cleaning data...")
        df_clean = df.copy()
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        # Remove rows with missing critical values
        critical_columns = ['price', 'surface']
        df_clean = df_clean.dropna(subset=critical_columns)
        
        # Remove outliers (e.g., price = 0 or negative)
        df_clean = df_clean[df_clean['price'] > 0]
        df_clean = df_clean[df_clean['surface'] > 0]
        
        print(f"Cleaned data: {len(df_clean)} properties remaining")
        return df_clean
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features from existing data.
        
        Args:
            df: Cleaned property DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        print("Engineering features...")
        df_features = df.copy()
        
        # Calculate price per square meter
        if 'price' in df.columns and 'surface' in df.columns:
            df_features['price_per_sqm'] = df_features.apply(
                lambda row: calculate_price_per_sqm(row['price'], row['surface']), 
                axis=1
            )
        
        # Calculate rental yield if rent data is available
        if 'monthly_rent' in df.columns and 'price' in df.columns:
            df_features['rental_yield'] = df_features.apply(
                lambda row: calculate_rental_yield(row['price'], row['monthly_rent']), 
                axis=1
            )
            
            # Calculate annual cash flow (simplified)
            df_features['annual_cash_flow'] = df_features['monthly_rent'] * 12
            
            # Calculate ROI
            df_features['roi'] = df_features.apply(
                lambda row: calculate_roi(row['price'], row['annual_cash_flow']), 
                axis=1
            )
        
        # Location score (placeholder - would be based on actual location data)
        if 'municipality' in df.columns:
            df_features['location_score'] = np.random.uniform(0.5, 1.0, len(df_features))
        
        print(f"Feature engineering complete. Total features: {len(df_features.columns)}")
        return df_features
    
    def normalize_data(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Normalize specified columns to [0, 1] range.
        
        Args:
            df: DataFrame to normalize
            columns: List of columns to normalize (None = all numeric columns)
            
        Returns:
            DataFrame with normalized columns
        """
        print("Normalizing data...")
        df_norm = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col in df_norm.columns:
                min_val = df_norm[col].min()
                max_val = df_norm[col].max()
                
                if max_val - min_val > 0:
                    df_norm[col + '_normalized'] = (df_norm[col] - min_val) / (max_val - min_val)
        
        print(f"Normalization complete")
        return df_norm
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Raw property DataFrame
            
        Returns:
            Fully preprocessed DataFrame
        """
        print("Starting preprocessing pipeline...")
        
        # Clean data
        df = self.clean_data(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        print("Preprocessing complete")
        return df


def preprocess_data(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Main preprocessing function.
    
    Args:
        input_path: Path to raw data CSV
        output_path: Path to save processed data
        
    Returns:
        Preprocessed DataFrame
    """
    # Load data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} records")
    
    # Initialize preprocessor
    preprocessor = PropertyDataPreprocessor()
    
    # Preprocess
    df_processed = preprocessor.preprocess(df)
    
    # Save
    df_processed.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    
    return df_processed


if __name__ == "__main__":
    # Example usage
    preprocess_data(
        input_path="data/raw/properties.csv",
        output_path="data/processed/properties_processed.csv"
    )
