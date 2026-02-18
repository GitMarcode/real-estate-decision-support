"""
Pareto optimization module for identifying non-dominated solutions.
"""

import pandas as pd
import numpy as np
from typing import List, Set, Dict


class ParetoOptimizer:
    """
    Pareto optimizer for multi-criteria property selection.
    """
    
    def __init__(self, criteria: List[str], maximize: List[bool]):
        """
        Initialize Pareto optimizer.
        
        Args:
            criteria: List of criteria column names
            maximize: List of booleans indicating if each criterion should be maximized
        """
        self.criteria = criteria
        self.maximize = maximize
    
    def dominates(self, 
                  solution_a: pd.Series, 
                  solution_b: pd.Series) -> bool:
        """
        Check if solution A dominates solution B.
        
        A dominates B if:
        - A is better or equal on all criteria
        - A is strictly better on at least one criterion
        
        Args:
            solution_a: First solution
            solution_b: Second solution
            
        Returns:
            True if A dominates B
        """
        better_count = 0
        worse_count = 0
        
        for i, criterion in enumerate(self.criteria):
            val_a = solution_a[criterion]
            val_b = solution_b[criterion]
            
            if self.maximize[i]:
                # For maximization criteria
                if val_a > val_b:
                    better_count += 1
                elif val_a < val_b:
                    worse_count += 1
            else:
                # For minimization criteria
                if val_a < val_b:
                    better_count += 1
                elif val_a > val_b:
                    worse_count += 1
        
        # A dominates B if A is better on at least one criterion and not worse on any
        return better_count > 0 and worse_count == 0
    
    def find_pareto_front(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Find Pareto-efficient solutions (non-dominated solutions).
        
        Args:
            df: DataFrame with alternatives and criteria
            
        Returns:
            DataFrame with only Pareto-efficient solutions
        """
        print("Finding Pareto front...")
        
        n = len(df)
        is_pareto = np.ones(n, dtype=bool)
        
        for i in range(n):
            if not is_pareto[i]:
                continue
                
            for j in range(n):
                if i == j or not is_pareto[j]:
                    continue
                
                # Check if j dominates i
                if self.dominates(df.iloc[j], df.iloc[i]):
                    is_pareto[i] = False
                    break
        
        pareto_df = df[is_pareto].copy()
        pareto_df['pareto_efficient'] = True
        
        print(f"Found {len(pareto_df)} Pareto-efficient solutions out of {n} total")
        return pareto_df
    
    def identify_robust_core(self, 
                            df: pd.DataFrame,
                            min_score_threshold: float = 0.8) -> pd.DataFrame:
        """
        Identify robust core - solutions in Pareto front with high scores.
        
        Args:
            df: DataFrame with Pareto-efficient solutions
            min_score_threshold: Minimum normalized score to be in robust core
            
        Returns:
            DataFrame with robust core solutions
        """
        print("Identifying robust core...")
        
        # Calculate aggregate score for each solution
        scores = []
        for idx, row in df.iterrows():
            score = 0
            for i, criterion in enumerate(self.criteria):
                val = row[criterion]
                
                # Normalize to [0, 1]
                min_val = df[criterion].min()
                max_val = df[criterion].max()
                
                if max_val - min_val > 0:
                    norm_val = (val - min_val) / (max_val - min_val)
                else:
                    norm_val = 0.5
                
                # Invert if minimization criterion
                if not self.maximize[i]:
                    norm_val = 1 - norm_val
                
                score += norm_val
            
            scores.append(score / len(self.criteria))
        
        df['aggregate_score'] = scores
        
        # Filter for robust core
        robust_core = df[df['aggregate_score'] >= min_score_threshold].copy()
        robust_core = robust_core.sort_values('aggregate_score', ascending=False)
        
        print(f"Robust core contains {len(robust_core)} solutions")
        return robust_core
    
    def optimize(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Complete Pareto optimization pipeline.
        
        Args:
            df: DataFrame with all alternatives
            
        Returns:
            Dictionary with 'pareto_front' and 'robust_core' DataFrames
        """
        print("Starting Pareto optimization...")
        
        # Find Pareto front
        pareto_front = self.find_pareto_front(df)
        
        # Identify robust core
        robust_core = self.identify_robust_core(pareto_front)
        
        print("Pareto optimization complete")
        
        return {
            'pareto_front': pareto_front,
            'robust_core': robust_core
        }


def run_pareto_optimization(input_path: str, 
                           output_path_front: str,
                           output_path_core: str,
                           criteria: List[str],
                           maximize: List[bool]) -> Dict[str, pd.DataFrame]:
    """
    Main function to run Pareto optimization.
    
    Args:
        input_path: Path to input data CSV
        output_path_front: Path to save Pareto front
        output_path_core: Path to save robust core
        criteria: List of criteria to optimize
        maximize: List indicating if each criterion should be maximized
        
    Returns:
        Dictionary with optimization results
    """
    # Load data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Initialize optimizer
    optimizer = ParetoOptimizer(criteria, maximize)
    
    # Run optimization
    results = optimizer.optimize(df)
    
    # Save results
    results['pareto_front'].to_csv(output_path_front, index=False)
    print(f"Pareto front saved to {output_path_front}")
    
    results['robust_core'].to_csv(output_path_core, index=False)
    print(f"Robust core saved to {output_path_core}")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Pareto optimization module ready")
