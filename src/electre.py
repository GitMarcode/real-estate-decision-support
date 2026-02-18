"""
ELECTRE III (Élimination Et Choix Traduisant la Réalité) implementation.
Multi-criteria decision analysis method for ranking alternatives.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

# Small constant to avoid division by zero
EPSILON = 1e-10


class ELECTREIII:
    """
    ELECTRE III algorithm implementation for property ranking.
    """
    
    def __init__(self, 
                 criteria: List[str],
                 weights: Dict[str, float],
                 preference_thresholds: Optional[Dict[str, float]] = None,
                 indifference_thresholds: Optional[Dict[str, float]] = None,
                 veto_thresholds: Optional[Dict[str, float]] = None,
                 maximize: Optional[Dict[str, bool]] = None):
        """
        Initialize ELECTRE III algorithm.
        
        Args:
            criteria: List of criterion names
            weights: Dictionary of criterion weights (sum should be 1)
            preference_thresholds: Preference thresholds for each criterion
            indifference_thresholds: Indifference thresholds for each criterion
            veto_thresholds: Veto thresholds for each criterion
            maximize: Dict indicating if criterion should be maximized (True) or minimized (False)
        """
        self.criteria = criteria
        self.weights = weights
        self.preference_thresholds = preference_thresholds or {}
        self.indifference_thresholds = indifference_thresholds or {}
        self.veto_thresholds = veto_thresholds or {}
        self.maximize = maximize or {c: True for c in criteria}
        
        # Validate weights sum to 1
        if not np.isclose(sum(weights.values()), 1.0):
            print(f"Warning: Weights sum to {sum(weights.values())}, normalizing...")
            total = sum(weights.values())
            self.weights = {k: v/total for k, v in weights.items()}
    
    def calculate_concordance(self, 
                             alt_a: pd.Series, 
                             alt_b: pd.Series) -> float:
        """
        Calculate concordance index for alternative a outranking b.
        
        Args:
            alt_a: Data for alternative a
            alt_b: Data for alternative b
            
        Returns:
            Concordance index [0, 1]
        """
        concordance = 0.0
        
        for criterion in self.criteria:
            weight = self.weights.get(criterion, 0)
            
            val_a = alt_a[criterion]
            val_b = alt_b[criterion]
            
            # Adjust for minimization criteria
            if not self.maximize.get(criterion, True):
                val_a, val_b = -val_a, -val_b
            
            q = self.indifference_thresholds.get(criterion, 0)
            p = self.preference_thresholds.get(criterion, q * 2)
            
            diff = val_a - val_b
            
            if diff >= p:
                c_j = 1.0
            elif diff <= q:
                c_j = 0.0
            else:
                c_j = (diff - q) / (p - q)
            
            concordance += weight * c_j
        
        return concordance
    
    def calculate_discordance(self,
                             alt_a: pd.Series,
                             alt_b: pd.Series) -> float:
        """
        Calculate discordance index for alternative a outranking b.
        
        Args:
            alt_a: Data for alternative a
            alt_b: Data for alternative b
            
        Returns:
            Discordance index [0, 1]
        """
        max_discordance = 0.0
        
        for criterion in self.criteria:
            val_a = alt_a[criterion]
            val_b = alt_b[criterion]
            
            # Adjust for minimization criteria
            if not self.maximize.get(criterion, True):
                val_a, val_b = -val_a, -val_b
            
            p = self.preference_thresholds.get(criterion, 0)
            v = self.veto_thresholds.get(criterion, p * 5)
            
            diff = val_b - val_a  # Reversed for discordance
            
            if diff >= v:
                d_j = 1.0
            elif diff <= p:
                d_j = 0.0
            else:
                d_j = (diff - p) / (v - p)
            
            max_discordance = max(max_discordance, d_j)
        
        return max_discordance
    
    def calculate_credibility(self,
                             concordance: float,
                             discordance: float) -> float:
        """
        Calculate credibility degree combining concordance and discordance.
        
        Args:
            concordance: Concordance index
            discordance: Discordance index
            
        Returns:
            Credibility degree [0, 1]
        """
        if discordance >= concordance:
            return concordance * (1 - discordance) / (1 - concordance + EPSILON)
        else:
            return concordance
    
    def build_outranking_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Build the outranking matrix for all alternatives.
        
        Args:
            df: DataFrame with alternatives and criteria values
            
        Returns:
            Matrix of credibility degrees
        """
        n = len(df)
        outranking_matrix = np.zeros((n, n))
        
        print("Building outranking matrix...")
        for i in range(n):
            for j in range(n):
                if i != j:
                    alt_a = df.iloc[i]
                    alt_b = df.iloc[j]
                    
                    concordance = self.calculate_concordance(alt_a, alt_b)
                    discordance = self.calculate_discordance(alt_a, alt_b)
                    credibility = self.calculate_credibility(concordance, discordance)
                    
                    outranking_matrix[i, j] = credibility
        
        return outranking_matrix
    
    def rank_alternatives(self, 
                         df: pd.DataFrame,
                         lambda_cut: float = 0.6) -> pd.DataFrame:
        """
        Rank alternatives using ELECTRE III.
        
        Args:
            df: DataFrame with alternatives and criteria
            lambda_cut: Credibility threshold for outranking
            
        Returns:
            DataFrame with ranking scores
        """
        print("Ranking alternatives with ELECTRE III...")
        
        # Build outranking matrix
        outranking_matrix = self.build_outranking_matrix(df)
        
        # Calculate ranking scores (simplified - count of alternatives outranked)
        ranking_scores = []
        for i in range(len(df)):
            outranked = np.sum(outranking_matrix[i, :] >= lambda_cut)
            outranked_by = np.sum(outranking_matrix[:, i] >= lambda_cut)
            score = outranked - outranked_by
            ranking_scores.append(score)
        
        # Add ranking to dataframe
        df_ranked = df.copy()
        df_ranked['electre_score'] = ranking_scores
        df_ranked = df_ranked.sort_values('electre_score', ascending=False)
        
        print(f"Ranking complete. Top score: {max(ranking_scores)}")
        return df_ranked


if __name__ == "__main__":
    # Example usage
    print("ELECTRE III module ready")
