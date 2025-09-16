import numpy as np
import pandas as pd
from scipy.optimize import linprog
from typing import Tuple, List, Optional, Dict, Union
import warnings
from dataclasses import dataclass


@dataclass
class DEAResults:
    """Container for DEA analysis results."""
    scores: np.ndarray
    lambdas: np.ndarray
    input_slacks: np.ndarray
    output_slacks: np.ndarray
    status: List[str]
    peers: List[List[int]]
    peer_weights: List[List[float]]
    
    def to_dataframe(self, dmu_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Convert results to a pandas DataFrame."""
        if dmu_names is None:
            dmu_names = [f"DMU_{i+1}" for i in range(len(self.scores))]
        
        return pd.DataFrame({
            'DMU': dmu_names,
            'Efficiency_Score': self.scores,
            'Status': self.status,
            'Peers': [', '.join(map(str, peers)) for peers in self.peers],
            'Input_Slacks': self.input_slacks.tolist(),
            'Output_Slacks': self.output_slacks.tolist()
        })


class DEAAnalyzer:
    """
    Data Envelopment Analysis (DEA) with comprehensive functionality.
    
    Supports both input and output orientations, constant (CRS) and variable (VRS)
    returns to scale, slack analysis, and peer identification.
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize DEA analyzer.
        
        Parameters
        ----------
        tolerance : float, default=1e-6
            Numerical tolerance for computations.
        """
        self.tolerance = tolerance
        
    def validate_inputs(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Validate input data."""
        if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
            raise TypeError("X and Y must be numpy arrays")
        
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must be 2D arrays")
        
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of DMUs")
        
        if np.any(X <= 0) or np.any(Y <= 0):
            raise ValueError("All inputs and outputs must be positive")
        
        n_dmus, n_inputs = X.shape
        _, n_outputs = Y.shape
        
        # Rule of thumb: n_dmus should be at least 3 times (n_inputs + n_outputs)
        min_dmus = 3 * (n_inputs + n_outputs)
        if n_dmus < min_dmus:
            warnings.warn(
                f"Number of DMUs ({n_dmus}) may be insufficient. "
                f"Recommend at least {min_dmus} DMUs for {n_inputs} inputs and {n_outputs} outputs.",
                UserWarning
            )
    
    def analyze(self, 
                X: np.ndarray, 
                Y: np.ndarray, 
                orientation: str = "input",
                rts: str = "CRS",
                slack_correction: bool = True) -> DEAResults:
        """
        Perform DEA analysis with slacks.
        
        Parameters
        ----------
        X : np.ndarray
            Input matrix (n_dmus x n_inputs).
        Y : np.ndarray
            Output matrix (n_dmus x n_outputs).
        orientation : str, default="input"
            "input" for input-oriented or "output" for output-oriented.
        rts : str, default="CRS"
            "CRS" for constant returns to scale, "VRS" for variable returns to scale.
        slack_correction : bool, default=True
            Whether to apply two-stage slack correction for efficient units.
            
        Returns
        -------
        DEAResults
            Complete results object with scores, slacks, and peer information.
        """
        # Validate inputs
        self.validate_inputs(X, Y)
        
        if orientation not in ["input", "output"]:
            raise ValueError("Orientation must be 'input' or 'output'")
        
        if rts not in ["CRS", "VRS"]:
            raise ValueError("RTS must be 'CRS' or 'VRS'")
        
        n_dmus, n_inputs = X.shape
        _, n_outputs = Y.shape
        
        # Initialize result arrays
        scores = np.zeros(n_dmus)
        lambdas = np.zeros((n_dmus, n_dmus))
        input_slacks = np.zeros((n_dmus, n_inputs))
        output_slacks = np.zeros((n_dmus, n_outputs))
        status = []
        peers = []
        peer_weights = []
        
        for k in range(n_dmus):
            result = self._solve_dmu(X, Y, k, orientation, rts, slack_correction)
            
            scores[k] = result['score']
            lambdas[k] = result['lambda']
            input_slacks[k] = result['input_slack']
            output_slacks[k] = result['output_slack']
            status.append(result['status'])
            peers.append(result['peers'])
            peer_weights.append(result['peer_weights'])
        
        return DEAResults(
            scores=scores,
            lambdas=lambdas,
            input_slacks=input_slacks,
            output_slacks=output_slacks,
            status=status,
            peers=peers,
            peer_weights=peer_weights
        )
    
    def _solve_dmu(self, 
                   X: np.ndarray, 
                   Y: np.ndarray, 
                   k: int, 
                   orientation: str, 
                   rts: str,
                   slack_correction: bool) -> Dict:
        """Solve DEA problem for a single DMU."""
        n_dmus, n_inputs = X.shape
        _, n_outputs = Y.shape
        
        # Stage 1: Efficiency optimization
        result_stage1 = self._solve_stage1(X, Y, k, orientation, rts)
        
        if not result_stage1['success']:
            return {
                'score': np.nan,
                'lambda': np.zeros(n_dmus),
                'input_slack': np.zeros(n_inputs),
                'output_slack': np.zeros(n_outputs),
                'status': 'Infeasible',
                'peers': [],
                'peer_weights': []
            }
        
        score = result_stage1['score']
        
        # Stage 2: Slack maximization (if efficient and slack_correction enabled)
        if slack_correction and abs(score - 1.0) < self.tolerance:
            result_stage2 = self._solve_stage2(X, Y, k, orientation, rts, score)
            if result_stage2['success']:
                lambda_weights = result_stage2['lambda']
                input_slack = result_stage2['input_slack']
                output_slack = result_stage2['output_slack']
            else:
                lambda_weights = result_stage1['lambda']
                input_slack = result_stage1['input_slack']
                output_slack = result_stage1['output_slack']
        else:
            lambda_weights = result_stage1['lambda']
            input_slack = result_stage1['input_slack']
            output_slack = result_stage1['output_slack']
        
        # Identify peers
        peers, peer_weights_list = self._identify_peers(lambda_weights)
        
        # Determine efficiency status
        is_efficient = abs(score - 1.0) < self.tolerance
        has_slacks = (np.sum(input_slack) + np.sum(output_slack)) > self.tolerance
        
        if is_efficient and not has_slacks:
            status = "Efficient"
        elif is_efficient and has_slacks:
            status = "Weakly Efficient"
        else:
            status = "Inefficient"
        
        return {
            'score': float(score),
            'lambda': lambda_weights,
            'input_slack': input_slack,
            'output_slack': output_slack,
            'status': status,
            'peers': peers,
            'peer_weights': peer_weights_list
        }
    
    def _solve_stage1(self, 
                      X: np.ndarray, 
                      Y: np.ndarray, 
                      k: int, 
                      orientation: str, 
                      rts: str) -> Dict:
        """Solve Stage 1: Efficiency optimization."""
        n_dmus, n_inputs = X.shape
        _, n_outputs = Y.shape
        
        # Variables: [λ₁, λ₂, ..., λₙ, θ/φ, s⁻₁, ..., s⁻ₘ, s⁺₁, ..., s⁺ₛ]
        n_vars = n_dmus + 1 + n_inputs + n_outputs
        
        # Objective function
        c = np.zeros(n_vars)
        if orientation == "input":
            c[n_dmus] = 1.0  # minimize θ
        else:
            c[n_dmus] = -1.0  # maximize φ (negative for minimization)
        
        # Inequality constraints
        A_ub = []
        b_ub = []
        
        # Input constraints: Σλⱼxⱼᵢ + s⁻ᵢ = θxₖᵢ (input-oriented) or Σλⱼxⱼᵢ + s⁻ᵢ = xₖᵢ (output-oriented)
        for i in range(n_inputs):
            row = np.zeros(n_vars)
            row[:n_dmus] = X[:, i]  # λ coefficients
            if orientation == "input":
                row[n_dmus] = -X[k, i]  # -θxₖᵢ
            else:
                b_ub.append(X[k, i])  # xₖᵢ
            row[n_dmus + 1 + i] = 1.0  # s⁻ᵢ
            A_ub.append(row)
            if orientation == "input":
                b_ub.append(0.0)
        
        # Output constraints: Σλⱼyⱼᵣ - s⁺ᵣ = yₖᵣ (input-oriented) or Σλⱼyⱼᵣ - s⁺ᵣ = φyₖᵣ (output-oriented)
        for r in range(n_outputs):
            row = np.zeros(n_vars)
            row[:n_dmus] = -Y[:, r]  # -λ coefficients (for ≤ constraint)
            if orientation == "output":
                row[n_dmus] = Y[k, r]  # φyₖᵣ
                b_ub.append(0.0)
            else:
                b_ub.append(-Y[k, r])  # -yₖᵣ
            row[n_dmus + 1 + n_inputs + r] = 1.0  # s⁺ᵣ
            A_ub.append(row)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Equality constraints (VRS only)
        A_eq = None
        b_eq = None
        if rts == "VRS":
            A_eq = np.zeros((1, n_vars))
            A_eq[0, :n_dmus] = 1.0  # Σλⱼ = 1
            b_eq = np.array([1.0])
        
        # Variable bounds
        bounds = [(0, None)] * n_vars
        
        # Solve
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                        bounds=bounds, method="highs", options={'presolve': True})
        
        if result.success:
            lambda_weights = result.x[:n_dmus]
            theta_phi = result.x[n_dmus]
            input_slack = result.x[n_dmus + 1:n_dmus + 1 + n_inputs]
            output_slack = result.x[n_dmus + 1 + n_inputs:]
            
            if orientation == "input":
                score = theta_phi
            else:
                score = 1.0 / theta_phi if theta_phi > self.tolerance else np.inf
            
            return {
                'success': True,
                'score': score,
                'lambda': lambda_weights,
                'input_slack': input_slack,
                'output_slack': output_slack
            }
        else:
            return {'success': False}
    
    def _solve_stage2(self, 
                      X: np.ndarray, 
                      Y: np.ndarray, 
                      k: int, 
                      orientation: str, 
                      rts: str,
                      fixed_score: float) -> Dict:
        """Solve Stage 2: Slack maximization for efficient units."""
        n_dmus, n_inputs = X.shape
        _, n_outputs = Y.shape
        
        # Variables: [λ₁, λ₂, ..., λₙ, s⁻₁, ..., s⁻ₘ, s⁺₁, ..., s⁺ₛ]
        n_vars = n_dmus + n_inputs + n_outputs
        
        # Objective: maximize sum of slacks (negative for minimization)
        c = np.zeros(n_vars)
        c[n_dmus:] = -1.0  # -Σs⁻ᵢ - Σs⁺ᵣ
        
        # Inequality constraints
        A_ub = []
        b_ub = []
        
        # Input constraints with fixed efficiency score
        for i in range(n_inputs):
            row = np.zeros(n_vars)
            row[:n_dmus] = X[:, i]
            row[n_dmus + i] = 1.0  # s⁻ᵢ
            A_ub.append(row)
            if orientation == "input":
                b_ub.append(fixed_score * X[k, i])
            else:
                b_ub.append(X[k, i])
        
        # Output constraints with fixed efficiency score
        for r in range(n_outputs):
            row = np.zeros(n_vars)
            row[:n_dmus] = -Y[:, r]
            row[n_dmus + n_inputs + r] = 1.0  # s⁺ᵣ
            A_ub.append(row)
            if orientation == "output":
                b_ub.append(-fixed_score * Y[k, r])
            else:
                b_ub.append(-Y[k, r])
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Equality constraints
        A_eq = None
        b_eq = None
        if rts == "VRS":
            A_eq = np.zeros((1, n_vars))
            A_eq[0, :n_dmus] = 1.0
            b_eq = np.array([1.0])
        
        # Variable bounds
        bounds = [(0, None)] * n_vars
        
        # Solve
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                        bounds=bounds, method="highs", options={'presolve': True})
        
        if result.success:
            return {
                'success': True,
                'lambda': result.x[:n_dmus],
                'input_slack': result.x[n_dmus:n_dmus + n_inputs],
                'output_slack': result.x[n_dmus + n_inputs:]
            }
        else:
            return {'success': False}
    
    def _identify_peers(self, lambda_weights: np.ndarray) -> Tuple[List[int], List[float]]:
        """Identify peers (reference units) and their weights."""
        peers = []
        weights = []
        
        for i, weight in enumerate(lambda_weights):
            if weight > self.tolerance:
                peers.append(i)
                weights.append(float(weight))
        
        return peers, weights
    
    def summary_statistics(self, results: DEAResults) -> Dict[str, float]:
        """Calculate summary statistics for DEA results."""
        valid_scores = results.scores[~np.isnan(results.scores)]
        
        return {
            'mean_efficiency': np.mean(valid_scores),
            'median_efficiency': np.median(valid_scores),
            'std_efficiency': np.std(valid_scores),
            'min_efficiency': np.min(valid_scores),
            'max_efficiency': np.max(valid_scores),
            'efficient_units': np.sum(np.abs(valid_scores - 1.0) < self.tolerance),
            'total_units': len(valid_scores)
        }


# Convenience function for backward compatibility
def dea_with_slacks(X: np.ndarray, 
                   Y: np.ndarray, 
                   orientation: str = "input", 
                   rts: str = "CRS") -> Tuple[List[float], List[List[float]], List[List[float]], List[List[float]]]:
    """
    Backward-compatible DEA function.
    
    Returns
    -------
    scores : List[float]
        Efficiency scores.
    lambdas_all : List[List[float]]
        Lambda weights for peers.
    slacks_in_all : List[List[float]]
        Input slacks.
    slacks_out_all : List[List[float]]
        Output slacks.
    """
    analyzer = DEAAnalyzer()
    results = analyzer.analyze(X, Y, orientation, rts)
    
    return (
        results.scores.tolist(),
        results.lambdas.tolist(),
        results.input_slacks.tolist(),
        results.output_slacks.tolist()
    )


# Example usage and testing
if __name__ == "__main__":
    # Sample data
    np.random.seed(42)
    X = np.array([
        [2, 3],
        [4, 2],
        [3, 4],
        [5, 1],
        [1, 5]
    ]).astype(float)
    
    Y = np.array([
        [1, 2],
        [2, 1],
        [3, 1],
        [1, 3],
        [2, 2]
    ]).astype(float)
    
    # Create analyzer
    analyzer = DEAAnalyzer()
    
    # Run analysis
    print("Input-oriented CRS DEA Analysis:")
    print("=" * 40)
    results_input_crs = analyzer.analyze(X, Y, orientation="input", rts="CRS")
    df = results_input_crs.to_dataframe()
    print(df)
    print()
    
    # Summary statistics
    stats = analyzer.summary_statistics(results_input_crs)
    print("Summary Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")
    print()
    
    # Output-oriented VRS analysis
    print("Output-oriented VRS DEA Analysis:")
    print("=" * 40)
    results_output_vrs = analyzer.analyze(X, Y, orientation="output", rts="VRS")
    df_vrs = results_output_vrs.to_dataframe()
    print(df_vrs)