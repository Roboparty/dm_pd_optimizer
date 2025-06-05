import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import logging
from typing import Dict, List, Tuple, Optional
import json

class PDParameterOptimizer:
    """
    Robot PD Parameter Optimizer
    Optimizes PD parameters by comparing policy output actions and actual executed actions
    """
    
    def __init__(self, data_folder: str, config: Optional[Dict] = None):
        """
        Initialize the optimizer
        
        Args:
            data_folder: Path to the folder containing PD sampling data
            config: Configuration parameter dictionary
        """
        self.data_folder = data_folder
        self.config = config or self._default_config()
        self.sampling_data = {}
        self.pd_combinations = []
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _default_config(self) -> Dict:
        """Default configuration parameters"""
        return {
            'file_pattern': '*.csv',  # Data file format
            'p_range': (0.1, 200.0),  # P parameter search range
            'd_range': (0.01, 5.0),  # D parameter search range
            'optimization_method': 'L-BFGS-B',  # Optimization algorithm
            'max_iterations': 1000,
            'tolerance': 1e-6,
            'weight_position': 1.0,  # Weight for position error
            'weight_velocity': 0.5,  # Weight for velocity error
            'weight_acceleration': 0.3,  # Weight for acceleration error
        }
    
    def load_sampling_data(self) -> None:
        """
        Load all PD parameter corresponding sampling data from the folder
        
        Expected file naming format: pd_p{P value}_d{D value}.csv
        CSV format: timestamp, policy_action, actual_action, position, velocity, acceleration
        """
        pattern = os.path.join(self.data_folder, self.config['file_pattern'])
        files = glob.glob(pattern)
        
        if not files:
            raise FileNotFoundError(f"No matching data files found: {pattern}")
        
        for file_path in files:
            try:
                # Extract PD parameters from filename
                filename = os.path.basename(file_path)
                p_value, d_value = self._extract_pd_from_filename(filename)
                
                # Read data
                data = pd.read_csv(file_path)
                self._validate_data_format(data)
                
                # Store data
                pd_key = (p_value, d_value)
                self.sampling_data[pd_key] = data
                self.pd_combinations.append(pd_key)
                
                self.logger.info(f"Loaded data: P={p_value}, D={d_value}, Sample count={len(data)}")
                
            except Exception as e:
                self.logger.error(f"Failed to load file {file_path}: {str(e)}")
                continue
        
        if not self.sampling_data:
            raise ValueError("No valid data successfully loaded")
        
        self.logger.info(f"Total loaded {len(self.sampling_data)} sets of PD parameter data")
    
    def _extract_pd_from_filename(self, filename: str) -> Tuple[float, float]:
        """Extract PD parameter values from filename"""
        import re
        
        # Match P and D values in filename
        p_match = re.search(r'p([\d.]+)', filename, re.IGNORECASE)
        d_match = re.search(r'd([\d.]+)', filename, re.IGNORECASE)
        
        if not p_match or not d_match:
            raise ValueError(f"Could not extract PD parameters from filename: {filename}")
        
        return float(p_match.group(1)), float(d_match.group(1))
    
    def _validate_data_format(self, data: pd.DataFrame) -> None:
        """Validate data format"""
        required_columns = ['timestamp', 'policy_action', 'actual_action', 'position']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if len(data) < 2:
            raise ValueError("Too few data points for analysis")
    
    def calculate_tracking_error(self, pd_params: Tuple[float, float]) -> float:
        """
        Calculate tracking error for specific PD parameters
        
        Args:
            pd_params: (P, D) parameter tuple
            
        Returns:
            Composite error value
        """
        if pd_params not in self.sampling_data:
            return float('inf')
        
        data = self.sampling_data[pd_params]
        
        # Calculate position tracking error
        position_error = np.mean((data['policy_action'] - data['actual_action']) ** 2)
        
        # Calculate velocity tracking error (if velocity data exists)
        velocity_error = 0
        if 'velocity' in data.columns and 'target_velocity' in data.columns:
            velocity_error = np.mean((data['target_velocity'] - data['velocity']) ** 2)
        
        # Calculate acceleration tracking error (if acceleration data exists)
        acceleration_error = 0
        if 'acceleration' in data.columns and 'target_acceleration' in data.columns:
            acceleration_error = np.mean((data['target_acceleration'] - data['acceleration']) ** 2)
        
        # Weighted composite error
        total_error = (self.config['weight_position'] * position_error +
                      self.config['weight_velocity'] * velocity_error +
                      self.config['weight_acceleration'] * acceleration_error)
        
        return total_error
    
    def interpolation_error_function(self, pd_params: np.ndarray) -> float:
        """
        Interpolation error function for optimization
        Estimates error using interpolation when PD parameters are not in sampling points
        """
        p, d = pd_params
        
        # Check parameter range
        if not (self.config['p_range'][0] <= p <= self.config['p_range'][1] and
                self.config['d_range'][0] <= d <= self.config['d_range'][1]):
            return float('inf')
        
        # Direct return if it's a sampling point
        if (p, d) in self.sampling_data:
            return self.calculate_tracking_error((p, d))
        
        # Use nearest neighbor interpolation or weighted average
        return self._interpolate_error(p, d)
    
    def _interpolate_error(self, p: float, d: float) -> float:
        """
        Estimate error for unsampled PD parameters using distance-weighted interpolation
        """
        errors = []
        weights = []
        
        for (p_sample, d_sample) in self.pd_combinations:
            # Calculate Euclidean distance
            distance = np.sqrt((p - p_sample)**2 + (d - d_sample)**2)
            
            if distance < 1e-10:  # Avoid division by zero
                return self.calculate_tracking_error((p_sample, d_sample))
            
            weight = 1.0 / distance
            error = self.calculate_tracking_error((p_sample, d_sample))
            
            errors.append(error)
            weights.append(weight)
        
        # Weighted average
        weights = np.array(weights)
        errors = np.array(errors)
        
        return np.sum(weights * errors) / np.sum(weights)
    
    def optimize_pd_parameters(self) -> Tuple[float, float, float]:
        """
        Optimize PD parameters using gradient descent
        
        Returns:
            (Optimal P, Optimal D, Minimum Error)
        """
        self.logger.info("Starting PD parameter optimization...")
        
        # Initial guess: use the sampling point with minimum error as starting point
        best_sampled_error = float('inf')
        best_sampled_params = None
        
        for pd_params in self.pd_combinations:
            error = self.calculate_tracking_error(pd_params)
            if error < best_sampled_error:
                best_sampled_error = error
                best_sampled_params = pd_params
        
        initial_guess = np.array(best_sampled_params)
        self.logger.info(f"Initial guess: P={initial_guess[0]:.4f}, D={initial_guess[1]:.4f}, "
                        f"Error={best_sampled_error:.6f}")
        
        # Set optimization bounds
        bounds = [self.config['p_range'], self.config['d_range']]
        
        # Perform optimization
        result = minimize(
            fun=self.interpolation_error_function,
            x0=initial_guess,
            method=self.config['optimization_method'],
            bounds=bounds,
            options={
                'maxiter': self.config['max_iterations'],
                'ftol': self.config['tolerance']
            }
        )
        
        if result.success:
            optimal_p, optimal_d = result.x
            optimal_error = result.fun
            
            self.logger.info("Optimization successful!")
            self.logger.info(f"Optimal parameters: P={optimal_p:.4f}, D={optimal_d:.4f}")
            self.logger.info(f"Minimum error: {optimal_error:.6f}")
            
            return optimal_p, optimal_d, optimal_error
        else:
            self.logger.error(f"Optimization failed: {result.message}")
            return best_sampled_params[0], best_sampled_params[1], best_sampled_error
    
    def analyze_pd_performance(self) -> pd.DataFrame:
        """Analyze performance of all sampled PD parameters"""
        results = []
        
        for pd_params in self.pd_combinations:
            p, d = pd_params
            error = self.calculate_tracking_error(pd_params)
            
            data = self.sampling_data[pd_params]
            
            # Additional analysis metrics
            settling_time = self._calculate_settling_time(data)
            overshoot = self._calculate_overshoot(data)
            steady_state_error = self._calculate_steady_state_error(data)
            
            results.append({
                'P': p,
                'D': d,
                'Total_Error': error,
                'Settling_Time': settling_time,
                'Overshoot': overshoot,
                'Steady_State_Error': steady_state_error
            })
        
        return pd.DataFrame(results).sort_values('Total_Error')
    
    def _calculate_settling_time(self, data: pd.DataFrame, tolerance: float = 0.02) -> float:
        """Calculate settling time"""
        if len(data) < 10:
            return float('inf')
        
        target = data['policy_action'].iloc[-1]
        error_threshold = abs(target) * tolerance
        
        for i in range(len(data)-1, -1, -1):
            if abs(data['actual_action'].iloc[i] - target) > error_threshold:
                return data['timestamp'].iloc[i] if i < len(data)-1 else float('inf')
        
        return data['timestamp'].iloc[0]
    
    def _calculate_overshoot(self, data: pd.DataFrame) -> float:
        """Calculate overshoot"""
        target = data['policy_action'].iloc[-1]
        actual_max = data['actual_action'].max()
        actual_min = data['actual_action'].min()
        
        if target > data['actual_action'].iloc[0]:  # Positive step
            overshoot = max(0, actual_max - target) / abs(target) * 100
        else:  # Negative step
            overshoot = max(0, target - actual_min) / abs(target) * 100
        
        return overshoot
    
    def _calculate_steady_state_error(self, data: pd.DataFrame) -> float:
        """Calculate steady-state error"""
        # Take last 10% of data to calculate steady-state error
        steady_start = int(len(data) * 0.9)
        target = data['policy_action'].iloc[-1]
        steady_actual = data['actual_action'].iloc[steady_start:].mean()
        
        return abs(target - steady_actual)
    
    def plot_results(self, save_path: str = None) -> None:
        """Plot result graphs"""
        if not self.sampling_data:
            self.logger.warning("No data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Error heatmap
        self._plot_error_heatmap(axes[0, 0])
        
        # 2. Waveform comparison of best PD parameters
        self._plot_best_waveform(axes[0, 1])
        
        # 3. PD parameter scatter plot
        self._plot_pd_scatter(axes[1, 0])
        
        # 4. Performance metric comparison
        self._plot_performance_metrics(axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Graph saved to: {save_path}")
        
        plt.show()
    
    def _plot_error_heatmap(self, ax):
        """Plot error heatmap"""
        # Prepare data
        p_values = sorted(set([p for p, d in self.pd_combinations]))
        d_values = sorted(set([d for p, d in self.pd_combinations]))
        
        error_matrix = np.full((len(d_values), len(p_values)), np.nan)
        
        for i, d in enumerate(d_values):
            for j, p in enumerate(p_values):
                if (p, d) in self.sampling_data:
                    error_matrix[i, j] = self.calculate_tracking_error((p, d))
        
        im = ax.imshow(error_matrix, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(p_values)))
        ax.set_xticklabels([f'{p:.2f}' for p in p_values])
        ax.set_yticks(range(len(d_values)))
        ax.set_yticklabels([f'{d:.2f}' for d in d_values])
        ax.set_xlabel('P Value')
        ax.set_ylabel('D Value')
        ax.set_title('PD Parameter Error Heatmap')
        plt.colorbar(im, ax=ax, label='Tracking Error')
    
    def _plot_best_waveform(self, ax):
        """Plot waveform comparison for best PD parameters"""
        # Find PD parameters with minimum error
        best_error = float('inf')
        best_params = None
        
        for pd_params in self.pd_combinations:
            error = self.calculate_tracking_error(pd_params)
            if error < best_error:
                best_error = error
                best_params = pd_params
        
        if best_params:
            data = self.sampling_data[best_params]
            ax.plot(data['timestamp'], data['policy_action'], 'r-', label='Policy Action', linewidth=2)
            ax.plot(data['timestamp'], data['actual_action'], 'b--', label='Actual Action', linewidth=2)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Action Value')
            ax.set_title(f'Best PD Parameter Waveform (P={best_params[0]:.3f}, D={best_params[1]:.3f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_pd_scatter(self, ax):
        """Plot PD parameter scatter plot"""
        p_values = [p for p, d in self.pd_combinations]
        d_values = [d for p, d in self.pd_combinations]
        errors = [self.calculate_tracking_error(pd) for pd in self.pd_combinations]
        
        scatter = ax.scatter(p_values, d_values, c=errors, cmap='viridis', s=100, alpha=0.7)
        ax.set_xlabel('P Value')
        ax.set_ylabel('D Value')
        ax.set_title('PD Parameter Distribution')
        plt.colorbar(scatter, ax=ax, label='Tracking Error')
        
        # Mark best point
        min_error_idx = np.argmin(errors)
        ax.scatter(p_values[min_error_idx], d_values[min_error_idx], 
                  c='red', s=200, marker='*', label='Best Parameters')
        ax.legend()
    
    def _plot_performance_metrics(self, ax):
        """Plot performance metric comparison"""
        analysis_df = self.analyze_pd_performance()
        top_5 = analysis_df.head(5)
        
        x = range(len(top_5))
        width = 0.25
        
        ax.bar([i - width for i in x], top_5['Overshoot'], width, label='Overshoot (%)', alpha=0.8)
        ax.bar(x, top_5['Settling_Time'], width, label='Settling Time (s)', alpha=0.8)
        ax.bar([i + width for i in x], top_5['Steady_State_Error']*100, width, 
               label='Steady-State Error (%)', alpha=0.8)
        
        ax.set_xlabel('PD Parameter Combinations')
        ax.set_ylabel('Performance Metrics')
        ax.set_title('Top 5 PD Parameter Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([f'P={row.P:.2f}\nD={row.D:.2f}' for _, row in top_5.iterrows()])
        ax.legend()
    
    def save_results(self, output_path: str) -> None:
        """Save optimization results"""
        # Perform optimization
        optimal_p, optimal_d, optimal_error = self.optimize_pd_parameters()
        
        # Prepare result data
        results = {
            'optimal_parameters': {
                'P': float(optimal_p),
                'D': float(optimal_d),
                'error': float(optimal_error)
            },
            'analysis': self.analyze_pd_performance().to_dict('records'),
            'config': self.config
        }
        
        # Save JSON results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to: {output_path}")

# Example usage
def main():
    """Main function example"""
    # Configuration parameters
    config = {
        'file_pattern': '*.csv',
        'p_range': (0.1, 50.0),
        'd_range': (0.01, 5.0),
        'optimization_method': 'L-BFGS-B',
        'max_iterations': 1000,
        'weight_position': 1.0,
        'weight_velocity': 0.5,
        'weight_acceleration': 0.3,
    }
    
    # Create optimizer
    optimizer = PDParameterOptimizer(
        data_folder='./pd_sampling_data',  # Modify to your data folder path
        config=config
    )
    
    try:
        # Load data
        optimizer.load_sampling_data()
        
        # Analyze performance of existing PD parameters
        analysis_df = optimizer.analyze_pd_performance()
        print("PD Parameter Performance Analysis:")
        print(analysis_df.head(10))
        
        # Optimize PD parameters
        optimal_p, optimal_d, optimal_error = optimizer.optimize_pd_parameters()
        
        print(f"\nOptimal PD Parameters:")
        print(f"P = {optimal_p:.4f}")
        print(f"D = {optimal_d:.4f}")
        print(f"Minimum Error = {optimal_error:.6f}")
        
        # Plot results
        optimizer.plot_results(save_path='pd_optimization_results.png')
        
        # Save results
        optimizer.save_results('pd_optimization_results.json')
        
    except Exception as e:
        print(f"An error occurred during optimization: {str(e)}")

if __name__ == "__main__":
    main()