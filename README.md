![20250615-162919](https://github.com/user-attachments/assets/cb96314d-9ad9-4abd-8806-39b765588be7)
PD Parameter Optimizer for Robotics
Overview

This Python package optimizes Proportional-Derivative (PD) controller parameters for robotic systems by comparing policy output actions with actual executed actions. It uses sampling data from various PD parameter combinations to find optimal values that minimize tracking error.

Key features:

    Loads and validates PD sampling data from CSV files

    Calculates weighted tracking errors (position, velocity, acceleration)

    Optimizes parameters using gradient-based methods

    Performs performance analysis (settling time, overshoot, steady-state error)

    Generates comprehensive visualizations

    Saves optimization results in JSON format

Requirements

    Python 3.7+
    Required packages:
    bash
    numpy
    pandas
    matplotlib
    scipy
    scikit-learn

Installation

    Clone the repository:
    bash

git clone https://github.com/yourusername/pd-parameter-optimizer.git
cd pd-parameter-optimizer

Install dependencies:
bash

    pip install -r requirements.txt

Usage
Data Preparation

    Create a folder for your sampling data (e.g., ./pd_sampling_data)

    Save CSV files with the naming format: pd_p{P value}_d{D value}.csv

        Example: pd_p10.0_d0.5.csv

    Each CSV file should contain these columns:

        timestamp

        policy_action

        actual_action

        position

        velocity (optional)

        acceleration (optional)

        target_velocity (optional)

        target_acceleration (optional)

Basic Example
python

from pd_optimizer import PDParameterOptimizer

# Configuration parameters (optional)
config = {
    'file_pattern': '*.csv',
    'p_range': (0.1, 50.0),
    'd_range': (0.01, 5.0),
    'weight_position': 1.0,
    'weight_velocity': 0.5,
    'weight_acceleration': 0.3
}

# Initialize optimizer
optimizer = PDParameterOptimizer(
    data_folder='./pd_sampling_data',
    config=config
)

# Load data
optimizer.load_sampling_data()

# Analyze existing PD parameters
analysis_df = optimizer.analyze_pd_performance()
print("Top performing PD parameters:")
print(analysis_df.head(5))

# Optimize parameters
optimal_p, optimal_d, optimal_error = optimizer.optimize_pd_parameters()
print(f"\nOptimal parameters: P={optimal_p:.4f}, D={optimal_d:.4f}")
print(f"Minimum error: {optimal_error:.6f}")

# Generate and save visualizations
optimizer.plot_results(save_path='optimization_results.png')

# Save full results
optimizer.save_results('optimization_results.json')

Configuration Options
Parameter	Default Value	Description
file_pattern	'*.csv'	File pattern for data loading
p_range	(0.1, 200.0)	Search range for P parameter
d_range	(0.01, 5.0)	Search range for D parameter
optimization_method	'L-BFGS-B'	Optimization algorithm
max_iterations	1000	Maximum optimization iterations
tolerance	1e-6	Optimization tolerance
weight_position	1.0	Weight for position error
weight_velocity	0.5	Weight for velocity error
weight_acceleration	0.3	Weight for acceleration error
Output Visualization

The plot_results() method generates a 2x2 grid of visualizations:

    Error Heatmap
    Shows tracking error across different PD parameter combinations

    Best Waveform Comparison
    Compares policy actions vs actual actions for optimal PD parameters

    PD Parameter Scatter Plot
    Visualizes parameter space with error coloring and marks best point

    Performance Metrics Comparison
    Compares overshoot, settling time, and steady-state error for top 5 PD combinations


The analyze_pd_performance() method returns a DataFrame with these metrics for each PD combination:
Metric	Description
Total_Error	Weighted composite tracking error
Settling_Time	Time to reach within 2% of target value
Overshoot	Maximum overshoot percentage
Steady_State_Error	Average error in last 10% of trajectory
Optimization Algorithm

The optimization process:

    Uses sampled PD points as starting references

    Estimates errors for unsampled points using distance-weighted interpolation

    Applies gradient-based optimization (L-BFGS-B by default) to find minimum

    Respects parameter boundaries defined in configuration

Contributing

Contributions are welcome! Please submit pull requests for:

    Additional optimization algorithms
    Improved interpolation methods
    Enhanced visualization options
    Extended performance metrics

License

This project is licensed under the MIT License - see the LICENSE file for details.
