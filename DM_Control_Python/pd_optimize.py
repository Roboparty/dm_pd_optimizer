import math
from DM_CAN import *
import serial
import time
import pandas as pd
from BFGS_pd_finder import PDParameterOptimizer

def collect_motor_data(p_value, d_value, data_points=1000, output_file=None):
    """
    Collect motor operation data and save as a CSV file
    
    Args:
        p_value: Proportional parameter value
        d_value: Derivative parameter value
        data_points: Number of data points to collect
        output_file: Output file name in the format pd_p{P value}_d{D value}.csv
    """
    # Initialize motor
    Motor1 = Motor(DM_Motor_Type.DM4310_48V, 0x01, 0x11)
    
    # Initialize serial communication
    serial_device = serial.Serial('COM4', 921600, timeout=0.5)
    MotorControl1 = MotorControl(serial_device)
    MotorControl1.addMotor(Motor1)
    
    # Switch control mode to MIT
    if MotorControl1.switchControlMode(Motor1, Control_Type.MIT):
        print(f"Successfully switched to MIT control with P={p_value}, D={d_value}")
    
    # Save motor parameters (optional)
    # MotorControl1.change_motor_param(Motor1, DM_variable.KP_APR, p_value)
    # MotorControl1.change_motor_param(Motor1, DM_variable.KD_APR, d_value)
    # MotorControl1.save_motor_param(Motor1)
    
    # Enable motor
    MotorControl1.enable(Motor1)
    
    # Data collection lists
    timestamps = []
    policy_actions = []  # Target positions
    actual_actions = []  # Actual positions
    velocities = []      # Actual velocities
    torques = []         # Torque values
    
    # Control motor operation and collect data
    start_time = time.time()
    i = 0
    while i < data_points:
        current_time = time.time() - start_time
        q = math.sin(current_time)  # Target position (sinusoidal wave)
        
        # Control motor
        MotorControl1.control_MIT(Motor1, p_value, d_value, 8*q, 0, 0)
        
        # Collect data
        timestamps.append(current_time)
        policy_actions.append(8*q)         # Target position
        actual_actions.append(Motor1.getPosition())  # Actual position
        velocities.append(Motor1.getVelocity())      # Actual velocity
        torques.append(Motor1.getTorque())          # Torque
        
        i += 1
        time.sleep(0.01)  # 10ms control cycle
    
    # Close serial port
    serial_device.close()
    
    # Save data to CSV
    if output_file:
        data = {
            'timestamp': timestamps,
            'policy_action': policy_actions,
            'actual_action': actual_actions,
            'position': actual_actions,  # Position is the same as actual action
            'velocity': velocities,
            'torque': torques
        }
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
    
    return df

# Test different PD parameter combinations and generate multiple datasets
if __name__ == "__main__":
    pd_combinations = [
        (10, 0.5), (10, 1), (10, 4),
        (20, 0.5), (20, 1), (20, 5),
        (40, 2),
        (50, 1.5), (50, 2.5),
        (80, 3),
        (100, 2), (100, 3),
        (120, 4),
        # (150, 3), (150, 4), (150, 5),
        # (180, 1), (180, 4),
        # (200, 0.5), (200, 5)
    ]
    
    for p, d in pd_combinations:
        output_file = f"pd_p{p}_d{d}.csv"
        collect_motor_data(p, d, data_points=1000, output_file=output_file)


# Configure optimizer
config = {
    'file_pattern': 'pd_p*.csv',  # Match all files in pd_p{d}_d{d}.csv format
    'p_range': (0.1, 200),        # Search range for P parameter
    'd_range': (0.5, 5),          # Search range for D parameter
    'optimization_method': 'L-BFGS-B',
    'max_iterations': 500,
    'tolerance': 1e-5,
    'weight_position': 1.0,       # Weight for position error
    'weight_velocity': 0.5,       # Weight for velocity error
    'weight_acceleration': 0.3,   # Weight for acceleration error (low since we don't have acceleration data)
}

# Create optimizer instance
optimizer = PDParameterOptimizer(
    data_folder='.',  # Current directory, modify to your data folder path
    config=config
)

try:
    # Load data
    optimizer.load_sampling_data()
    
    # Analyze performance of existing PD parameters
    analysis_df = optimizer.analyze_pd_performance()
    print("PD parameter performance analysis:")
    print(analysis_df.head(10))
    
    # Optimize PD parameters
    optimal_p, optimal_d, optimal_error = optimizer.optimize_pd_parameters()
    
    print(f"\nOptimal PD parameters:")
    print(f"P = {optimal_p:.4f}")
    print(f"D = {optimal_d:.4f}")
    print(f"Minimum error = {optimal_error:.6f}")
    
    # Plot results
    optimizer.plot_results(save_path='pd_optimization_results.png')
    
    # Save results to JSON file
    optimizer.save_results('pd_optimization_results.json')
    
except Exception as e:
    print(f"An error occurred during optimization: {str(e)}")