import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Any
import glob
import numpy as np

def read_json_files(folder_path: str) -> List[Dict[str, Any]]:
    """
    Read all JSON files from a folder and return a list of dictionaries.
    
    Args:
        folder_path (str): Path to the folder containing JSON files
        
    Returns:
        List[Dict[str, Any]]: List of parsed JSON data
    """
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    data = []
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
                # Add filename for tracking
                json_data['filename'] = os.path.basename(file_path)
                data.append(json_data)
        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"Error reading {file_path}: {e}")
    
    return data

def create_coverage_dataframe(json_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a pandas DataFrame with coverage data over time.
    
    Args:
        json_data (List[Dict[str, Any]]): List of JSON data dictionaries
        
    Returns:
        pd.DataFrame: DataFrame containing coverage metrics over time
    """
    records = []
    
    for file_data in json_data:
        filename = file_data.get('filename', 'unknown')
        contract_name = list(file_data.keys())[0] if file_data else 'unknown'
        
        if contract_name in file_data and 'generations' in file_data[contract_name]:
            generations = file_data[contract_name]['generations']
            cumulative_time = 0
            
            for gen_data in generations:
                cumulative_time += gen_data['time']
                record = {
                    'filename': filename,
                    'contract': contract_name,
                    'generation': gen_data['generation'],
                    'time_elapsed': cumulative_time,
                    'generation_time': gen_data['time'],
                    'total_transactions': gen_data['total_transactions'],
                    'unique_transactions': gen_data['unique_transactions'],
                    'code_coverage': gen_data['code_coverage'],
                    'branch_coverage': gen_data['branch_coverage'],
                    'total_execution_time': file_data[contract_name].get('execution_time', 0),
                    'memory_consumption': file_data[contract_name].get('memory_consumption', 0),
                    'seed': file_data[contract_name].get('seed', 0)
                }
                records.append(record)
    
    return pd.DataFrame(records)

def create_time_binned_dataframe(df: pd.DataFrame, time_interval: float = 5.0) -> pd.DataFrame:
    """
    Create time-binned data for consistent time-based analysis.
    
    Args:
        df (pd.DataFrame): Original coverage DataFrame
        time_interval (float): Time interval in seconds for binning
        
    Returns:
        pd.DataFrame: Time-binned coverage data
    """
    if df.empty:
        return df
    
    # Create time bins
    max_time = df['time_elapsed'].max()
    time_bins = np.arange(0, max_time + time_interval, time_interval)
    
    binned_records = []
    
    for contract in df['contract'].unique():
        contract_data = df[df['contract'] == contract].copy()
        
        for i in range(len(time_bins) - 1):
            start_time = time_bins[i]
            end_time = time_bins[i + 1]
            
            # Get data points within this time bin
            time_bin_data = contract_data[
                (contract_data['time_elapsed'] >= start_time) & 
                (contract_data['time_elapsed'] < end_time)
            ]
            
            if not time_bin_data.empty:
                # Use the last measurement in the time bin
                latest_data = time_bin_data.iloc[-1]
                record = {
                    'contract': contract,
                    'time_bin_start': start_time,
                    'time_bin_center': (start_time + end_time) / 2,
                    'time_bin_end': end_time,
                    'code_coverage': latest_data['code_coverage'],
                    'branch_coverage': latest_data['branch_coverage'],
                    'total_transactions': latest_data['total_transactions'],
                    'unique_transactions': latest_data['unique_transactions']
                }
                binned_records.append(record)
    
    return pd.DataFrame(binned_records)

def calculate_time_based_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for code and branch coverage over time.
    
    Args:
        df (pd.DataFrame): DataFrame with coverage data over time
        
    Returns:
        Dict[str, Any]: Dictionary containing various statistics
    """
    stats = {
        'overall': {
            'mean_code_coverage': df['code_coverage'].mean(),
            'mean_branch_coverage': df['branch_coverage'].mean(),
            'min_code_coverage': df['code_coverage'].min(),
            'min_branch_coverage': df['branch_coverage'].min(),
            'max_code_coverage': df['code_coverage'].max(),
            'max_branch_coverage': df['branch_coverage'].max(),
            'std_code_coverage': df['code_coverage'].std(),
            'std_branch_coverage': df['branch_coverage'].std(),
            'total_analysis_time': df['time_elapsed'].max(),
            'average_time_to_max_coverage': None
        },
        'by_time_interval': df.groupby(pd.cut(df['time_elapsed'], bins=10)).agg({
            'code_coverage': ['mean', 'min', 'max', 'std'],
            'branch_coverage': ['mean', 'min', 'max', 'std'],
            'total_transactions': 'mean'
        }).round(2),
        'by_contract': df.groupby('contract').agg({
            'code_coverage': ['mean', 'min', 'max', 'std'],
            'branch_coverage': ['mean', 'min', 'max', 'std'],
            'time_elapsed': 'max',
            'total_transactions': 'max'
        }).round(2),
        'coverage_convergence': {}
    }
    
    # Calculate coverage convergence statistics
    for contract in df['contract'].unique():
        contract_data = df[df['contract'] == contract]
        max_code_coverage = contract_data['code_coverage'].max()
        max_branch_coverage = contract_data['branch_coverage'].max()
        
        # Time to reach 90% of max coverage
        time_to_90pct_code = contract_data[
            contract_data['code_coverage'] >= 0.9 * max_code_coverage
        ]['time_elapsed'].min()
        
        time_to_90pct_branch = contract_data[
            contract_data['branch_coverage'] >= 0.9 * max_branch_coverage
        ]['time_elapsed'].min()
        
        stats['coverage_convergence'][contract] = {
            'time_to_90pct_code': time_to_90pct_code,
            'time_to_90pct_branch': time_to_90pct_branch,
            'max_code_coverage': max_code_coverage,
            'max_branch_coverage': max_branch_coverage
        }
    
    return stats

def create_time_based_plots(df: pd.DataFrame, binned_df: pd.DataFrame, output_dir: str = "coverage_plots"):
    """
    Create various plots for coverage analysis over time.
    
    Args:
        df (pd.DataFrame): Original DataFrame with time data
        binned_df (pd.DataFrame): Time-binned DataFrame
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Plot 1: Coverage progression over time (all contracts)
    fig, axes = plt.subplots(2,2, figsize=(16, 12))
    fig.suptitle('Code and Branch Coverage Analysis Over Time', fontsize=16, fontweight='bold')
    
    # Plot coverage over time for each contract
    contracts = df['contract'].unique()
    # _plot_coverage_over_time_contracts(axes=axes[0], df=df, contracts=contracts)
    
    # Plot 2: Mean coverage over time (using binned data)
    _plot_mean_coverage_over_time(binned_df=binned_df, axes=axes[1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'coverage_over_time.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Coverage convergence analysis
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for contract in contracts:
        contract_data = df[df['contract'] == contract]
        
        # Normalize time to percentage of total time
        max_time = contract_data['time_elapsed'].max()
        normalized_time = (contract_data['time_elapsed'] / max_time) * 100
        
        axes[0].plot(normalized_time, contract_data['code_coverage'], 
                   marker='o', markersize=3, linewidth=2, label=contract)
        axes[1].plot(normalized_time, contract_data['branch_coverage'], 
                   marker='s', markersize=3, linewidth=2, label=contract)
    
    axes[0].set_xlabel('Normalized Time (% of Total Analysis Time)')
    axes[0].set_ylabel('Code Coverage (%)')
    axes[0].set_title('Code Coverage vs Normalized Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Normalized Time (% of Total Analysis Time)')
    axes[1].set_ylabel('Branch Coverage (%)')
    axes[1].set_title('Branch Coverage vs Normalized Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'coverage_convergence.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Coverage vs Transactions over time
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for contract in contracts:
        contract_data = df[df['contract'] == contract]
        
        axes[0].plot(contract_data['time_elapsed'], contract_data['total_transactions'], 
                   marker='o', linewidth=2, label=contract)
        axes[1].plot(contract_data['time_elapsed'], contract_data['unique_transactions'], 
                   marker='s', linewidth=2, label=contract)
    
    axes[0].set_xlabel('Time Elapsed (seconds)')
    axes[0].set_ylabel('Total Transactions')
    axes[0].set_title('Total Transactions Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Time Elapsed (seconds)')
    axes[1].set_ylabel('Unique Transactions')
    axes[1].set_title('Unique Transactions Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'transactions_over_time.png'), dpi=300, bbox_inches='tight')
    plt.close()

def _plot_coverage_over_time_contracts(axes, df, contracts):
    """
    Plot coverage over time for each contract
    """
    colors = plt.cm.Set3(np.linspace(0, 1, len(contracts)))
    
    for i, contract in enumerate(contracts):
        contract_data = df[df['contract'] == contract]
        color = colors[i]
        
        axes[0].plot(contract_data['time_elapsed'], contract_data['code_coverage'], 
                       marker='o', markersize=4, linewidth=2, label=contract, color=color)
        axes[1].plot(contract_data['time_elapsed'], contract_data['branch_coverage'], 
                       marker='s', markersize=4, linewidth=2, label=contract, color=color)
    
    axes[0].set_xlabel('Time Elapsed (seconds)')
    axes[0].set_ylabel('Code Coverage (%)')
    axes[0].set_title('Code Coverage Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Time Elapsed (seconds)')
    axes[1].set_ylabel('Branch Coverage (%)')
    axes[1].set_title('Branch Coverage Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

def _plot_mean_coverage_over_time(binned_df, axes):
    """ 
    Plot Mean coverage over time (using binned data)
    """
    if not binned_df.empty:
        time_binned_avg = binned_df.groupby('time_bin_center').agg({
            'code_coverage': 'mean',
            'branch_coverage': 'mean'
        }).reset_index()
        
        axes[0].plot(time_binned_avg['time_bin_center'], time_binned_avg['code_coverage'], 
                       marker='o', linewidth=3, label='Mean Code Coverage', color='blue', alpha=0.8)
        axes[0].fill_between(time_binned_avg['time_bin_center'],
                               time_binned_avg['code_coverage'] - time_binned_avg['code_coverage'].std(),
                               time_binned_avg['code_coverage'] + time_binned_avg['code_coverage'].std(),
                               alpha=0.2, color='blue', label='±1 Std Dev')
        axes[0].set_xlabel('Time (seconds)')
        axes[0].set_ylabel('Code Coverage (%)')
        axes[0].set_title('Mean Code Coverage Over Time with Variability')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(time_binned_avg['time_bin_center'], time_binned_avg['branch_coverage'], 
                       marker='s', linewidth=3, label='Mean Branch Coverage', color='red', alpha=0.8)
        axes[1].fill_between(time_binned_avg['time_bin_center'],
                               time_binned_avg['branch_coverage'] - time_binned_avg['branch_coverage'].std(),
                               time_binned_avg['branch_coverage'] + time_binned_avg['branch_coverage'].std(),
                               alpha=0.2, color='red', label='±1 Std Dev')
        axes[1].set_xlabel('Time (seconds)')
        axes[1].set_ylabel('Branch Coverage (%)')
        axes[1].set_title('Mean Branch Coverage Over Time with Variability')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

def generate_time_based_report(stats: Dict[str, Any], output_dir: str = "coverage_report"):
    """
    Generate a text report with time-based statistics.
    
    Args:
        stats (Dict[str, Any]): Statistics dictionary
        output_dir (str): Directory to save report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, 'time_based_coverage_report.txt')
    with open(report_path, 'w') as f:
        f.write("TIME-BASED COVERAGE ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("OVERALL STATISTICS:\n")
        f.write("-" * 30 + "\n")
        overall = stats['overall']
        f.write(f"Mean Code Coverage: {overall['mean_code_coverage']:.2f}%\n")
        f.write(f"Mean Branch Coverage: {overall['mean_branch_coverage']:.2f}%\n")
        f.write(f"Min Code Coverage: {overall['min_code_coverage']:.2f}%\n")
        f.write(f"Min Branch Coverage: {overall['min_branch_coverage']:.2f}%\n")
        f.write(f"Max Code Coverage: {overall['max_code_coverage']:.2f}%\n")
        f.write(f"Max Branch Coverage: {overall['max_branch_coverage']:.2f}%\n")
        f.write(f"Std Code Coverage: {overall['std_code_coverage']:.2f}%\n")
        f.write(f"Std Branch Coverage: {overall['std_branch_coverage']:.2f}%\n")
        f.write(f"Total Analysis Time: {overall['total_analysis_time']:.2f} seconds\n\n")
        
        f.write("COVERAGE CONVERGENCE ANALYSIS:\n")
        f.write("-" * 35 + "\n")
        for contract, convergence in stats['coverage_convergence'].items():
            f.write(f"Contract: {contract}\n")
            f.write(f"  Max Code Coverage: {convergence['max_code_coverage']:.2f}%\n")
            f.write(f"  Max Branch Coverage: {convergence['max_branch_coverage']:.2f}%\n")
            if pd.notna(convergence['time_to_90pct_code']):
                f.write(f"  Time to 90% of max code coverage: {convergence['time_to_90pct_code']:.2f}s\n")
            else:
                f.write(f"  Time to 90% of max code coverage: Not reached\n")
            if pd.notna(convergence['time_to_90pct_branch']):
                f.write(f"  Time to 90% of max branch coverage: {convergence['time_to_90pct_branch']:.2f}s\n")
            else:
                f.write(f"  Time to 90% of max branch coverage: Not reached\n")
            f.write("\n")
        
        f.write("COVERAGE BY TIME INTERVAL:\n")
        f.write("-" * 30 + "\n")
        f.write(stats['by_time_interval'].to_string())
        f.write("\n\n")
        
        f.write("COVERAGE BY CONTRACT:\n")
        f.write("-" * 30 + "\n")
        f.write(stats['by_contract'].to_string())
    
    print(f"Report saved to: {report_path}")

def analyze_coverage_over_time(folder_path: str, output_dir: str = "coverage_analysis", time_interval: float = 5.0):
    """
    Main function to analyze coverage data over time from JSON files.
    
    Args:
        folder_path (str): Path to folder containing JSON files
        output_dir (str): Directory to save outputs
        time_interval (float): Time interval for binning in seconds
    """
    print(f"Reading JSON files from: {folder_path}")
    
    # Read JSON files
    json_data = read_json_files(folder_path)
    
    if not json_data:
        print("No valid JSON files found or no data to process.")
        return
    
    print(f"Successfully read {len(json_data)} JSON files")
    
    # Create DataFrame with time data
    df = create_coverage_dataframe(json_data)
    
    if df.empty:
        print("No coverage data found in the JSON files.")
        return
    
    print(f"Created DataFrame with {len(df)} records")
    print(f"Contracts found: {df['contract'].unique().tolist()}")
    print(f"Time range: {df['time_elapsed'].min():.2f}s to {df['time_elapsed'].max():.2f}s")
    
    # Create time-binned data
    binned_df = create_time_binned_dataframe(df, time_interval)
    print(f"Created time-binned data with {len(binned_df)} records")
    
    # Calculate time-based statistics
    stats = calculate_time_based_statistics(df)
    
    # Create time-based plots
    print("Creating time-based visualizations...")
    create_time_based_plots(df, binned_df, output_dir)
    
    # Generate report
    generate_time_based_report(stats, output_dir)
    
    # Save DataFrames to CSV
    df.to_csv(os.path.join(output_dir, 'coverage_data_over_time.csv'), index=False)
    binned_df.to_csv(os.path.join(output_dir, 'time_binned_coverage_data.csv'), index=False)
    print(f"Data saved to CSV files in: {output_dir}")
    
    # Print summary
    print("\nTIME-BASED SUMMARY:")
    print(f"Overall Mean Code Coverage: {stats['overall']['mean_code_coverage']:.2f}%")
    print(f"Overall Mean Branch Coverage: {stats['overall']['mean_branch_coverage']:.2f}%")
    print(f"Total Analysis Time Range: {stats['overall']['total_analysis_time']:.2f} seconds")
    
    print("\nCoverage Convergence Analysis:")
    for contract, convergence in stats['coverage_convergence'].items():
        print(f"  {contract}:")
        if pd.notna(convergence['time_to_90pct_code']):
            print(f"    Code coverage reached 90% in {convergence['time_to_90pct_code']:.2f}s")
        if pd.notna(convergence['time_to_90pct_branch']):
            print(f"    Branch coverage reached 90% in {convergence['time_to_90pct_branch']:.2f}s")
    
    print(f"\nAnalysis complete! Results saved in: {output_dir}")
    
    return df, binned_df, stats

# Example usage
if __name__ == "__main__":
    # Replace with your folder path
    folder_path = "results/pso"
    
    # Run time-based analysis
    df, binned_df, stats = analyze_coverage_over_time(folder_path, time_interval=2.0)
    
    # You can also access the DataFrames for further analysis
    if df is not None:
        print("\nDataFrame preview (with time data):")
        print(df[['contract', 'time_elapsed', 'code_coverage', 'branch_coverage']].head())
