import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Any
import glob
import numpy as np

def find_json_files_recursively(folder_path: str) -> List[str]:
    """
    Recursively find all JSON files in folder and subfolders.
    """
    json_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def read_json_files(folder_path: str) -> List[Dict[str, Any]]:
    """
    Read all JSON files from a folder recursively.
    """
    json_files = find_json_files_recursively(folder_path)
    data = []
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
                json_data['filename'] = os.path.basename(file_path)
                json_data['relative_path'] = os.path.relpath(file_path, folder_path)
                json_data['folder'] = os.path.dirname(os.path.relpath(file_path, folder_path))
                data.append(json_data)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    print(f"Found {len(json_files)} JSON files in {folder_path} and its subfolders")
    return data

def create_coverage_dataframe(json_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a pandas DataFrame with coverage data over time.
    """
    records = []
    
    for file_data in json_data:
        filename = file_data.get('filename', 'unknown')
        relative_path = file_data.get('relative_path', 'unknown')
        folder = file_data.get('folder', 'unknown')
        
        contract_keys = [k for k in file_data.keys() if k not in ['filename', 'relative_path', 'folder']]
        if not contract_keys:
            continue
            
        contract_name = contract_keys[0]
        
        if contract_name in file_data and 'generations' in file_data[contract_name]:
            generations = file_data[contract_name]['generations']
            cumulative_time = 0
            
            for gen_data in generations:
                cumulative_time += gen_data['time']
                record = {
                    'filename': filename,
                    'relative_path': relative_path,
                    'folder': folder,
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
                    'seed': file_data[contract_name].get('seed', 0),
                    'tag': file_data[contract_name].get('tag', ""),
                    'algorithm': file_data[contract_name].get('algorithm', 'unknown')
                }
                records.append(record)
    
    return pd.DataFrame(records)

def create_smooth_time_series_with_std(df: pd.DataFrame, time_interval: float = 5.0, max_time: float = 600.0) -> pd.DataFrame:
    """
    Create smooth time series data with standard deviation for plotting coverage over time.
    
    Args:
        df: Input DataFrame
        time_interval: Time interval in seconds for sampling
        max_time: Maximum time to show on X-axis
    """
    if df.empty:
        return pd.DataFrame()
    
    # Cap the time points at max_time
    time_points = np.arange(0, min(df['time_elapsed'].max(), max_time) + time_interval, time_interval)
    
    smooth_records = []
    
    # Group by algorithm
    for algorithm in df['algorithm'].unique():
        algorithm_data = df[df['algorithm'] == algorithm]
        
        for time_point in time_points:
            if time_point > max_time:
                continue
                
            # Find all data points at or before this time point for each contract
            code_coverage_values = []
            branch_coverage_values = []
            for contract in algorithm_data['contract'].unique():
                contract_data = algorithm_data[algorithm_data['contract'] == contract]
                time_slice = contract_data[contract_data['time_elapsed'] <= time_point]
                if not time_slice.empty:
                    latest_data = time_slice.iloc[-1]
                    code_coverage_values.append(latest_data['code_coverage'])
                    branch_coverage_values.append(latest_data['branch_coverage'])
        

            if code_coverage_values:
                # Separate records for code and branch coverage
                code_coverage_record = {
                    'algorithm': algorithm,
                    'time_elapsed': time_point,
                    'coverage_type': 'code',
                    'mean': np.mean(code_coverage_values),
                    'std': np.std(code_coverage_values),
                    'min': np.min(code_coverage_values),
                    'max': np.max(code_coverage_values),
                    'sample_size': len(code_coverage_values)
                }
                smooth_records.append(code_coverage_record)    
                
            if branch_coverage_values:
                branch_coverage_record = {
                    'algorithm': algorithm,
                    'time_elapsed': time_point,
                    'coverage_type': 'branch',
                    'mean': np.mean(branch_coverage_values),
                    'std': np.std(branch_coverage_values),
                    'min': np.min(branch_coverage_values),
                    'max': np.max(branch_coverage_values),
                    'sample_size': len(branch_coverage_values)
                }
                smooth_records.append(branch_coverage_record)
    
    return pd.DataFrame(smooth_records)

def create_conference_style_plots_with_std(df: pd.DataFrame, smooth_df: pd.DataFrame, output_dir: str = "conference_plots", max_time: float = 600.0):
    """
    Create conference-style plots with standard deviation borders.
    
    Args:
        df: Original DataFrame
        smooth_df: Smoothed DataFrame with std calculations
        output_dir: Output directory
        max_time: Maximum time to show on X-axis (upper bound)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style for academic/conference papers
    plt.style.use('default')
    sns.set_style("whitegrid")
    
    # Enhanced color palette for algorithms
    algorithm_colors = {
        'collaborative': '#1f77b4',  # Blue
        'confuzzius': '#ff7f0e',     # Orange

        'default': '#7f7f7f'      # Gray for others
    }
    
    # Plot 1: Main code Coverage Comparison with STD
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    _plot_code_coverage_with_std(ax, smooth_df, algorithm_colors, max_time)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'code_coverage_with_std.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    # plt.savefig(os.path.join(output_dir, 'code_coverage_with_std.pdf'), 
    #             bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Plot 2: Small vs Large Contracts with STD
    _create_contract_size_plots_with_std(df, output_dir, algorithm_colors, max_time)
    
    # Plot 3: All coverage types with STD
    _create_detailed_coverage_plots_with_std(smooth_df, output_dir, algorithm_colors, max_time)

def _plot_code_coverage_with_std(ax, smooth_df, algorithm_colors, max_time):
    """
    Plot code coverage with standard deviation borders.
    """
    if smooth_df.empty:
        ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes, 
                ha='center', va='center', fontsize=12)
        return
    
    # Plot each algorithm with std borders
    for algorithm in smooth_df['algorithm'].unique():
        algorithm_data = smooth_df[
                (smooth_df['algorithm'] == algorithm) & 
                (smooth_df['coverage_type'] == 'code')
            ].sort_values('time_elapsed')
        
        
        if algorithm_data.empty:
            continue
            
        color = algorithm_colors.get(algorithm, algorithm_colors['default'])
        
        # Plot mean line
        ax.plot(algorithm_data['time_elapsed'], algorithm_data['mean'],
               color=color, linewidth=3, label=algorithm, alpha=0.9)
        
        # Plot standard deviation area
        ax.fill_between(algorithm_data['time_elapsed'],
                       algorithm_data['mean'] - algorithm_data['std'],
                       algorithm_data['mean'] + algorithm_data['std'],
                       color=color, alpha=0.2, label=f'{algorithm} ±1σ')
        
        # Plot min/max borders (optional, more transparent)
        ax.fill_between(algorithm_data['time_elapsed'],
                       algorithm_data['min'],
                       algorithm_data['max'],
                       color=color, alpha=0.1, label=f'{algorithm} range')
    
    # Enhanced styling
    ax.set_xlabel('Time in Seconds', fontsize=14, fontweight='bold')
    ax.set_ylabel('Code Coverage (%)', fontsize=14, fontweight='bold')
    ax.set_title('Overall Code Coverage Over Time with Variability', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Set axis limits and formatting
    # ax.set_ylim(0, 100)
    ax.set_xlim(0, max_time)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))
    
    # # Set x-axis ticks
    # x_ticks = np.arange(0, max_time + 100, 100)
    # ax.set_xticks(x_ticks)
    # ax.set_xticklabels([str(int(x)) for x in x_ticks])
    
    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Improved legend
    handles, labels = ax.get_legend_handles_labels()
    # Filter to show only mean lines in legend to avoid clutter
    mean_handles = [h for h, l in zip(handles, labels) if '±' not in l and 'range' not in l]
    mean_labels = [l for l in labels if '±' not in l and 'range' not in l]
    ax.legend(mean_handles, mean_labels, 
              bbox_to_anchor=(1.05, 1), 
              loc='upper left', 
              frameon=True,
              fancybox=True,
              shadow=True,
              framealpha=0.9,
              fontsize=10)

def _create_contract_size_plots_with_std(df, output_dir, algorithm_colors, max_time):
    """
    Create separate plots for small vs large contracts with STD.
    """
    if df.empty:
        return
    
    # Classify contracts as small or large based on final transaction count
    contract_sizes = {}
    for contract in df['contract'].unique():
        contract_data = df[df['contract'] == contract]
        max_tx = contract_data['total_transactions'].max()
        contract_sizes[contract] = 'Small Contracts' if max_tx <= 3632 else 'Large Contracts'
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, size_category in enumerate(['Small Contracts', 'Large Contracts']):
        ax = axes[idx]
        
        # Filter data for this size category
        size_contracts = [c for c, s in contract_sizes.items() if s == size_category]
        size_data = df[df['contract'].isin(size_contracts)]
        
        if size_data.empty:
            ax.text(0.5, 0.5, f'No {size_category.lower()} data', 
                    transform=ax.transAxes, ha='center', va='center', fontsize=12)
            continue
        
        # Create smooth time series for this size category
        smooth_size_data = create_smooth_time_series_with_std(size_data, max_time=max_time)
        
        if smooth_size_data.empty:
            continue
        
        # Plot each algorithm
        for algorithm in smooth_size_data['algorithm'].unique():
            algorithm_data = smooth_size_data[
                (smooth_size_data['algorithm'] == algorithm) & 
                (smooth_size_data['coverage_type'] == 'code')
            ].sort_values('time_elapsed')
            color = algorithm_colors.get(algorithm, algorithm_colors['default'])
            
            # Plot mean line
            ax.plot(algorithm_data['time_elapsed'], algorithm_data['mean'],
                   color=color, linewidth=2.5, label=algorithm)
            
            # Plot standard deviation area
            ax.fill_between(algorithm_data['time_elapsed'],
                           algorithm_data['mean'] - algorithm_data['std'],
                           algorithm_data['mean'] + algorithm_data['std'],
                           color=color, alpha=0.2)
        
        # Styling
        ax.set_xlabel('Time in Seconds', fontsize=12, fontweight='bold')
        ax.set_ylabel('Instruction Coverage (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{size_category}', fontsize=14, fontweight='bold')
        # ax.set_ylim(0, 100)
        ax.set_xlim(0, max_time)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'contract_size_comparison_with_std.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def _create_detailed_coverage_plots_with_std(smooth_data, output_dir, algorithm_colors, max_time):
    """
    Create detailed plots for all coverage types with STD.
    """
    coverage_types = ['code', 'branch']
    titles = [ 'Code Coverage', 'Branch Coverage']
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    
    for idx, (coverage_type, title) in enumerate(zip(coverage_types, titles)):
        ax = axes[idx]
        
        
        if smooth_data.empty:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, 
                    ha='center', va='center', fontsize=12)
            continue
        
        # Plot each algorithm
        for algorithm in smooth_data['algorithm'].unique():
            algorithm_data = smooth_data[
                (smooth_data['algorithm'] == algorithm) & 
                (smooth_data['coverage_type'] == coverage_type)
            ].sort_values('time_elapsed')
            color = algorithm_colors.get(algorithm, algorithm_colors['default'])
            
            ax.plot(algorithm_data['time_elapsed'], algorithm_data['mean'],
                   color=color, linewidth=2, label=algorithm)
            
            ax.fill_between(algorithm_data['time_elapsed'],
                           algorithm_data['mean'] - algorithm_data['std'],
                           algorithm_data['mean'] + algorithm_data['std'],
                           color=color, alpha=0.2)
        
        # Styling
        ax.set_xlabel('Time (seconds)', fontsize=11)
        ax.set_ylabel(f'{title} (%)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        # ax.set_ylim(0, 100)
        ax.set_xlim(0, max_time)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))
        ax.grid(True, alpha=0.3)
        
        if idx == 0:  # Only show legend on first plot
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detailed_coverage_with_std.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def generate_enhanced_report(df: pd.DataFrame, smooth_df: pd.DataFrame, output_dir: str = "conference_plots"):
    """
    Generate enhanced report with variability statistics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, 'performance_analysis_with_variability.txt')
    
    with open(report_path, 'w') as f:
        f.write("PERFORMANCE ANALYSIS WITH VARIABILITY\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Variability statistics
        f.write("COVERAGE VARIABILITY BY ALGORITHM:\n")
        f.write("-" * 35 + "\n")
        
        if not smooth_df.empty:
            variability_stats = smooth_df.groupby('algorithm').agg({
                'mean': ['mean', 'max'],
                'std': ['mean', 'max'],
                'sample_size': 'mean'
            }).round(2)
            
            f.write(variability_stats.to_string())
            f.write("\n\n")
        
        # Final coverage with variability
        f.write("FINAL COVERAGE STATISTICS:\n")
        f.write("-" * 30 + "\n")
        
        final_coverage = df.groupby(['algorithm', 'contract']).agg({
            'code_coverage': 'last'
        }).reset_index()
        
        final_stats = final_coverage.groupby('algorithm').agg({
            'code_coverage': ['mean', 'std', 'min', 'max']
        }).round(2)
        
        f.write(final_stats.to_string())
        f.write("\n\n")

def create_conference_style_graphics_with_std(folder_path: str, output_dir: str = "conference_plots", 
                                            time_interval: float = 2.0, max_time: float = 600.0):
    """
    Main function to create conference-style graphics with standard deviation.
    
    Args:
        folder_path: Path to folder containing JSON files (searched recursively)
        output_dir: Output directory for plots and reports
        time_interval: Time interval for sampling in seconds
        max_time: Maximum time to show on X-axis (upper bound)
    """
    print(f"Creating conference-style graphics with STD from: {folder_path}")
    print(f"X-axis upper bound: {max_time} seconds")
    
    # Read JSON files recursively
    json_data = read_json_files(folder_path)
    
    if not json_data:
        print("No valid JSON files found.")
        return
    
    # Create DataFrame
    df = create_coverage_dataframe(json_data)
    
    if df.empty:
        print("No coverage data found.")
        return
    
    print(f"Analyzing data from {len(df['algorithm'].unique())} algorithms")
    print(f"Algorithms found: {df['algorithm'].unique().tolist()}")
    print(f"Time range in data: {df['time_elapsed'].min():.1f}s to {df['time_elapsed'].max():.1f}s")
    
    # Create smooth time series with standard deviation
    smooth_df = create_smooth_time_series_with_std(df, time_interval, max_time)
    
    if smooth_df.empty:
        print("No data available after time filtering.")
        return
    
    # Create conference-style plots with STD
    create_conference_style_plots_with_std(df, smooth_df, output_dir, max_time)
    
    # Generate enhanced report
    generate_enhanced_report(df, smooth_df, output_dir)
    
    # Save data
    df.to_csv(os.path.join(output_dir, 'performance_data.csv'), index=False)
    smooth_df.to_csv(os.path.join(output_dir, 'smooth_performance_with_std.csv'), index=False)
    
    print(f"\nConference-style graphics with STD created in: {output_dir}")
    print("Files generated:")
    print("  - code_coverage_with_std.png (Main plot with variability)")
    print("  - contract_size_comparison_with_std.png")
    print("  - detailed_coverage_with_std.png")
    print("  - performance_analysis_with_variability.txt")
    print("  - smooth_performance_with_std.csv (Data with mean and std)")
    
    return df, smooth_df

# Example usage with configurable max_time
if __name__ == "__main__":
    # You can change max_time to any value you want (e.g., 300, 600, 900 seconds)
    folder_path = "results"
    max_time_seconds = 60  # Change this value to set X-axis upper bound
    
    df, smooth_df = create_conference_style_graphics_with_std(
        folder_path=folder_path,
        output_dir="conference_results_with_std",
        time_interval=2.0,
        max_time=max_time_seconds
    )
    
    if df is not None:
        print(f"\nAnalysis completed with X-axis limit: {max_time_seconds} seconds")
        print("\nSample of smoothed data with STD:")
        print(smooth_df[['algorithm', 'time_elapsed', 'mean', 'std']].head(10))