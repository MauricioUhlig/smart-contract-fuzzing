import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import sys
import argparse

def create_conference_style_plots_with_std(df, output_dir, max_time):
    """
    Create conference-style plots using the aggregated data.
    df contains columns: algorithm, time_elapsed, coverage_type, category, mean, std, min, max, sample_size
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use('default')
    sns.set_style("whitegrid")
    
    algorithm_colors = {
        'divfuzz': '#000000',  # DivFuzz
        'confuzzius': '#808080',     # Confuzzius
        'default': '#999999'
    }
    
    # Plot 1: Main code coverage with STD (global)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    global_df = df[df['category'] == 'global']
    _plot_coverage_with_std(ax, global_df, algorithm_colors, max_time, 'code')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'code_coverage_with_std.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Plot 2: Small vs Large contracts
    _create_contract_size_plots(df, output_dir, algorithm_colors, max_time)
    
    # Plot 3: Detailed coverage (code and branch) using global data
    _create_detailed_coverage_plots(df, output_dir, algorithm_colors, max_time)

def _plot_coverage_with_std(ax, df, algorithm_colors, max_time, coverage_type):
    """
    Plot coverage type (code or branch) for a given dataframe (which already has category filter).
    """
    if df.empty:
        ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes, 
                ha='center', va='center', fontsize=14)
        return
    
    data = df[df['coverage_type'] == coverage_type]
    if data.empty:
        ax.text(0.5, 0.5, f'No {coverage_type} data', transform=ax.transAxes,
                ha='center', va='center', fontsize=14)
        return
    
    for algorithm in np.flip(data['algorithm'].unique()):
        algo_data = data[data['algorithm'] == algorithm].sort_values('time_elapsed')
        if algo_data.empty:
            continue
        
        color = algorithm_colors.get(algorithm, algorithm_colors['default'])
        final_mean = algo_data['mean'].iloc[-1]
        label = '{}'.format(algorithm if algorithm != "divfuzz" else "DivFuzz")
        label = '{}'.format(label if label != "confuzzius" else "Confuzzius")
        
        ax.plot(algo_data['time_elapsed'], algo_data['mean'],
                color=color, linewidth=3, label=label, alpha=0.9)
        
        # Optional: standard deviation area
        # ax.fill_between(algo_data['time_elapsed'],
        #                 algo_data['mean'] - algo_data['std'],
        #                 algo_data['mean'] + algo_data['std'],
        #                 color=color, alpha=0.2)
        
        ax.annotate(f'{final_mean:.1f}%', 
                    xy=(algo_data['time_elapsed'].iloc[-1], final_mean),
                    xytext=(10, 0), textcoords='offset points',
                    color=color, fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=color, alpha=0.8))
    
    # Styling
    if coverage_type == 'code':
        ylabel = 'Cobertura de código (%)'
        title = 'Cobertura de Código Geral ao Longo do Tempo'
    else:
        ylabel = 'Cobertura de ramificações (%)'
        title = 'Cobertura de Ramificações ao Longo do Tempo'
    
    ax.set_xlabel('Tempo em Segundos', fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(0, max_time)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower right', fontsize=14)

def _create_contract_size_plots(df, output_dir, algorithm_colors, max_time):
    """
    Create side-by-side plots for small and large contracts.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, category in enumerate(['small', 'large']):
        ax = axes[idx]
        cat_df = df[(df['category'] == category) & (df['coverage_type'] == 'code')]
        
        if cat_df.empty:
            ax.text(0.5, 0.5, f'No {category} data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)
            continue
        
        for algorithm in np.flip(cat_df['algorithm'].unique()):
            algo_data = cat_df[cat_df['algorithm'] == algorithm].sort_values('time_elapsed')
            if algo_data.empty:
                continue
            
            color = algorithm_colors.get(algorithm, algorithm_colors['default'])
            final_mean = algo_data['mean'].iloc[-1]
            label = '{}'.format(algorithm if algorithm != "divfuzz" else "DivFuzz")
            label = '{}'.format(label if label != "confuzzius" else "Confuzzius")
            
            ax.plot(algo_data['time_elapsed'], algo_data['mean'],
                    color=color, linewidth=2.5, label=label)
            
            # Optional: standard deviation area
            # ax.fill_between(algo_data['time_elapsed'],
            #                 algo_data['mean'] - algo_data['std'],
            #                 algo_data['mean'] + algo_data['std'],
            #                 color=color, alpha=0.2)
            
            ax.annotate(f'{final_mean:.1f}%', 
                       xy=(algo_data['time_elapsed'].iloc[-1], final_mean),
                       xytext=(10, 0), textcoords='offset points',
                       color=color, fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor=color, alpha=0.8))
        
        ax.set_xlabel('Tempo em Segundos', fontsize=14, fontweight='bold')
        ax.set_ylabel('Cobertura de código (%)', fontsize=14, fontweight='bold')
        title = 'Contratos pequenos' if category == 'small' else 'Contratos grandes'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim(0, max_time)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))
        ax.grid(True, alpha=0.3)
        
        # Legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='lower right', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'contract_size_comparison_with_std.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def _create_detailed_coverage_plots(df, output_dir, algorithm_colors, max_time):
    """
    Create side-by-side plots for code and branch coverage (global).
    """
    global_df = df[df['category'] == 'global']
    coverage_types = ['code', 'branch']
    titles = ['Cobertura de Código', 'Cobertura de Ramificações']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, (cov_type, title) in enumerate(zip(coverage_types, titles)):
        ax = axes[idx]
        data = global_df[global_df['coverage_type'] == cov_type]
        if data.empty:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)
            continue
        
        for algorithm in np.flip(data['algorithm'].unique()):
            algo_data = data[data['algorithm'] == algorithm].sort_values('time_elapsed')
            if algo_data.empty:
                continue
            
            color = algorithm_colors.get(algorithm, algorithm_colors['default'])
            final_mean = algo_data['mean'].iloc[-1]
            label = '{}'.format(algorithm if algorithm != "divfuzz" else "DivFuzz")
            label = '{}'.format(label if label != "confuzzius" else "Confuzzius")
            
            ax.plot(algo_data['time_elapsed'], algo_data['mean'],
                   color=color, linewidth=2, label=label)
            
            # Optional: standard deviation area
            # ax.fill_between(algo_data['time_elapsed'],
            #                 algo_data['mean'] - algo_data['std'],
            #                 algo_data['mean'] + algo_data['std'],
            #                 color=color, alpha=0.2)
            
            ax.annotate(f'{final_mean:.1f}%', 
                       xy=(algo_data['time_elapsed'].iloc[-1], final_mean),
                       xytext=(10, 0), textcoords='offset points',
                       color=color, fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor=color, alpha=0.8))
        
        ax.set_xlabel('Tempo em Segundos', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'{title} (%)', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlim(0, max_time)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(25))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detailed_coverage_with_std.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate plots from aggregated data.')
    parser.add_argument('--output_dir', default='conference_results_with_std', help='Directory containing CSV file')
    parser.add_argument('--max_time', type=float, default=600.0, help='Maximum time for X-axis')
    args = parser.parse_args()
    
    csv_path = os.path.join(args.output_dir, 'aggregated_smooth_data.csv')
    if not os.path.exists(csv_path):
        print(f"Aggregated CSV not found: {csv_path}. Run Go processing first.")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    create_conference_style_plots_with_std(df, args.output_dir, args.max_time)
    print(f"Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()