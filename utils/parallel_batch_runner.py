#!/usr/bin/env python3
"""
Enhanced multicore batch runner for fuzzing
"""

import json
import sys
import os
import subprocess
import argparse
import concurrent.futures
import multiprocessing
from typing import List, Dict, Any
import time


def load_json_config(json_file: str) -> List[Dict[str, Any]]:
    """Load and parse the JSON configuration file"""
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file '{json_file}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{json_file}': {e}")
        sys.exit(1)


def invoke_fuzzer(base_path: str, task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Invoke the fuzzer with the specified arguments
    Returns: Dictionary with execution results
    """
    source_file = task['source_file']
    contract_name = task['contract_name']
    extra_args = task['extra_args']
    
    # Build the command
    cmd = [
        "python3", "fuzzer/main.py",
        "--source", base_path + source_file,
        "--contract", contract_name,
        "-r", f"results/{extra_args[(extra_args.index('--algorithm')+1)]}/{source_file.replace('/', '-').replace('.sol', '.json')}"
    ]
    
    # Add extra arguments
    cmd.extend(extra_args)
    
    print(f"[PID {os.getpid()}] Executing: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        # Execute the command with timeout (6 hours default)
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True,
            timeout=21600  # 6 hours timeout
        )
        execution_time = time.time() - start_time
        
        return {
            'success': True,
            'source_file': source_file,
            'contract_name': contract_name,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'execution_time': execution_time,
            'pid': os.getpid()
        }
        
    except subprocess.CalledProcessError as e:
        execution_time = time.time() - start_time
        return {
            'success': False,
            'source_file': source_file,
            'contract_name': contract_name,
            'stdout': e.stdout,
            'stderr': e.stderr,
            'exit_code': e.returncode,
            'execution_time': execution_time,
            'pid': os.getpid()
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'source_file': source_file,
            'contract_name': contract_name,
            'error': 'Timeout (6 hours) exceeded',
            'execution_time': 21600,
            'pid': os.getpid()
        }
    except FileNotFoundError:
        return {
            'success': False,
            'source_file': source_file,
            'contract_name': contract_name,
            'error': 'fuzzer/main.py not found',
            'pid': os.getpid()
        }


def create_tasks(config: List[Dict[str, Any]], extra_args: List[str]) -> List[Dict[str, Any]]:
    """Create task list from configuration"""
    tasks = []
    
    for file_config in config:
        source_file = file_config.get("file")
        contracts = file_config.get("contracts", [])
        
        if not source_file:
            print("Warning: Missing 'file' field in configuration, skipping...")
            continue
        
        # Process each contract in the file
        for contract_config in contracts:
            contract_name = contract_config.get("name")
            annotations = contract_config.get("annotations", {})
            
            if not contract_name:
                print(f"Warning: Missing 'name' field for contract in {source_file}, skipping...")
                continue
            
            task = {
                'source_file': source_file,
                'contract_name': contract_name,
                'annotations': annotations,
                'extra_args': extra_args.copy()
            }
            tasks.append(task)
    
    return tasks

def run_parallel(tasks: List[Dict[str, Any]], base_path: str, max_workers: int = None) -> List[Dict[str, Any]]:
    """
    Run tasks in parallel using ThreadPoolExecutor
    Note: Using threads instead of processes to avoid pickling issues
    """
    if max_workers is None:
        max_workers = min(len(tasks), multiprocessing.cpu_count())
    
    print(f"Running {len(tasks)} tasks with {max_workers} workers")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(invoke_fuzzer, base_path, task): task for task in tasks}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                
                # Print progress
                success_count = sum(1 for r in results if r.get('success', False))
                print(f"Progress: {len(results)}/{len(tasks)} completed ({success_count} successful)")
                
            except Exception as exc:
                print(f"Task {task['source_file']}->{task['contract_name']} generated an exception: {exc}")
                results.append({
                    'success': False,
                    'source_file': task['source_file'],
                    'contract_name': task['contract_name'],
                    'error': str(exc)
                })
    
    return results


# def run_process_pool(tasks: List[Dict[str, Any]], max_workers: int = None) -> List[Dict[str, Any]]:
#     """
#     Run tasks using ProcessPoolExecutor (more isolated, but may have pickling issues)
#     Use this if your fuzzer is CPU-intensive and threads don't provide true parallelism
#     """
#     if max_workers is None:
#         max_workers = min(len(tasks), multiprocessing.cpu_count())
    
#     print(f"Running {len(tasks)} tasks with {max_workers} processes")
    
#     results = []
#     with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
#         future_to_task = {executor.submit(invoke_fuzzer, task): task for task in tasks}
        
#         for future in concurrent.futures.as_completed(future_to_task):
#             task = future_to_task[future]
#             try:
#                 result = future.result()
#                 results.append(result)
                
#                 success_count = sum(1 for r in results if r.get('success', False))
#                 print(f"Progress: {len(results)}/{len(tasks)} completed ({success_count} successful)")
                
#             except Exception as exc:
#                 print(f"Task {task['source_file']}->{task['contract_name']} generated an exception: {exc}")
#                 results.append({
#                     'success': False,
#                     'source_file': task['source_file'],
#                     'contract_name': task['contract_name'],
#                     'error': str(exc)
#                 })
    
#     return results


def print_results_summary(results: List[Dict[str, Any]]) -> None:
    """Print detailed summary of execution results"""
    success_count = sum(1 for r in results if r.get('success', False))
    total_count = len(results)
    
    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    print(f"Total tasks: {total_count}")
    print(f"Successful:  {success_count}")
    print(f"Failed:      {total_count - success_count}")
    print(f"Success rate: {success_count/total_count*100:.1f}%")
    
    # Print timing information
    if any('execution_time' in r for r in results):
        successful_times = [r['execution_time'] for r in results if r.get('success', False)]
        if successful_times:
            avg_time = sum(successful_times) / len(successful_times)
            max_time = max(successful_times)
            min_time = min(successful_times)
            print(f"\nExecution times (successful tasks):")
            print(f"  Average: {avg_time/60:.1f} minutes")
            print(f"  Minimum: {min_time/60:.1f} minutes")
            print(f"  Maximum: {max_time/60:.1f} minutes")
    
    # Print failures
    failures = [r for r in results if not r.get('success', False)]
    if failures:
        print(f"\nFailed tasks:")
        for failure in failures:
            error_msg = failure.get('error') or failure.get('stderr', 'Unknown error')[:100]
            print(f"  - {failure['source_file']} -> {failure['contract_name']}: {error_msg}")


def main():
    parser = argparse.ArgumentParser(description='Multicore batch runner for fuzzing')
    parser.add_argument('json_file', help='Path to the JSON configuration file')
    parser.add_argument('base_path', help='Path to the contract files')
    parser.add_argument('--parallel', '-p', action='store_true', 
                       help='Run tasks in parallel using multiple cores')
    parser.add_argument('--processes', type=int, 
                       help='Number of parallel processes to use (default: CPU count)')
    parser.add_argument('--timeout', type=int, default=21600,
                       help='Timeout per task in seconds (default: 6 hours)')
    parser.add_argument('extra_args', nargs=argparse.REMAINDER, 
                       help='Extra arguments to pass to the fuzzer')
    
    args = parser.parse_args()
    
    # Load JSON configuration
    config = load_json_config(args.json_file)
    
    # Create tasks
    tasks = create_tasks(config, args.extra_args)
    
    if not tasks:
        print("No valid tasks found in configuration")
        sys.exit(1)
    
    print(f"Loaded {len(tasks)} fuzzing tasks")
    
    # Execute tasks
    start_time = time.time()
    
    
    if args.processes:
        results = run_parallel(tasks, args.base_path, max_workers=args.processes)
    else:
        results = run_parallel(tasks, args.base_path)
    
    total_time = time.time() - start_time
    
    # Print results
    print_results_summary(results)
    print(f"\nTotal batch execution time: {total_time/60:.1f} minutes")
    
    # Exit with appropriate code
    success_count = sum(1 for r in results if r.get('success', False))
    if success_count < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()