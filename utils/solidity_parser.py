#!/usr/bin/env python3
"""
Script to extract contract names and annotations from Solidity files (.sol) in a given folder.
"""

import os
import re
import sys
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

def extract_annotations_and_contracts(file_path: str) -> Dict[str, Any]:
    """
    Extract contract names and annotations from a Solidity file.
    
    Args:
        file_path (str): Path to the Solidity file
        
    Returns:
        Dict[str, Any]: Dictionary containing file info with contracts and their annotations
    """
    result = {
        "file": str(file_path),
        "contracts": []
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # Remove comments for contract extraction first
            content_no_comments = re.sub(r'//.*', '', content)
            content_no_comments = re.sub(r'/\*.*?\*/', '', content_no_comments, flags=re.DOTALL)
            
            # Extract contract names with their positions
            contract_patterns = [
                (r'\bcontract\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:is\s+[^\{]*?)?\{', 'contract'),
                (r'\babstract\s+contract\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:is\s+[^\{]*?)?\{', 'abstract contract'),
                (r'\binterface\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:is\s+[^\{]*?)?\{', 'interface'),
                (r'\blibrary\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\{', 'library')
            ]
            
            contracts_info = []
            for pattern, contract_type in contract_patterns:
                matches = re.finditer(pattern, content_no_comments)
                for match in matches:
                    contract_name = match.group(1)
                    # Get line number of contract start
                    lines_before = content[:match.start()].count('\n') + 1
                    contracts_info.append({
                        "name": contract_name,
                        "type": contract_type,
                        "start_line": lines_before,
                        "annotations": {
                            "vulnerable_lines": [],
                            "reports": []
                        }
                    })
            
            # Now extract annotations and associate them with contracts
            lines = content.split('\n')
            
            # Extract multi-line comment annotations and associate with contracts
            for i, line in enumerate(lines, 1):
                # Check for multi-line comment start
                if '/*' in line and '*/' not in line:
                    # Multi-line comment spanning multiple lines
                    comment_lines = []
                    j = i
                    while j <= len(lines):
                        comment_lines.append(lines[j-1])
                        if '*/' in lines[j-1]:
                            break
                        j += 1
                    
                    comment_text = '\n'.join(comment_lines)
                    
                    # Extract vulnerable lines from multi-line comments
                    vulnerable_match = re.search(r'@vulnerable_at_lines?:\s*([\d,\s]+)', comment_text)
                    if vulnerable_match:
                        lines_str = vulnerable_match.group(1)
                        vuln_lines = [int(line.strip()) for line in lines_str.split(',') if line.strip().isdigit()]
                        
                        # Associate vulnerable lines with contracts
                        for contract in contracts_info:
                            for vuln_line in vuln_lines:
                                if contract["start_line"] <= vuln_line:
                                    # Find the next contract to determine range
                                    next_contract_start = None
                                    for other_contract in contracts_info:
                                        if other_contract["start_line"] > contract["start_line"]:
                                            if next_contract_start is None or other_contract["start_line"] < next_contract_start:
                                                next_contract_start = other_contract["start_line"]
                                    
                                    # If vuln_line is before next contract or no next contract, assign to this contract
                                    if next_contract_start is None or vuln_line < next_contract_start:
                                        if vuln_line not in contract["annotations"]["vulnerable_lines"]:
                                            contract["annotations"]["vulnerable_lines"].append(vuln_line)
            
            # Extract single-line annotations and associate with contracts
            for i, line in enumerate(lines, 1):
                # Single-line report annotations
                report_match = re.search(r'//\s*<yes>\s*<report>\s*([^\s]+)', line)
                if report_match:
                    report_type = report_match.group(1).strip()
                    report_type = _normalize_vulnerability_name(report_type)
                    # Find which contract this line belongs to
                    for contract in contracts_info:
                        next_contract_start = None
                        for other_contract in contracts_info:
                            if other_contract["start_line"] > contract["start_line"]:
                                if next_contract_start is None or other_contract["start_line"] < next_contract_start:
                                    next_contract_start = other_contract["start_line"]
                        
                        # Check if this line is within the current contract's range
                        if next_contract_start is None:
                            if i >= contract["start_line"]:
                                contract["annotations"]["reports"].append(report_type)
                        else:
                            if contract["start_line"] <= i < next_contract_start:
                                contract["annotations"]["reports"].append(report_type)
                
                # Single-line vulnerable_at_lines annotations
                vuln_match = re.search(r'//\s*@vulnerable_at_lines?:\s*([\d,\s]+)', line)
                if vuln_match:
                    lines_str = vuln_match.group(1)
                    vuln_lines = [int(line.strip()) for line in lines_str.split(',') if line.strip().isdigit()]
                    
                    # Associate vulnerable lines with contracts
                    for contract in contracts_info:
                        for vuln_line in vuln_lines:
                            if contract["start_line"] <= vuln_line:
                                next_contract_start = None
                                for other_contract in contracts_info:
                                    if other_contract["start_line"] > contract["start_line"]:
                                        if next_contract_start is None or other_contract["start_line"] < next_contract_start:
                                            next_contract_start = other_contract["start_line"]
                                
                                if next_contract_start is None or vuln_line < next_contract_start:
                                    if vuln_line not in contract["annotations"]["vulnerable_lines"]:
                                        contract["annotations"]["vulnerable_lines"].append(vuln_line)
            
            # Sort the lists and add to result
            for contract in contracts_info:
                if contract["annotations"]["vulnerable_lines"] == [] and contract["annotations"]["reports"] == []:
                   final_contract = {
                    "name": contract["name"]
                   }
                else:
                    # Remove type and start_line from final output
                    final_contract = {
                        "name": contract["name"],
                        "annotations": contract["annotations"]
                    }
                result["contracts"].append(final_contract)
            
    except Exception as e:
        print(f"Error reading file {file_path}: {e}", file=sys.stderr)
        result["error"] = str(e)
    
    return result

def _normalize_vulnerability_name(name: str) -> str:
    match (name):
        case ("ARITHMETIC"):
            return "INTEGER_OVERFLOW" 
        case ("FRONT_RUNNING"):
            return "TRANSACTION_ORDER_DEPENDENCY"
        case ("BAD_RANDOMNESS"):
            return "BLOCK_DEPENDENCY"
        case ("TIME"):
            return "BLOCK_DEPENDENCY"
        case ("TIME_MANIPULATION"):
            return "BLOCK_DEPENDENCY"
        case ("UNCHECKED_LL_CALLS"):
            return "UNHANDLED_EXCEPTION"
        
        case (_):
            return name

def get_solidity_contracts_with_annotations(folder_path: str) -> List[Dict[str, Any]]:
    """
    Get all Solidity files with their contracts and annotations.
    
    Args:
        folder_path (str): Path to the folder containing Solidity files
        
    Returns:
        List[Dict[str, Any]]: List of file information with contracts and annotations
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder '{folder_path}' does not exist")
    
    if not folder.is_dir():
        raise NotADirectoryError(f"'{folder_path}' is not a directory")
    
    # Find all .sol files recursively
    sol_files = list(folder.rglob("*.sol"))
    
    if not sol_files:
        print(f"No .sol files found in '{folder_path}'")
        return []
    
    results = []
    
    for sol_file in sol_files:
        file_info = extract_annotations_and_contracts(sol_file)
        # Use relative path for cleaner output
        file_info["file"] = str(sol_file.relative_to(folder))
        # Remove empty contracts array if no contracts found
        if not file_info["contracts"]:
            file_info["contracts"] = []
        results.append(file_info)
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Extract contract names and annotations from Solidity files in a folder"
    )
    parser.add_argument(
        "folder",
        help="Path to the folder containing Solidity files (.sol)"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Search recursively in subdirectories (default: True)"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results in JSON format"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path for JSON results"
    )
    
    args = parser.parse_args()
    
    try:
        files_info = get_solidity_contracts_with_annotations(args.folder)
        
        if not files_info:
            if args.json or args.output:
                output_json = []
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(output_json, f, indent=2)
                    print(f"Empty results written to {args.output}")
                else:
                    print("[]")
            else:
                print("No contracts found.")
            return
        
        if args.json or args.output:
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(files_info, f)
                print(f"Results written to {args.output}")
            else:
                print(json.dumps(files_info, indent=2))
        else:
            print(f"\nFound {len(files_info)} Solidity file(s):\n")
            
            for file_info in files_info:
                print(f"📄 {file_info['file']}:")
                
                if file_info['contracts']:
                    for contract in file_info['contracts']:
                        print(f"   📝 Contract: {contract['name']}")
                        
                        annotations = contract['annotations']
                        if annotations['vulnerable_lines']:
                            vuln_str = ", ".join(map(str, annotations['vulnerable_lines']))
                            print(f"      🚨 Vulnerable lines: [{vuln_str}]")
                        
                        if annotations['reports']:
                            reports_str = ", ".join(annotations['reports'])
                            print(f"      📋 Reports: {reports_str}")
                else:
                    print(f"   ⚠️  No contracts found")
                
                print()
            
    except Exception as e:
        if args.json or args.output:
            error_output = {"error": str(e)}
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(error_output, f, indent=2)
                print(f"Error written to {args.output}")
            else:
                print(json.dumps(error_output, indent=2))
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()