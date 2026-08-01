#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from z3 import simplify
from utils.utils import convert_stack_value_to_int

class ReentrancyDetector():
    def __init__(self):
        self.init()

    def init(self):
        self.swc_id = 107
        self.severity = "High"
        # Stores the PC (Program Counter) and transaction index of each SLOAD, keyed by storage index
        self.sloads = {}
        # FIX: Stores the PC and transaction index of each SSTORE, keyed by storage index
        # This allows us to check if the state was updated before an external call.
        self.sstores = {}
        # Stores the PCs of external CALLs that could be dangerous (gas > 2300, value > 0 or tainted)
        self.calls = set()

    def detect_reentrancy(self, tainted_record, current_instruction, transaction_index):
        # 1. REMEMBER STORAGE READS (SLOAD)
        if current_instruction["op"] == "SLOAD":
            if tainted_record and tainted_record.stack and tainted_record.stack[-1]:
                storage_index = convert_stack_value_to_int(current_instruction["stack"][-1])
                self.sloads[storage_index] = current_instruction["pc"], transaction_index

        # 2. REMEMBER STORAGE WRITES (SSTORE) AND CHECK CLASSIC REENTRANCY (SLOAD -> CALL -> SSTORE)
        elif current_instruction["op"] == "SSTORE":
            storage_index = convert_stack_value_to_int(current_instruction["stack"][-1])
            
            # Persist the SSTORE record for future CALL checks
            self.sstores[storage_index] = current_instruction["pc"], transaction_index

            # If a CALL happened BEFORE this SSTORE, it is a TRUE POSITIVE (vulnerable)
            if tainted_record and tainted_record.stack and tainted_record.stack[-1]:
                if self.calls and storage_index in self.sloads:
                    for pc, index in self.calls:
                        if pc < current_instruction["pc"]:
                            return pc, index

        # 3. CHECK EXTERNAL CALLS (CALL) AND FILTER OUT FALSE POSITIVES
        elif current_instruction["op"] == "CALL" and self.sloads:
            gas = convert_stack_value_to_int(current_instruction["stack"][-1])
            value = convert_stack_value_to_int(current_instruction["stack"][-3])
            
            # Case 1: The call sends Ether (value > 0 or symbolic)
            if gas > 2300 and (value > 0 or (tainted_record and tainted_record.stack and tainted_record.stack[-3])):
                self.calls.add((current_instruction["pc"], transaction_index))
                
                # Iterate over each SLOAD individually
                for storage_index, (pc_sload, index) in self.sloads.items():
                    if pc_sload < current_instruction["pc"]:
                        # Check if the state was safely updated (CEI pattern)
                        # We look for an SSTORE on this specific key that happened AFTER the SLOAD but BEFORE this CALL
                        state_updated = False
                        if storage_index in self.sstores:
                            pc_sstore, _ = self.sstores[storage_index]
                            if pc_sstore > pc_sload and pc_sstore < current_instruction["pc"]:
                                state_updated = True
                        
                        # If the state was NOT updated before the call, it's a real vulnerability (or high risk)
                        if not state_updated:
                            return current_instruction["pc"], index

            # Case 2: The destination (msg.sender) is controlled by the user (tainted)
            if gas > 2300 and tainted_record and tainted_record.stack and tainted_record.stack[-2]:
                self.calls.add((current_instruction["pc"], transaction_index))
                
                # Apply the same state-update check here
                for storage_index, (pc_sload, index) in self.sloads.items():
                    if pc_sload < current_instruction["pc"]:
                        state_updated = False
                        if storage_index in self.sstores:
                            pc_sstore, _ = self.sstores[storage_index]
                            if pc_sstore > pc_sload and pc_sstore < current_instruction["pc"]:
                                state_updated = True
                        if not state_updated:
                            return current_instruction["pc"], index

        # 4. CLEAR STATE AT THE END OF THE TRANSACTION
        elif current_instruction["op"] in ["STOP", "RETURN", "REVERT", "ASSERTFAIL", "INVALID", "SUICIDE", "SELFDESTRUCT"]:
            self.sloads = {}
            self.sstores = {}   # FIX 4: Clear the writes as well
            self.calls = set()

        return None, None