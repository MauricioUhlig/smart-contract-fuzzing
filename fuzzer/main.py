#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import solcx
import random
import logging
import argparse

from eth_utils import encode_hex, decode_hex, to_canonical_address
from z3 import Solver

from evm import InstrumentedEVM
from detectors import DetectorExecutor
from engine import EvolutionaryFuzzingEngine
from engine.pso_engine import ParticleSwarmEngine
from engine.collaborative_engine import CollaborativeEngine
from engine.pyswarms_engine import PySwarmsEngine
from engine.components import Generator, Individual, Population
from engine.analysis import SymbolicTaintAnalyzer
from engine.analysis import ExecutionTraceAnalyzer
from engine.environment import FuzzingEnvironment
from engine.operators import LinearRankingSelection
from engine.operators import DataDependencyLinearRankingSelection
from engine.operators import Crossover
from engine.operators import DataDependencyCrossover
from engine.operators import DiversityCrossover
from engine.operators import Mutation
from engine.operators import DiversityMutation
from engine.fitness import fitness_function

from utils import settings
from utils.source_map import SourceMap
from utils.utils import initialize_logger, compile, get_interface_from_abi, get_pcs_and_jumpis, get_function_signature_mapping
from utils.control_flow_graph import ControlFlowGraph

class Fuzzer:
    def __init__(self, contract_name, abi, deployment_bytecode, runtime_bytecode, test_instrumented_evm, blockchain_state, solver, args, seed, source_map=None):
        global logger

        logger = initialize_logger("Fuzzer  ")
        logger.title("Fuzzing contract %s", contract_name)

        cfg = ControlFlowGraph()
        cfg.build(runtime_bytecode, settings.EVM_VERSION)

        self.contract_name = contract_name
        self.interface = get_interface_from_abi(abi)
        self.deployement_bytecode = deployment_bytecode
        self.blockchain_state = blockchain_state
        self.instrumented_evm = test_instrumented_evm
        self.solver = solver
        self.args = args

        # Get some overall metric on the code
        self.overall_pcs, self.overall_jumpis = get_pcs_and_jumpis(runtime_bytecode)

        # Initialize results
        self.results = {"errors": {}}

        # Initialize fuzzing environment
        self.env = FuzzingEnvironment(instrumented_evm=self.instrumented_evm,
                                      contract_name=self.contract_name,
                                      solver=self.solver,
                                      results=self.results,
                                      symbolic_taint_analyzer=SymbolicTaintAnalyzer(),
                                      detector_executor=DetectorExecutor(source_map, get_function_signature_mapping(abi)),
                                      interface=self.interface,
                                      overall_pcs=self.overall_pcs,
                                      overall_jumpis=self.overall_jumpis,
                                      len_overall_pcs_with_children=0,
                                      other_contracts = list(),
                                      args=args,
                                      seed=seed,
                                      cfg=cfg,
                                      abi=abi)

    def run(self):
        contract_address = None
        self.instrumented_evm.create_fake_accounts()

        if self.args.source:
            for transaction in self.blockchain_state:
                if transaction['from'].lower() not in self.instrumented_evm.accounts:
                    self.instrumented_evm.accounts.append(self.instrumented_evm.create_fake_account(transaction['from']))

                if not transaction['to']:
                    result = self.instrumented_evm.deploy_contract(transaction['from'], transaction['input'], int(transaction['value']), int(transaction['gas']), int(transaction['gasPrice']))
                    if result.is_error:
                        logger.error("Problem while deploying contract %s using account %s. Error message: %s", self.contract_name, transaction['from'], result._error)
                        sys.exit(-2)
                    else:
                        contract_address = encode_hex(result.msg.storage_address)
                        self.instrumented_evm.accounts.append(contract_address)
                        self.env.nr_of_transactions += 1
                        logger.debug("Contract deployed at %s", contract_address)
                        self.env.other_contracts.append(to_canonical_address(contract_address))
                        cc, _ = get_pcs_and_jumpis(self.instrumented_evm.get_code(to_canonical_address(contract_address)).hex())
                        self.env.len_overall_pcs_with_children += len(cc)
                else:
                    input = {}
                    input["block"] = {}
                    input["transaction"] = {
                        "from": transaction["from"],
                        "to": transaction["to"],
                        "gaslimit": int(transaction["gas"]),
                        "value": int(transaction["value"]),
                        "data": transaction["input"]
                    }
                    input["global_state"] = {}
                    out = self.instrumented_evm.deploy_transaction(input, int(transaction["gasPrice"]))

            if "constructor" in self.interface:
                del self.interface["constructor"]

            if not contract_address:
                if "constructor" not in self.interface:
                    result = self.instrumented_evm.deploy_contract(self.instrumented_evm.accounts[0], self.deployement_bytecode)
                    if result.is_error:
                        logger.error("Problem while deploying contract %s using account %s. Error message: %s", self.contract_name, self.instrumented_evm.accounts[0], result._error)
                        sys.exit(-2)
                    else:
                        contract_address = encode_hex(result.msg.storage_address)
                        self.instrumented_evm.accounts.append(contract_address)
                        self.env.nr_of_transactions += 1
                        logger.debug("Contract deployed at %s", contract_address)

            if contract_address in self.instrumented_evm.accounts:
                self.instrumented_evm.accounts.remove(contract_address)

            self.env.overall_pcs, self.env.overall_jumpis = get_pcs_and_jumpis(self.instrumented_evm.get_code(to_canonical_address(contract_address)).hex())

        if self.args.abi:
            contract_address = self.args.contract

        self.instrumented_evm.create_snapshot()

        generator = Generator(interface=self.interface,
                              bytecode=self.deployement_bytecode,
                              accounts=self.instrumented_evm.accounts,
                              contract=contract_address)

        # Create initial population
        size = 2 * len(self.interface)
        population = Population(indv_template=Individual(generator=generator),
                                indv_generator=generator,
                                size=settings.POPULATION_SIZE if settings.POPULATION_SIZE else size).init()
        
        # Create genetic operators
        if self.args.data_dependency:
            selection = DataDependencyLinearRankingSelection(env=self.env)
            crossover = DataDependencyCrossover(pc=settings.PROBABILITY_CROSSOVER, env=self.env)
            mutation = Mutation(pm=settings.PROBABILITY_MUTATION)
        if self.args.diversity:
            selection = LinearRankingSelection()
            crossover = DiversityCrossover(pc=settings.PROBABILITY_CROSSOVER)
            mutation = DiversityMutation(pm=settings.PROBABILITY_MUTATION)
        else:
            selection = LinearRankingSelection()
            crossover = Crossover(pc=settings.PROBABILITY_CROSSOVER)
            mutation = Mutation(pm=settings.PROBABILITY_MUTATION)
            

        if self.args.algorithm == 'pso':
            # Create PSO engine
            engine = ParticleSwarmEngine(
                population=population,
                mapping=get_function_signature_mapping(self.env.abi),
                w=self.args.pso_w,
                c1=self.args.pso_c1,
                c2=self.args.pso_c2
            )
            logger.info("Using Particle Swarm Optimization (w=%.2f, c1=%.2f, c2=%.2f)", 
                       self.args.pso_w, self.args.pso_c1, self.args.pso_c2)
        elif self.args.algorithm == 'pyswarms':
            # Create PySwarms engine
            options = {
                'c1': self.args.pyswarms_c1,
                'c2': self.args.pyswarms_c2,
                'w': self.args.pyswarms_w,
                'k': self.args.pyswarms_k,
                'p': self.args.pyswarms_p,
            }
            engine = PySwarmsEngine(
                population=population,
                mapping=get_function_signature_mapping(self.env.abi),
                optimizer_type=self.args.pyswarms_optimizer,
                options=options
            )
            logger.info("Using PySwarms %s optimizer (w=%.2f, c1=%.2f, c2=%.2f, k=%d, p=%d)", 
                       self.args.pyswarms_optimizer, 
                       self.args.pyswarms_w, 
                       self.args.pyswarms_c1, 
                       self.args.pyswarms_c2,
                       self.args.pyswarms_k,
                       self.args.pyswarms_p)
        elif self.args.algorithm == 'collaborative':
            # Create Collaborative Diversity Engine
            engine = CollaborativeEngine(
                population=population,
                generator=generator,
                crossover=crossover,
                mutation=mutation,
                args=self.args
            )
            logger.info("Using Collaborative Diversity Engine (diversity_weight=%.2f, novelty_threshold=%.2f)", 
                       self.args.diversity_weight, self.args.novelty_threshold)
        else:
            # Create and run evolutionary fuzzing engine (GA)
            engine = EvolutionaryFuzzingEngine(
                population=population,
                selection=selection,
                crossover=crossover,
                mutation=mutation,
                mapping=get_function_signature_mapping(self.env.abi)
            )
            logger.info("Using Genetic Algorithm")

        engine.fitness_register(lambda x: fitness_function(x, self.env))
        engine.analysis.append(ExecutionTraceAnalyzer(self.env))
        self.env.execution_begin = time.time()
        self.env.population = population
        if self.args.algorithm == 'collaborative':
            engine.run(self.env, ng=settings.GENERATIONS)
        else:
            engine.run(ng=settings.GENERATIONS)

        if self.env.args.cfg:
            if self.env.args.source:
                self.env.cfg.save_control_flow_graph(os.path.splitext(self.env.args.source)[0]+'-'+self.contract_name, 'pdf')
            elif self.env.args.abi:
                self.env.cfg.save_control_flow_graph(os.path.join(os.path.dirname(self.env.args.abi), self.contract_name), 'pdf')

        self.instrumented_evm.reset()

def main():
    print_logo()
    args = launch_argument_parser()

    logger = initialize_logger("Main    ")

    # Check if contract has already been analyzed
    if args.results and os.path.exists(args.results):
        if args.results.endswith(".json"):
            resultJson = args.results
        else:
            resultJson = args.results + '/' + os.path.splitext(os.path.basename(args.contract))[0] + '.json'
        
        if os.path.exists(resultJson):
            os.remove(resultJson)
            logger.info("Contract "+str(args.source)+" has already been analyzed: "+str(resultJson))
            # sys.exit(0)

    # Initializing random
    if args.seed:
        seed = args.seed
        if not "PYTHONHASHSEED" in os.environ:
            logger.debug("Please set PYTHONHASHSEED to '1' for Python's hash function to behave deterministically.")
    else:
        seed = random.random()
    random.seed(seed)
    logger.title("Initializing seed to %s", seed)

    # Initialize EVM
    instrumented_evm = InstrumentedEVM(settings.RPC_HOST, settings.RPC_PORT)
    instrumented_evm.set_vm_by_name(settings.EVM_VERSION)

    # Create Z3 solver instance
    solver = Solver()
    solver.set("timeout", settings.SOLVER_TIMEOUT)

    # Parse blockchain state if provided
    blockchain_state = []
    if args.blockchain_state:
        if args.blockchain_state.endswith(".json"):
            with open(args.blockchain_state) as json_file:
                for line in json_file.readlines():
                    blockchain_state.append(json.loads(line))
        elif args.blockchain_state.isnumeric():
            settings.BLOCK_HEIGHT = int(args.blockchain_state)
            instrumented_evm.set_vm(settings.BLOCK_HEIGHT)
        else:
            logger.error("Unsupported input file: " + args.blockchain_state)
            sys.exit(-1)

    # Compile source code to get deployment bytecode, runtime bytecode and ABI
    if args.source:
        if args.source.endswith(".sol"):
            compiler_output = compile(args.solc_version, settings.EVM_VERSION, args.source)
            if not compiler_output:
                logger.error("No compiler output for: " + args.source)
                sys.exit(-1)
            for contract_name, contract in compiler_output['contracts'][args.source].items():
                if args.contract and contract_name != args.contract:
                    continue
                if contract['abi'] and contract['evm']['bytecode']['object'] and contract['evm']['deployedBytecode']['object']:
                    source_map = SourceMap(':'.join([args.source, contract_name]), compiler_output)
                    Fuzzer(contract_name, contract["abi"], contract['evm']['bytecode']['object'], contract['evm']['deployedBytecode']['object'], instrumented_evm, blockchain_state, solver, args, seed, source_map).run()
        else:
            logger.error("Unsupported input file: " + args.source)
            sys.exit(-1)

    if args.abi:
        with open(args.abi) as json_file:
            abi = json.load(json_file)
            runtime_bytecode = instrumented_evm.get_code(to_canonical_address(args.contract)).hex()
            Fuzzer(args.contract, abi, None, runtime_bytecode, instrumented_evm, blockchain_state, solver, args, seed).run()

def launch_argument_parser():
    parser = argparse.ArgumentParser()

    # Contract parameters
    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument("-s", "--source", type=str,
                        help="Solidity smart contract source code file (.sol).")
    group1.add_argument("-a", "--abi", type=str,
                        help="Smart contract ABI file (.json).")

    #group2 = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument("-c", "--contract", type=str,
                        help="Contract name to be fuzzed (if Solidity source code file provided) or blockchain contract address (if ABI file provided).")

    parser.add_argument("-b", "--blockchain-state", type=str,
                        help="Initialize fuzzer with a blockchain state by providing a JSON file (if Solidity source code file provided) or a block number (if ABI file provided).")

    # Compiler parameters
    parser.add_argument("--solc", help="Solidity compiler version (default '" + str(
        solcx.get_solc_version()) + "'). Installed compiler versions: " + str(solcx.get_installed_solc_versions()) + ".",
                        action="store", dest="solc_version", type=str)
    parser.add_argument("--evm", help="Ethereum VM (default '" + str(
        settings.EVM_VERSION) + "'). Available VM's: 'homestead', 'byzantium' or 'petersburg'.", action="store",
                        dest="evm_version", type=str)

    parser.add_argument("--algorithm", 
                        help="Optimization algorithm: 'ga' (Genetic Algorithm, default), 'pso' (custom PSO), 'pyswarms' (PySwarms library), or 'collaborative' (Collaborative Diversity).",
                        action="store", dest="algorithm", type=str, default="ga", 
                        choices=['ga', 'pso', 'pyswarms', 'collaborative'])
    
    parser.add_argument("--pso-w", 
                        help="PSO inertia weight (default 0.7).",
                        action="store", dest="pso_w", type=float, default=0.7)
    parser.add_argument("--pso-c1", 
                        help="PSO cognitive coefficient (default 1.5).",
                        action="store", dest="pso_c1", type=float, default=1.5)
    parser.add_argument("--pso-c2", 
                        help="PSO social coefficient (default 1.5).",
                        action="store", dest="pso_c2", type=float, default=1.5)
    
    parser.add_argument("--pyswarms-optimizer", 
                        help="PySwarms optimizer type: 'global' (GlobalBestPSO, default) or 'local' (LocalBestPSO).",
                        action="store", dest="pyswarms_optimizer", type=str, default="global",
                        choices=['global', 'local'])
    parser.add_argument("--pyswarms-w", 
                        help="PySwarms inertia weight (default 0.7).",
                        action="store", dest="pyswarms_w", type=float, default=0.7)
    parser.add_argument("--pyswarms-c1", 
                        help="PySwarms cognitive coefficient (default 1.5).",
                        action="store", dest="pyswarms_c1", type=float, default=1.5)
    parser.add_argument("--pyswarms-c2", 
                        help="PySwarms social coefficient (default 1.5).",
                        action="store", dest="pyswarms_c2", type=float, default=1.5)
    parser.add_argument("--pyswarms-k", 
                        help="PySwarms number of neighbors to be considered. Must be a positive integer less than :code:`n_particles` (default 5)",
                        action="store", dest="pyswarms_k", type=int, default=5)
    parser.add_argument("--pyswarms-p", 
                        help="PySwarms the Minkowski p-norm to use. 1 is the sum-of-absolute values (or L1 distance) while 2 is the Euclidean (or L2) distance (default 1)",
                        action="store", dest="pyswarms_p", type=int, default=1)
    parser.add_argument("--diversity-weight", 
                        help="Collaborative: Weight for diversity in fitness (default 0.3).",
                        action="store", dest="diversity_weight", type=float, default=0.3)
    parser.add_argument("--novelty-threshold", 
                        help="Collaborative: Threshold for novelty in archive (default 0.5).",
                        action="store", dest="novelty_threshold", type=float, default=0.5)

    # Evolutionary parameters
    group3 = parser.add_mutually_exclusive_group(required=False)
    group3.add_argument("-g", "--generations",
                        help="Number of generations (default " + str(settings.GENERATIONS) + ").", action="store",
                        dest="generations", type=int)
    group3.add_argument("-t", "--timeout",
                        help="Number of seconds for fuzzer to stop.", action="store",
                        dest="global_timeout", type=int)
    parser.add_argument("-n", "--population-size",
                        help="Size of the population.", action="store",
                        dest="population_size", type=int)
    parser.add_argument("-pc", "--probability-crossover",
                        help="Size of the population.", action="store",
                        dest="probability_crossover", type=float)
    parser.add_argument("-pm", "--probability-mutation",
                        help="Size of the population.", action="store",
                        dest="probability_mutation", type=float)

    # Miscellaneous parameters
    parser.add_argument("-r", "--results", type=str, help="Folder or JSON file where results should be stored.")
    parser.add_argument("--seed", type=float, help="Initialize the random number generator with a given seed.")
    parser.add_argument("--cfg", help="Build control-flow graph and highlight code coverage.", action="store_true")
    parser.add_argument("--rpc-host", help="Ethereum client RPC hostname.", action="store", dest="rpc_host", type=str)
    parser.add_argument("--rpc-port", help="Ethereum client RPC port.", action="store", dest="rpc_port", type=int)

    parser.add_argument("--data-dependency",
                        help="Disable/Enable data dependency analysis: 0 - Disable, 1 - Enable (default: 0)", action="store",
                        dest="data_dependency", type=int)
    parser.add_argument("--diversity",
                        help="Disable/Enable diversity analysis: 0 - Disable, 1 - Enable (default: 0)", action="store",
                        dest="diversity", type=int)
    parser.add_argument("--constraint-solving",
                        help="Disable/Enable constraint solving: 0 - Disable, 1 - Enable (default: 1)", action="store",
                        dest="constraint_solving", type=int)
    parser.add_argument("--environmental-instrumentation",
                        help="Disable/Enable environmental instrumentation: 0 - Disable, 1 - Enable (default: 1)", action="store",
                        dest="environmental_instrumentation", type=int)
    parser.add_argument("--max-individual-length",
                        help="Maximal length of an individual (default: " + str(settings.MAX_INDIVIDUAL_LENGTH) + ")", action="store",
                        dest="max_individual_length", type=int)
    parser.add_argument("--max-symbolic-execution",
                        help="Maximum number of symbolic execution calls before restting population (default: " + str(settings.MAX_SYMBOLIC_EXECUTION) + ")", action="store",
                        dest="max_symbolic_execution", type=int)

    version = "ConFuzzius - Version 0.0.2 - "
    version += "\"By three methods we may learn wisdom:\n"
    version += "First, by reflection, which is noblest;\n"
    version += "Second, by imitation, which is easiest;\n"
    version += "And third by experience, which is the bitterest.\"\n"
    parser.add_argument("-v", "--version", action="version", version=version)

    args = parser.parse_args()

    if not args.contract:
        args.contract = ""

    if args.source and args.contract.startswith("0x"):
        parser.error("--source requires --contract to be a name, not an address.")
    if args.source and args.blockchain_state and args.blockchain_state.isnumeric():
        parser.error("--source requires --blockchain-state to be a file, not a number.")

    if args.abi and not args.contract.startswith("0x"):
        parser.error("--abi requires --contract to be an address, not a name.")
    if args.abi and args.blockchain_state and not args.blockchain_state.isnumeric():
        parser.error("--abi requires --blockchain-state to be a number, not a file.")

    if args.evm_version:
        settings.EVM_VERSION = args.evm_version
    if not args.solc_version:
        args.solc_version = solcx.get_solc_version()
    if args.generations:
        settings.GENERATIONS = args.generations
    if args.global_timeout:
        settings.GLOBAL_TIMEOUT = args.global_timeout
    if args.population_size:
        settings.POPULATION_SIZE = args.population_size
    if args.probability_crossover:
        settings.PROBABILITY_CROSSOVER = args.probability_crossover
    if args.probability_mutation:
        settings.PROBABILITY_MUTATION = args.probability_mutation

    if args.data_dependency == None:
        args.data_dependency = 0
    if args.diversity == None:
        args.diversity = 0
    if args.constraint_solving == None:
        args.constraint_solving = 1
    if args.environmental_instrumentation == None:
        args.environmental_instrumentation = 1

    if args.environmental_instrumentation == 1:
        settings.ENVIRONMENTAL_INSTRUMENTATION = True
    elif args.environmental_instrumentation == 0:
        settings.ENVIRONMENTAL_INSTRUMENTATION = False

    if args.max_individual_length:
        settings.MAX_INDIVIDUAL_LENGTH = args.max_individual_length
    if args.max_symbolic_execution:
        settings.MAX_SYMBOLIC_EXECUTION = args.max_symbolic_execution

    if args.abi:
        settings.REMOTE_FUZZING = True

    if args.rpc_host:
        settings.RPC_HOST = args.rpc_host
    if args.rpc_port:
        settings.RPC_PORT = args.rpc_port

    return args

def print_logo():
    print("")
    print("     ______            ______                _           ")
    print("    / ____/___  ____  / ____/_  __________  (_)_  _______")
    print("   / /   / __ \/ __ \/ /_  / / / /_  /_  / / / / / / ___/")
    print("  / /___/ /_/ / / / / __/ / /_/ / / /_/ /_/ / /_/ (__  ) ")
    print("  \____/\____/_/ /_/_/    \__,_/ /___/___/_/\__,_/____/  ")
    print("")

if '__main__' == __name__:
    main()
