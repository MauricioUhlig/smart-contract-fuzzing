#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' PySwarms-based optimization engine for smart contract fuzzing '''

import math
import time
import logging
import random
import numpy as np

from functools import wraps

# Imports for profiling.
import cProfile
import pstats
import os

# PySwarms imports
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx

from utils import settings

from .components import Individual, Population
from .plugin_interfaces.analysis import OnTheFlyAnalysis

def do_profile(filename, sortby='tottime'):
    '''
    Constructor for function profiling decorator.
    '''
    def _do_profile(func):
        '''
        Function profiling decorator.
        '''
        @wraps(func)
        def profiled_func(*args, **kwargs):
            '''
            Decorated function.
            '''
            # Flag for doing profiling or not.
            DO_PROF = os.getenv('PROFILING')

            if DO_PROF:
                profile = cProfile.Profile()
                profile.enable()
                result = func(*args, **kwargs)
                profile.disable()
                ps = pstats.Stats(profile).sort_stats(sortby)
                ps.dump_stats(filename)
            else:
                result = func(*args, **kwargs)
            return result
        return profiled_func

    return _do_profile


class StatVar(object):
    def __init__(self, name):
        '''
        Descriptor for statistical variables which need to be memoized when
        engine is running.
        '''
        # Protected.
        self.name = '_{}'.format(name)

    def __get__(self, engine, cls):
        '''
        Getter.
        '''
        stat_var = getattr(engine, self.name)
        if stat_var is None:
            if 'min' in self.name and 'ori' in self.name:
                stat_var = engine.population.min(engine.ori_fitness)
            elif 'min' in self.name:
                stat_var = engine.population.min(engine.fitness)
            elif 'max' in self.name and 'ori' in self.name:
                stat_var = engine.population.max(engine.ori_fitness)
            elif 'max' in self.name:
                stat_var = engine.population.max(engine.fitness)
            elif 'mean' in self.name and 'ori' in self.name:
                stat_var = engine.population.mean(engine.ori_fitness)
            elif 'mean' in self.name:
                stat_var = engine.population.mean(engine.fitness)
            setattr(engine, self.name, stat_var)
        return stat_var

    def __set__(self, engine, value):
        '''
        Setter.
        '''
        setattr(engine, self.name, value)


class PySwarmsEngine(object):
    # Statistical attributes for population.
    fmax, fmin, fmean = StatVar('fmax'), StatVar('fmin'), StatVar('fmean')
    ori_fmax, ori_fmin, ori_fmean = (StatVar('ori_fmax'),
                                     StatVar('ori_fmin'),
                                     StatVar('ori_fmean'))

    def __init__(self, population, fitness=None, analysis=None, mapping=None,
                 optimizer_type='global',
                 options=None):
        '''
        Initialize PySwarms-based optimization engine.
        
        :param population: Population of individuals
        :param fitness: Fitness function
        :param analysis: Analysis plugins
        :param mapping: Function signature mapping
        :param optimizer_type: Type of PSO optimizer ('global', 'local')
        :param options: Dictionary of PSO hyperparameters
                       {'c1': cognitive, 'c2': social, 'w': inertia}
        '''
        # Set logger.
        logger_name = 'engine.{}'.format(self.__class__.__name__)
        self.logger = logging.getLogger(logger_name)

        # Attributes assignment.
        self.population = population
        self.fitness = fitness
        self.analysis = [] if analysis is None else [a() for a in analysis]
        self.mapping = mapping

        # PySwarms configuration
        self.optimizer_type = optimizer_type

        
        # Default PSO hyperparameters
        if options is None:
            self.options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7, 'k': 5, 'p': 1}
        else:
            self.options = options

        # Maxima and minima in population.
        self._fmax, self._fmin, self._fmean = None, None, None
        self._ori_fmax, self._ori_fmin, self._ori_fmean = None, None, None

        # Default fitness functions.
        self.ori_fitness = None if self.fitness is None else self.fitness

        # Store current generation number.
        self.current_generation = -1  # Starts from 0.

        # Encoding parameters for continuous space mapping
        self.dimension = None
        self.bounds = None
        
        # PySwarms optimizer instance
        self.optimizer = None

        # Check parameters validity.
        self._check_parameters()

        # generation counter
        self.ng = self._counter_generator()

    @do_profile(filename='pyswarms_engine_run.prof')
    def run(self, ng):
        '''
        Run the PySwarms optimization with specified number of iterations.
        
        :param ng: Number of generations/iterations
        '''
        try:
            execution_begin = time.time()

            if self.fitness is None:
                raise AttributeError('No fitness function in PySwarms engine')

            # Initialize encoding scheme
            self._initialize_encoding()

            # Setup analysis objects.
            for a in self.analysis:
                a.setup(ng=ng, engine=self)
                a.register_step(g=-1, population=self.population, engine=self)

            # Create PySwarms optimizer
            self._create_optimizer(ng)

            # Define the objective function for PySwarms (it expects minimization)
            def objective_function(positions):
                '''
                Objective function that PySwarms will optimize.
                Converts continuous positions to discrete individuals and evaluates fitness.
                
                :param positions: numpy array of shape (n_particles, dimensions)
                :param ng: number of current generation
                :return: numpy array of fitness values (negated for minimization)
                '''
                fitness_values = np.zeros(positions.shape[0])
                
                individuals = []
                for position in positions:
                    # Decode position to individual
                    individual = self._decode_position(position)
                    
                    individuals.append(individual)

                # May be wrong
                self.population.init(individuals)
            
                current_count = next(self.ng)
                # Final analysis
                for a in self.analysis:
                    a.register_step(g=current_count, population=self.population, engine=self)

                for i, individual in enumerate(individuals):
                    # Evaluate fitness (negate because PySwarms minimizes)
                    fitness_val = self.fitness(individual)
                    fitness_values[i] = -fitness_val  # Negate for minimization
                
                return fitness_values

            # Run optimization
            self.logger.info("Starting PySwarms optimization with %d iterations", ng)
            cost, pos = self.optimizer.optimize(objective_function, iters=ng)

            # Update population with final best positions
            self._update_population_from_optimizer(pos)

            self.logger.info("PySwarms optimization completed. Best cost: %f", cost)

        except Exception as e:
            # Log exception info.
            msg = '{} exception is catched'.format(type(e).__name__)
            self.logger.exception(msg)
            raise e
        finally:
            # Perform the analysis post processing.
            for a in self.analysis:
                a.finalize(population=self.population, engine=self)

    def _initialize_encoding(self):
        '''
        Initialize the encoding scheme to map between continuous PSO positions
        and discrete chromosome representations.
        '''
        # Analyze chromosome structure to determine dimensionality
        sample_individual = self.population.individuals[0]
        
        # Calculate dimensions needed for encoding
        # Each gene (transaction) needs encoding for:
        # - function selector (categorical)
        # - arguments (varies by function)
        # - amount, gaslimit, account, etc.
        
        # For simplicity, we'll use a fixed-size encoding
        max_genes = settings.MAX_INDIVIDUAL_LENGTH
        dims_per_gene = 10  # account, amount, gaslimit, function, + 6 for arguments
        
        self.dimension = max_genes * dims_per_gene
        
        # Set bounds for continuous space
        # Most values will be normalized to [0, 1] and then mapped to discrete values
        self.bounds = (np.zeros(self.dimension), np.ones(self.dimension))
        
        self.logger.info("Initialized encoding with %d dimensions", self.dimension)

    def _create_optimizer(self, iters):
        '''
        Create the PySwarms optimizer instance based on configuration.
        
        :param iters: Number of iterations
        '''
        n_particles = self.population.size
        dimensions = self.dimension
        
        if self.optimizer_type == 'global':
            # Global best PSO
            self.optimizer = ps.single.GlobalBestPSO(
                n_particles=n_particles,
                dimensions=dimensions,
                options=self.options,
                bounds=self.bounds
            )
            self.logger.info("Created GlobalBestPSO optimizer")
            
        elif self.optimizer_type == 'local':
            self.optimizer = ps.single.LocalBestPSO(
                n_particles=n_particles,
                dimensions=dimensions,
                options=self.options,
                bounds=self.bounds
            )
            self.logger.info("Created LocalBestPSO optimizer")
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")

    def _encode_individual(self, individual):
        '''
        Encode an individual's chromosome into a continuous position vector.
        
        :param individual: Individual to encode
        :return: numpy array representing position in continuous space
        '''
        position = np.zeros(self.dimension)
        
        chromosome = individual.chromosome
        dims_per_gene = self.dimension // settings.MAX_INDIVIDUAL_LENGTH
        
        for i, gene in enumerate(chromosome):
            if i >= settings.MAX_INDIVIDUAL_LENGTH:
                break
                
            offset = i * dims_per_gene
            
            # Encode account (normalized)
            if 'account' in gene:
                account_idx = hash(gene['account']) % 100
                position[offset] = account_idx / 100.0
            
            # Encode amount (normalized)
            if 'amount' in gene:
                position[offset + 1] = min(gene['amount'] / 10000.0, 1.0)
            
            # Encode gaslimit (normalized)
            if 'gaslimit' in gene:
                position[offset + 2] = min(gene['gaslimit'] / 10000000.0, 1.0)
            
            # Encode function (categorical, normalized)
            if 'arguments' in gene and len(gene['arguments']) > 0:
                func_hash = hash(str(gene['arguments'][0])) % 100
                position[offset + 3] = func_hash / 100.0
            
            # Encode first few arguments (simplified)
            if 'arguments' in gene:
                for j, arg in enumerate(gene['arguments'][1:6]):  # Up to 5 args
                    if isinstance(arg, int):
                        position[offset + 4 + j] = min(abs(arg) / 1000.0, 1.0)
                    elif isinstance(arg, (list, bytearray)):
                        position[offset + 4 + j] = len(arg) / 100.0
        
        return position

    def _decode_position(self, position):
        '''
        Decode a continuous position vector into an individual's chromosome.
        
        :param position: numpy array representing position in continuous space
        :return: Individual object
        '''
        individual = Individual(self.population.individuals[0].generator)
        chromosome = []
        
        dims_per_gene = self.dimension // settings.MAX_INDIVIDUAL_LENGTH
        
        # Determine number of genes based on position values
        num_genes = random.randint(1, settings.MAX_INDIVIDUAL_LENGTH)
        
        for i in range(num_genes):
            offset = i * dims_per_gene
            
            # Create a gene by sampling from the generator and modifying based on position
            sample_gene = individual.generator.generate_random_individual()[0]
            gene = sample_gene.copy()
            
            # Decode amount
            gene['amount'] = int(position[offset + 1] * 10000)
            
            # Decode gaslimit
            gene['gaslimit'] = int(position[offset + 2] * 10000000)
            gene['gaslimit'] = max(100000, min(gene['gaslimit'], 8000000))
            
            # Keep other fields from sample (account, contract, arguments)
            # Could be enhanced to decode these as well
            
            chromosome.append(gene)
        
        individual.init(chromosome=chromosome)
        return individual

    def _update_population_from_optimizer(self, best):
        '''
        Update the population with individuals decoded from optimizer's final positions.
        '''
        
        # Decode each position to an individual
        new_individuals = []
    
        individual = self._decode_position(best)
        new_individuals.append(individual)
        
        self.population.individuals = new_individuals
        self.logger.info("Updated population from optimizer's final positions")

    def _update_statvars(self):
        '''
        Private helper function to update statistic variables in engine.
        '''
        # Wrt original fitness.
        self.ori_fmax = self.population.max(self.ori_fitness)
        self.ori_fmin = self.population.min(self.ori_fitness)
        self.ori_fmean = self.population.mean(self.ori_fitness)

        # Wrt decorated fitness.
        self.fmax = self.population.max(self.fitness)
        self.fmin = self.population.min(self.fitness)
        self.fmean = self.population.mean(self.fitness)

    def _check_parameters(self):
        '''
        Helper function to check parameters of engine.
        '''
        if not isinstance(self.population, Population):
            raise TypeError('population must be a Population object')

        for ap in self.analysis:
            if not isinstance(ap, OnTheFlyAnalysis):
                msg = '{} is not subclass of OnTheFlyAnalysis'.format(ap.__name__)
                raise TypeError(msg)

    # Decorators.

    def fitness_register(self, fn):
        '''
        A decorator for fitness function register.
        '''
        @wraps(fn)
        def _fn_with_fitness_check(indv):
            '''
            A wrapper function for fitness function with fitness value check.
            '''
            # Check indv type.
            if not isinstance(indv, Individual):
                raise TypeError('indv\'s class must be Individual or a subclass of Individual')

            # Check fitness.
            fitness = fn(indv)
            is_invalid = (type(fitness) is not float) or (math.isnan(fitness))
            if is_invalid:
                msg = 'Fitness value(value: {}, type: {}) is invalid'
                msg = msg.format(fitness, type(fitness))
                raise ValueError(msg)
            return fitness

        self.fitness = _fn_with_fitness_check
        if self.ori_fitness is None:
            self.ori_fitness = _fn_with_fitness_check

    def analysis_register(self, analysis_cls):
        '''
        A decorator for analysis register.
        '''
        if not issubclass(analysis_cls, OnTheFlyAnalysis):
            raise TypeError('analysis class must be subclass of OnTheFlyAnalysis')

        # Add analysis instance to engine.
        analysis = analysis_cls()
        self.analysis.append(analysis)

    def minimize(self, fn):
        '''
        A decorator for minimizing the fitness function.
        '''
        @wraps(fn)
        def _minimize(indv):
            return -fn(indv)
        return _minimize

    def _counter_generator(self):
        count = -1
        while True:
            count += 1
            yield count

