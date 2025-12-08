#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pprint

from utils import settings

import random
import math
import time
from utils.utils import *
from engine.fitness import fitness_function, compute_branch_coverage_fitness, compute_data_dependency_fitness
from functools import wraps

from .components import Individual, Population

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

class CollaborativeEngine:
    """
    Collaborative Evolutionary Engine with Diversity-Based Optimization
    
    Features:
    - Population-based fitness function (considers group performance)
    - Diversity-based selection (maintains population diversity)
    - Enhanced crossover for diversity (promotes diverse offspring)
    """
    
    def __init__(self, population, generator, selection, crossover, mutation, args, fitness=None, analysis=None):
        self.logger = initialize_logger("CollaborativeEngine")
        self.population = population
        self.generator = generator
        self.args = args
        self.analysis = [] if analysis is None else [a() for a in analysis]
        self.fitness = fitness
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        # Default fitness functions.
        self.ori_fitness = None if self.fitness is None else self.fitness
        
        self.data_dependency_weight = args.data_dependency_weight if hasattr(args, 'data_dependency_weight') else 1.0
        self.branch_weight = args.branch_weight if hasattr(args, 'branch_weight') else 1.0
        self.unique_branch_weight = args.unique_branch_weight if hasattr(args, 'unique_branch_weight') else 1.0
        
        # Archive of best diverse solutions
        self.archive = []
        self.archive_size = 20
        
        # Track population diversity over time
        self.diversity_history = []

        # Store current generation number.
        self.current_generation = -1  # Starts from 0.
        
    def run(self, env, ng):
        """Main evolution loop with collaborative fitness and diversity maintenance"""
        self.logger.info("Starting Collaborative Evolutionary Engine")
        self.logger.info(f"Population size: {self.population.size}, Generations: {ng}")
        self.logger.info("Weights:")
        self.logger.info(f"-> Data Dependency: {self.data_dependency_weight}")
        self.logger.info(f"-> Branch coverage: {self.branch_weight}")
        self.logger.info(f"-> Unique Branch coverage: {self.unique_branch_weight}")

        self.env = env

        try: 
            execution_begin = time.time()

            for a in self.analysis:
                a.setup(ng=ng, engine=self)
                a.register_step(g=-1, population=self.population, engine=self)
            
             # Enter evolution iteration.
            g = 0
            while g < ng or settings.GLOBAL_TIMEOUT:
                if settings.GLOBAL_TIMEOUT and time.time() - execution_begin >= settings.GLOBAL_TIMEOUT:
                    break
                gen_start = time.time()
                
                self.current_generation = g

                # Execute all individuals and collect results
                self._execute_population(g)
                
                # Calculate collaborative fitness (individual + population-based)
                fitness_scores = self._calculate_collaborative_fitness(env)
                
                # Create new generation using diversity-aware operators
                new_population = []
                
                while len(new_population) < self.population.size:
                    # Diversity-based selection
                    parents = self.selection.select(self.population, fitness=self.fitness)
                    
                    # Crossover.
                    children = self.crossover.cross(*parents)
                    # Mutation.
                    children = [self.mutation.mutate(child, self) for child in children]
                    # Collect children.
                    new_population.extend(children)
                    
                
                # Replace population
                self.population.individuals = new_population[:self.population.size]
                
                
                
                # Logging
                best_fitness = max(fitness_scores.values())
                avg_fitness = sum(fitness_scores.values()) / len(fitness_scores)
                
                self.logger.info(f"Generation {g + 1}/{ng} | "
                            f"Best Fitness: {best_fitness:.2f} | "
                            f"Avg Fitness: {avg_fitness:.2f}")
                
                g += 1
        except Exception as e:
            # Log exception info.
            msg = '{} exception is catched'.format(type(e).__name__)
            self.logger.exception(msg)
            raise e
        finally:
            # Perform the analysis post processing.
            for a in self.analysis:
                a.finalize(population=self.population, engine=self)
    
    def _execute_population(self, generation):
        """Execute all individuals in the population"""
        if hasattr(self, 'analysis') and self.analysis:
            for analyzer in self.analysis:
                if hasattr(analyzer, 'env'):
                    # Execute the transaction to populate coverage
                    analyzer.register_step(g=generation, population=self.population, engine=self)
    
    def _calculate_collaborative_fitness(self, env):
        """
        Calculate fitness considering data dependencies, branch coverage and unique branch coverage
        """
        fitness_scores = {}
        branch_coverage = {}
        data_dependencie = {}
        
        unique_contributions = self._calculate_collaborative_unique_contribution(env)
        
        for individual in self.population.individuals:
            branch_coverage[individual.hash] = compute_branch_coverage_fitness(env.individual_branches[individual.hash], env.code_coverage)
            data_dependencie[individual.hash] = compute_data_dependency_fitness(individual, env.data_dependencies)
        
        
        for individual in self.population.individuals:
            individual_hash = individual.hash
            fitness_scores[individual_hash] = data_dependencie[individual_hash] * self.data_dependency_weight - (branch_coverage[individual_hash] * self.branch_weight / (unique_contributions[individual_hash] * self.unique_branch_weight +1))
        
        return fitness_scores
        
    def _calculate_collaborative_unique_contribution(self, env):
        """
        Calculation of unique contributions
        """
        # Build coverage data
        branch_coverage = {}  # (jumpi, dest) -> set of individual hashes
        individual_coverage = {}  # individual hash -> set of (jumpi, dest)
        valid_individuals = set()
        
        # Single pass to build all data structures
        for individual in self.population.individuals:
            individual_hash = individual.hash
            if individual_hash not in env.individual_branches:
                continue
                
            valid_individuals.add(individual_hash)
            individual_branches = set()
            
            for jumpi, destinations in env.individual_branches[individual_hash].items():
                for dest, covered in destinations.items():
                    if covered:
                        branch = (jumpi, dest)
                        individual_branches.add(branch)
                        
                        if branch not in branch_coverage:
                            branch_coverage[branch] = set()
                        branch_coverage[branch].add(individual_hash)
            
            individual_coverage[individual_hash] = individual_branches
        
        unique_contributions = {}
        
        for individual_hash, branches in individual_coverage.items():
            unique_count = 0
            
            for branch in branches:
                covering_individuals = branch_coverage[branch]
                
                # Unique contribution
                if len(covering_individuals) == 1:
                    unique_count += 1
            
            unique_contributions[individual_hash] = float(unique_count)
        
        # Handle individuals without coverage data
        for individual in self.population.individuals:
            if individual.hash not in valid_individuals:
                unique_contributions[individual.hash] = 0.0
        
        return unique_contributions

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

    def _update_statvars(self):
        '''
        Private helper function to update statistic variables in PSO engine, like
        maximum, minimum and mean values.
        '''
        # Wrt original fitness.
        self.ori_fmax = self.population.max(self.ori_fitness)
        self.ori_fmin = self.population.min(self.ori_fitness)
        self.ori_fmean = self.population.mean(self.ori_fitness)

        # Wrt decorated fitness.
        self.fmax = self.population.max(self.fitness)
        self.fmin = self.population.min(self.fitness)
        self.fmean = self.population.mean(self.fitness)