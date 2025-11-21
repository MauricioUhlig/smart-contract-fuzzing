#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pprint

from utils import settings

import random
import math
import time
from utils.utils import *
from engine.fitness import fitness_function
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
    
    def __init__(self, population, generator, crossover, mutation, args, fitness=None, analysis=None):
        self.logger = initialize_logger("CollaborativeEngine")
        self.population = population
        self.generator = generator
        self.args = args
        self.analysis = [] if analysis is None else [a() for a in analysis]
        self.fitness = fitness
        self.crossover = crossover
        self.mutation = mutation
        # Default fitness functions.
        self.ori_fitness = None if self.fitness is None else self.fitness

        # Diversity parameters
        self.diversity_weight = args.diversity_weight if hasattr(args, 'diversity_weight') else 0.3
        self.novelty_threshold = args.novelty_threshold if hasattr(args, 'novelty_threshold') else 0.5
        
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
        self.logger.info(f"Diversity weight: {self.diversity_weight}, Novelty threshold: {self.novelty_threshold}")

        self.env = env

        try: 
            execution_begin = time.time()

            if self.fitness is None:
                raise AttributeError('No fitness function in Collaborative engine')

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
                
                # Calculate diversity metrics
                diversity_score = self._calculate_population_diversity()
                self.diversity_history.append(diversity_score)
                
                # Update archive with novel solutions
                self._update_archive(fitness_scores)
                
                # Create new generation using diversity-aware operators
                new_population = []
                
                while len(new_population) < self.population.size:
                    # Diversity-based selection
                    parents = self._diversity_based_selection(fitness_scores)
                    

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
                            f"Avg Fitness: {avg_fitness:.2f} | "
                            f"Diversity: {diversity_score:.3f}")
                
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
        Calculate fitness considering both individual and population-level performance
        
        Collaborative fitness = Individual fitness + Group contribution bonus
        """
        fitness_scores = {}
        
        # Calculate individual fitness
        all_fitness = self.population.all_fits(self.fitness)                    
        
        for i in range(self.population.size):
            individual = self.population.individuals[i]
            individual_fitness = all_fitness[i]
            fitness_scores[individual.hash] = individual_fitness
        
        # Add group contribution bonus
        for individual in self.population.individuals:
            # Bonus for covering unique branches
            unique_contribution = self._calculate_unique_contribution(individual, env)
            
            # Bonus for complementing other individuals
            complementarity = self._calculate_complementarity(individual, env)
            
            # Combined collaborative fitness
            group_bonus = unique_contribution + complementarity
            fitness_scores[individual.hash] += group_bonus * self.diversity_weight
        
        return fitness_scores
    
    def _calculate_unique_contribution(self, individual, env):
        """Calculate how many unique branches this individual covers"""
        if individual.hash not in env.individual_branches:
            print("not in!")
            return 0.0
        
        individual_branches = set()
        for jumpi in env.individual_branches[individual.hash]:
            for dest in env.individual_branches[individual.hash][jumpi]:
                if env.individual_branches[individual.hash][jumpi][dest]:
                    individual_branches.add((jumpi, dest))
        
        # Count branches covered by this individual but not by others
        unique_count = 0
        for branch in individual_branches:
            covered_by_others = False
            for other in self.population.individuals:
                if other.hash == individual.hash:
                    continue
                if other.hash in env.individual_branches:
                    jumpi, dest = branch
                    if jumpi in env.individual_branches[other.hash]:
                        if dest in env.individual_branches[other.hash][jumpi]:
                            if env.individual_branches[other.hash][jumpi][dest]:
                                covered_by_others = True
                                break
            if not covered_by_others:
                unique_count += 1
        
        return float(unique_count)
    
    def _calculate_complementarity(self, individual, env):
        """Calculate how well this individual complements the population"""
        if individual.hash not in env.individual_branches:
            return 0.0
        
        # Reward individuals that cover branches not well-covered by population
        complementarity_score = 0.0
        
        for jumpi in env.individual_branches[individual.hash]:
            for dest in env.individual_branches[individual.hash][jumpi]:
                if env.individual_branches[individual.hash][jumpi][dest]:
                    # Count how many other individuals cover this branch
                    coverage_count = 0
                    for other in self.population.individuals:
                        if other.hash != individual.hash and other.hash in env.individual_branches:
                            if jumpi in env.individual_branches[other.hash]:
                                if dest in env.individual_branches[other.hash][jumpi]:
                                    if env.individual_branches[other.hash][jumpi][dest]:
                                        coverage_count += 1
                    
                    # Higher score for branches covered by fewer individuals
                    if coverage_count < self.population.size * 0.3:
                        complementarity_score += 1.0
        
        return complementarity_score
    
    def _calculate_population_diversity(self):
        """Calculate overall population diversity using behavioral distance"""
        if len(self.population.individuals) < 2:
            return 0.0
        
        total_distance = 0.0
        comparisons = 0
        
        for i in range(len(self.population.individuals)):
            for j in range(i + 1, len(self.population.individuals)):
                distance = self._behavioral_distance(
                    self.population.individuals[i],
                    self.population.individuals[j]
                )
                total_distance += distance
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0
    
    def _behavioral_distance(self, ind1, ind2):
        """
        Calculate behavioral distance between two individuals
        Based on differences in chromosome structure and function calls
        """
        # Chromosome length difference
        len_diff = abs(len(ind1.chromosome) - len(ind2.chromosome))
        
        # Function call differences
        functions1 = set(gene["arguments"][0] for gene in ind1.chromosome)
        functions2 = set(gene["arguments"][0] for gene in ind2.chromosome)
        function_diff = len(functions1.symmetric_difference(functions2))
        
        # Account differences
        accounts1 = set(gene["account"] for gene in ind1.chromosome)
        accounts2 = set(gene["account"] for gene in ind2.chromosome)
        account_diff = len(accounts1.symmetric_difference(accounts2))
        
        # Normalize and combine
        max_len = max(len(ind1.chromosome), len(ind2.chromosome), 1)
        distance = (len_diff + function_diff + account_diff) / (max_len * 3)
        
        return distance
    
    def _update_archive(self, fitness_scores):
        """Update archive with novel high-performing solutions"""
        for individual in self.population.individuals:
            if fitness_scores[individual.hash] > 0:
                # Check if individual is novel compared to archive
                is_novel = True
                if len(self.archive) > 0:
                    min_distance = min(
                        self._behavioral_distance(individual, archived)
                        for archived in self.archive
                    )
                    is_novel = min_distance > self.novelty_threshold
                
                if is_novel:
                    self.archive.append(individual.clone())
                    
                    # Maintain archive size
                    if len(self.archive) > self.archive_size:
                        # Remove least fit individual from archive
                        archive_fitness = [
                            fitness_scores.get(ind.hash, 0) for ind in self.archive
                        ]
                        min_idx = archive_fitness.index(min(archive_fitness))
                        self.archive.pop(min_idx)
    
    def _diversity_based_selection(self, fitness_scores):
        """
        Select parents balancing fitness and diversity
        Uses tournament selection with diversity consideration
        """
        tournament_size = 3
        
        def select_one():
            # Random tournament
            candidates = random.sample(self.population.individuals, tournament_size)
            
            # Score each candidate: fitness + diversity bonus
            best_candidate = None
            best_score = -float('inf')
            
            for candidate in candidates:
                # Base fitness
                score = fitness_scores[candidate.hash]
                
                # Diversity bonus: average distance to population
                diversity_bonus = 0.0
                for other in self.population.individuals:
                    if other.hash != candidate.hash:
                        diversity_bonus += self._behavioral_distance(candidate, other)
                diversity_bonus /= (len(self.population.individuals) - 1)
                
                # Combined score
                total_score = score + diversity_bonus * self.diversity_weight * 10
                
                if total_score > best_score:
                    best_score = total_score
                    best_candidate = candidate
            
            return best_candidate
        
        parent1 = select_one()
        parent2 = select_one()
        
        # Ensure parents are different
        attempts = 0
        while parent2.hash == parent1.hash and attempts < 10:
            parent2 = select_one()
            attempts += 1
        
        return parent1, parent2

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