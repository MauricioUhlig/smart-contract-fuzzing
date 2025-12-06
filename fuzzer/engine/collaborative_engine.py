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
                # self._update_archive(fitness_scores)
                
                # Create new generation using diversity-aware operators
                new_population = []
                
                for individual in self.population.individuals:
                    new_population.append(self.mutation.mutate(individual, self))
                # while len(new_population) < self.population.size:
                #     # Diversity-based selection
                #     parents = self._diversity_based_selection(fitness_scores)
                    

                #     # Crossover.
                #     children = self.crossover.cross(*parents)
                #     # Mutation.
                #     children = [self.mutation.mutate(child, self) for child in children]
                #     # Collect children.
                #     new_population.extend(children)
                    
                
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
        # Get base fitness scores
        all_fitness = self.population.all_fits(self.fitness)
        self.current_fitness_base_scores = {}
        
        # Build fitness mapping
        for i, individual in enumerate(self.population.individuals):
            self.current_fitness_base_scores[individual.hash] = all_fitness[i]
        
        # Calculate bonuses in single pass
        unique_contributions, complementarities = self._calculate_collaborative_bonuses_single_pass(env)
        
        # Apply bonuses
        fitness_scores = {}
        for individual in self.population.individuals:
            individual_hash = individual.hash
            group_bonus = unique_contributions[individual_hash] + complementarities[individual_hash]
            fitness_scores[individual_hash] = self.current_fitness_base_scores[individual.hash] + group_bonus * self.diversity_weight
        
        return fitness_scores
        
    def _calculate_collaborative_bonuses_single_pass(self, env):
        """
        Single-pass calculation of both unique contributions and complementarity
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
        
        # Calculate bonuses in single pass
        population_size = len(valid_individuals)
        coverage_threshold = population_size * 0.3
        unique_contributions = {}
        complementarities = {}
        
        for individual_hash, branches in individual_coverage.items():
            unique_count = 0
            complementarity_score = 0
            
            for branch in branches:
                covering_individuals = branch_coverage[branch]
                
                # Unique contribution
                if len(covering_individuals) == 1:
                    unique_count += 1
                
                # Complementarity
                if len(covering_individuals) < coverage_threshold:
                    complementarity_score += 1
            
            unique_contributions[individual_hash] = float(unique_count)
            complementarities[individual_hash] = float(complementarity_score)
        
        # Handle individuals without coverage data
        for individual in self.population.individuals:
            if individual.hash not in valid_individuals:
                unique_contributions[individual.hash] = 0.0
                complementarities[individual.hash] = 0.0
        
        return unique_contributions, complementarities

    def _calculate_unique_contribution(self, individual, env):
        """Calculate how many unique branches this individual covers"""
        individual_hash = individual.hash
        
        # Early return if individual has no branch data
        if individual_hash not in env.individual_branches:
            return 0.0
        
        # Get population branch coverage (excludes current individual)
        population_coverage = self._get_population_branch_coverage(env, exclude_individual_hash=individual_hash)
        
        # Extract individual's covered branches efficiently
        individual_branches = set()
        individual_data = env.individual_branches[individual_hash]
        
        for jumpi, destinations in individual_data.items():
            for dest, covered in destinations.items():
                if covered:
                    individual_branches.add((jumpi, dest))
        
        # Count branches that are ONLY covered by this individual
        unique_count = 0
        for branch in individual_branches:
            if branch not in population_coverage:
                unique_count += 1
        
        return float(unique_count)

    def _get_population_branch_coverage(self, env, exclude_individual_hash=None):
        """
        Precompute all branches covered by the population (excluding specified individual)
        Returns: set of (jumpi, dest) branches covered by population
        """
        # Cache key for memoization
        cache_key = f"population_coverage_exclude_{exclude_individual_hash}"
        
        if hasattr(self, '_coverage_cache') and cache_key in self._coverage_cache:
            return self._coverage_cache[cache_key]
        
        population_branches = set()
        
        for other in self.population.individuals:
            other_hash = other.hash
            
            # Skip excluded individual
            if other_hash == exclude_individual_hash:
                continue
                
            if other_hash in env.individual_branches:
                other_data = env.individual_branches[other_hash]
                
                for jumpi, destinations in other_data.items():
                    for dest, covered in destinations.items():
                        if covered:
                            population_branches.add((jumpi, dest))
        
        # Initialize cache if needed
        if not hasattr(self, '_coverage_cache'):
            self._coverage_cache = {}
        
        self._coverage_cache[cache_key] = population_branches
        return population_branches    
    
    def _calculate_complementarity(self, individual, env):
        """Calculate how well this individual complements the population"""
        individual_hash = individual.hash
        
        # Early return if individual has no branch data
        if individual_hash not in env.individual_branches:
            return 0.0
        
        # Precompute population branch coverage
        branch_coverage = self._get_population_branch_coverage(env, individual_hash)
        coverage_threshold = self.population.size * 0.3
        
        complementarity_score = 0.0
        individual_branches = env.individual_branches[individual_hash]
        
        # Single pass through individual's branches
        for jumpi, destinations in individual_branches.items():
            for dest, covered in destinations.items():
                if covered:
                    branch = (jumpi, dest)
                    coverage_count = branch_coverage.get(branch, 0)
                    
                    # Reward branches with below-average coverage
                    if coverage_count < coverage_threshold:
                        complementarity_score += 1.0
        
        return complementarity_score

    def _get_population_branch_coverage(self, env, exclude_individual_hash=None):
        """
        Precompute how many individuals cover each branch in the population
        Returns: dict{(jumpi, dest): coverage_count}
        """
        # Use caching to avoid recomputation
        if hasattr(self, '_branch_coverage_cache'):
            return self._branch_coverage_cache
        
        branch_coverage = {}
        
        # Build branch coverage counts
        for individual in self.population.individuals:
            individual_hash = individual.hash
            
            # Skip excluded individual if provided
            if individual_hash == exclude_individual_hash:
                continue
                
            if individual_hash in env.individual_branches:
                individual_branches = env.individual_branches[individual_hash]
                
                for jumpi, destinations in individual_branches.items():
                    for dest, covered in destinations.items():
                        if covered:
                            branch = (jumpi, dest)
                            branch_coverage[branch] = branch_coverage.get(branch, 0) + 1
        
        # Cache for potential reuse
        self._branch_coverage_cache = branch_coverage
        return branch_coverage
    
    def _behavioral_distance(self, features1, features2):
        """
        Behavioral distance calculation using precomputed features
        """
        # Chromosome length difference (normalized)
        len_diff = abs(features1['length'] - features2['length'])
        max_len = max(features1['length'], features2['length'], 1)
        
        # Function call differences using set operations
        function_diff = len(features1['functions'] ^ features2['functions'])  # Symmetric difference
        
        # Account differences
        account_diff = len(features1['accounts'] ^ features2['accounts'])
        
        # Combine distances (normalized by max length)
        distance = (len_diff + function_diff + account_diff) / (max_len * 3)
        
        return distance
    
    def _extract_behavioral_features(self, individual):
        """
        Extract and cache behavioral features for efficient distance calculation
        """
        # Use caching to avoid recomputation
        if hasattr(individual, '_cached_features'):
            return individual._cached_features
        
        chromosome = individual.chromosome
        features = {
            'length': len(chromosome),
            'functions': set(),
            'accounts': set(),
            'function_count': {},
            'account_count': {}
        }
        
        # Single pass through chromosome
        for gene in chromosome:
            # Extract function (first argument)
            if gene["arguments"]:
                function = gene["arguments"][0]
                features['functions'].add(function)
                features['function_count'][function] = features['function_count'].get(function, 0) + 1
            
            # Extract account
            account = gene["account"]
            features['accounts'].add(account)
            features['account_count'][account] = features['account_count'].get(account, 0) + 1
        
        # Cache for future use
        individual._cached_features = features
        return features

    def _calculate_population_diversity_optimized(self):
        """
        Fully optimized population diversity calculation
        Automatically chooses the best strategy based on population size
        """
        population_size = len(self.population.individuals)
        
        if population_size < 2:
            return 0.0
        elif population_size <= 20:
            return self._calculate_population_diversity_batch()
        else:
            return self._calculate_diversity_sampled(sample_size=50)

    def _calculate_population_diversity_batch(self):
        """
        Batch-optimized diversity calculation for large populations
        Uses sampling for very large populations
        """
        population_size = len(self.population.individuals)
        
        # Precompute all features
        features_list = [self._extract_behavioral_features(ind) for ind in self.population.individuals]
        
        # Calculate using efficient pairwise computation
        total_distance = 0.0
        comparisons = 0
        
        for i in range(population_size):
            features_i = features_list[i]
            for j in range(i + 1, population_size):
                distance = self._behavioral_distance(features_i, features_list[j])
                total_distance += distance
                comparisons += 1
        
        return total_distance / comparisons

    def _calculate_diversity_sampled(self, sample_size=30):
        """
        Use random sampling to estimate diversity for large populations
        """
        population = self.population.individuals
        if len(population) <= sample_size:
            return self._calculate_population_diversity_batch()
        
        # Randomly sample individuals
        sampled_individuals = random.sample(population, sample_size)
        
        # Precompute features for sampled individuals
        sampled_features = [self._extract_behavioral_features(ind) for ind in sampled_individuals]
        
        # Calculate diversity within sample
        total_distance = 0.0
        comparisons = 0
        sample_size = len(sampled_individuals)
        
        for i in range(sample_size):
            features_i = sampled_features[i]
            for j in range(i + 1, sample_size):
                distance = self._behavioral_distance(features_i, sampled_features[j])
                total_distance += distance
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0

    # Update the main method to use the optimized version
    def _calculate_population_diversity(self):
        """Public interface - uses optimized implementation"""
        return self._calculate_population_diversity_optimized()

    def _diversity_based_selection_cached(self, fitness_scores):
        """
        Version with comprehensive caching for repeated calls
        """
        tournament_size = 3
        population = self.population.individuals
        
        # Check if we can reuse cached features
        if not hasattr(self, '_selection_features_cache'):
            self._selection_features_cache = {}
            self._last_population_hash = None
        
        # Generate population hash for cache validation
        current_pop_hash = hash(tuple(ind.hash for ind in population))
        
        # Rebuild cache if population changed
        if current_pop_hash != self._last_population_hash:
            self._selection_features_cache.clear()
            for individual in population:
                self._selection_features_cache[individual.hash] = self._extract_behavioral_features(individual)
            self._last_population_hash = current_pop_hash
        
        features_cache = self._selection_features_cache
        
        def select_one_cached():
            candidates = random.sample(population, tournament_size)
            
            best_candidate = None
            best_score = -float('inf')
            
            for candidate in candidates:
                candidate_hash = candidate.hash
                candidate_feat = features_cache[candidate_hash]
                
                # Efficient diversity calculation
                total_distance = 0.0
                count = 0
                
                # Use a sample for large populations
                if len(population) > 20:
                    sample_size = min(15, len(population) - 1)
                    sample_others = random.sample([ind for ind in population if ind.hash != candidate_hash], sample_size)
                    for other in sample_others:
                        total_distance += self._behavioral_distance(candidate_feat, features_cache[other.hash])
                        count += 1
                else:
                    for other in population:
                        if other.hash != candidate_hash:
                            total_distance += self._behavioral_distance(candidate_feat, features_cache[other.hash])
                            count += 1
                
                diversity_bonus = total_distance / count if count > 0 else 0.0
                base_score = fitness_scores[candidate_hash]
                total_score = base_score + diversity_bonus * self.diversity_weight * 10
                
                if total_score > best_score:
                    best_score = total_score
                    best_candidate = candidate
            
            return best_candidate
        
        parent1 = select_one_cached()
        parent2 = select_one_cached()
        
        # Quick diversity enforcement
        if parent1.hash == parent2.hash:
            # Select most different candidate from tournament
            candidates = random.sample(population, tournament_size)
            different_candidates = [c for c in candidates if c.hash != parent1.hash]
            if different_candidates:
                parent2 = different_candidates[0]
        
        return parent1, parent2
        
    def _diversity_based_selection(self, fitness_scores):
        """
        Select parents balancing fitness and diversity
        Uses tournament selection with diversity consideration
        """
        return self._diversity_based_selection_cached(fitness_scores)

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