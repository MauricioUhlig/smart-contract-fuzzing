#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from itertools import accumulate
from bisect import bisect_right

from engine.operators import DataDependencyLinearRankingSelection

from ...plugin_interfaces.operators.selection import Selection

class DiversityLinearRankingSelection(Selection):
    def __init__(self, env, pmin=0.1, pmax=0.9):
        self.env = env
        '''
        Selection operator using Linear Ranking selection method.

        Reference: Baker J E. Adaptive selection methods for genetic
        algorithms[C]//Proceedings of an International Conference on Genetic
        Algorithms and their applications. 1985: 101-111.
        '''
        # Selection probabilities for the worst and best individuals.
        self.pmin, self.pmax = pmin, pmax

    def select(self, population, fitness):
        '''
        Select a pair of parent individuals using linear ranking method.
        '''

        # Add rank to all individuals in population.
        all_fits = population.all_fits(fitness)
        indvs = population.individuals
        sorted_indvs = sorted(indvs, key=lambda indv: all_fits[indvs.index(indv)])

        # Individual number.
        NP = len(sorted_indvs)

        # Assign selection probabilities linearly.
        # NOTE: Here the rank i belongs to {1, ..., N}
        p = lambda i: (self.pmin + (self.pmax - self.pmin)*(i-1)/(NP-1))
        probabilities = [self.pmin] + [p(i) for i in range(2, NP)] + [self.pmax]

        # Normalize probabilities.
        psum = sum(probabilities)
        wheel = list(accumulate([p/psum for p in probabilities]))

        # Select parents.
        father_idx = bisect_right(wheel, random.random())
        father = sorted_indvs[father_idx]

        father_reads, father_writes = DataDependencyLinearRankingSelection.extract_reads_and_writes(father, self.env)
        f_a = [i["arguments"][0] for i in father.chromosome]

        random.shuffle(indvs)
        for ind in indvs:
            i_a = [i["arguments"][0] for i in ind.chromosome]
            if f_a != i_a:
                i_reads, i_writes = DataDependencyLinearRankingSelection.extract_reads_and_writes(ind, self.env)
                if not i_reads.isdisjoint(father_writes) or not father_reads.isdisjoint(i_writes):
                    return father, ind

        
        mother = DiversityLinearRankingSelection.find_most_diverse_from_individual(population, father, 20)

        return father, mother

    @staticmethod
    def find_most_diverse_from_individual(population, target_individual, sample_size=20):
        """
        Find the most diverse individual (greatest distance) from the target individual
        Returns: (most_diverse_individual, max_distance)
        """
        
        if len(population) < 2:
            return None

        # Remove target individual from population for comparison
        other_individuals = [ind for ind in population if ind.hash != target_individual.hash]
        
        if len(population) > sample_size:
            other_individuals = random.sample(other_individuals, sample_size)
        
        if not other_individuals:
            return None
        
        return DiversityLinearRankingSelection.find_most_diverse(target_individual, other_individuals)


    @staticmethod
    def find_most_diverse(target_individual, other_individuals):
        """
        Find most diverse individual using exhaustive search
        """
        # Extract features for target individual
        target_features = DiversityLinearRankingSelection.extract_behavioral_features(target_individual)
        
        max_distance = -float('inf')
        most_diverse = None
        
        # Compare against all other individuals
        for other in other_individuals:
            other_features = DiversityLinearRankingSelection.extract_behavioral_features(other)
            distance = DiversityLinearRankingSelection.behavioral_distance(target_features, other_features)
            
            if distance > max_distance:
                max_distance = distance
                most_diverse = other
        
        return most_diverse

    @staticmethod
    def extract_behavioral_features(individual):
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

    @staticmethod
    def behavioral_distance(features1, features2):
        """
        Optimized behavioral distance calculation using precomputed features
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