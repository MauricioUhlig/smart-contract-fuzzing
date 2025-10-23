#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Particle Swarm Optimization engine definition '''

import math
import time
import logging
import random

from functools import wraps

# Imports for profiling.
import cProfile
import pstats
import os

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


class ParticleSwarmEngine(object):
    # Statistical attributes for population.
    fmax, fmin, fmean = StatVar('fmax'), StatVar('fmin'), StatVar('fmean')
    ori_fmax, ori_fmin, ori_fmean = (StatVar('ori_fmax'),
                                     StatVar('ori_fmin'),
                                     StatVar('ori_fmean'))

    def __init__(self, population, fitness=None, analysis=None, mapping=None, 
                 w=0.7, c1=1.5, c2=1.5):
        '''
        Initialize Particle Swarm Optimization engine.
        
        :param population: Population of particles (individuals)
        :param fitness: Fitness function
        :param analysis: Analysis plugins
        :param mapping: Function signature mapping
        :param w: Inertia weight (default: 0.7)
        :param c1: Cognitive coefficient (default: 1.5)
        :param c2: Social coefficient (default: 1.5)
        '''
        # Set logger.
        logger_name = 'engine.{}'.format(self.__class__.__name__)
        self.logger = logging.getLogger(logger_name)

        # Attributes assignment.
        self.population = population
        self.fitness = fitness
        self.analysis = [] if analysis is None else [a() for a in analysis]
        self.mapping = mapping

        # PSO parameters
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive coefficient
        self.c2 = c2  # Social coefficient

        # Maxima and minima in population.
        self._fmax, self._fmin, self._fmean = None, None, None
        self._ori_fmax, self._ori_fmin, self._ori_fmean = None, None, None

        # Default fitness functions.
        self.ori_fitness = None if self.fitness is None else self.fitness

        # Store current generation number.
        self.current_generation = -1  # Starts from 0.

        # PSO-specific: velocities, personal best, and global best
        self.velocities = []
        self.personal_best = []
        self.personal_best_fitness = []
        self.global_best = None
        self.global_best_fitness = float('-inf')

        # Check parameters validity.
        self._check_parameters()

    @do_profile(filename='pso_engine_run.prof')
    def run(self, ng):
        '''
        Run the Particle Swarm Optimization iteration with specified parameters.
        '''
        try:
            execution_begin = time.time()

            if self.fitness is None:
                raise AttributeError('No fitness function in PSO engine')

            # Initialize PSO-specific structures
            self._initialize_pso()

            # # Setup analysis objects.
            # for a in self.analysis:
            #     a.setup(ng=ng, engine=self)
            #     a.register_step(g=-1, population=self.population, engine=self)

            # Enter PSO iteration.
            g = 0
            while g < ng or settings.GLOBAL_TIMEOUT:
                if settings.GLOBAL_TIMEOUT and time.time() - execution_begin >= settings.GLOBAL_TIMEOUT:
                    break

                self.current_generation = g

                all_fitness = self.population.all_fits(fitness=self.fitness)

                # Update each particle
                for i in range(self.population.size):
                    particle = self.population.individuals[i]
                    
                    # # Execute the particle to get coverage data
                    # if self.analysis:
                    #     for analyzer in self.analysis:
                    #         if hasattr(analyzer, 'env'):
                    #             analyzer.register_step(g=-1, population=self.population, engine=self)
                    # Evaluate fitness
                    current_fitness = all_fitness[i] # self.fitness(particle)
                    # print(current_fitness, self.personal_best_fitness[g], self.global_best_fitness)
                    
                    # Update personal best
                    if current_fitness > self.personal_best_fitness[i]:
                        self.personal_best[i] = particle.clone()
                        self.personal_best_fitness[i] = current_fitness
                    
                    # Update global best
                    if current_fitness > self.global_best_fitness:
                        self.global_best = particle.clone()
                        self.global_best_fitness = current_fitness
                
                # Update velocities and positions for all particles
                new_particles = []
                for i in range(self.population.size):
                    new_particle = self._update_particle(i)
                    new_particles.append(new_particle)
                
                # Replace population with updated particles
                self.population.individuals = new_particles

                # Run all analysis if needed.
                for a in self.analysis:
                    if g % a.interval == 0:
                        a.register_step(g=g, population=self.population, engine=self)

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

    def _initialize_pso(self):
        '''
        Initialize PSO-specific structures: velocities, personal best, global best.
        '''
        self.velocities = []
        self.personal_best = []
        self.personal_best_fitness = []

         # This ensures the individual has been executed and coverage data exists
        if hasattr(self, 'analysis') and self.analysis:
            for analyzer in self.analysis:
                if hasattr(analyzer, 'env'):
                    # Execute the transaction to populate coverage
                    analyzer.register_step(g=-1, population=self.population, engine=self)
        
        all_fitness = self.population.all_fits(self.fitness)                    
        
        for i in range(self.population.size):
            particle = self.population.individuals[i]
            # Initialize velocity as empty (will be built during updates)
            self.velocities.append([])
            
            # Initialize personal best as current position
            self.personal_best.append(particle.clone())
            
            # Execute the individual first to populate coverage data
            try:
                # Now calculate fitness
                fitness_val = all_fitness[i] # self.fitness(particle)
                self.personal_best_fitness.append(fitness_val)
                
                # Update global best
                if fitness_val > self.global_best_fitness:
                    self.global_best = particle.clone()
                    self.global_best_fitness = fitness_val
                    
            except KeyError as e:
                # If there's still a KeyError, initialize with minimum fitness
                self.logger.warning(f"KeyError during PSO initialization for particle: {e}")
                self.personal_best_fitness.append(0.0)
                if self.global_best is None:
                    self.global_best = particle.clone()
                    self.global_best_fitness = 0.0

    def _update_particle(self, particle_idx):
        '''
        Update a particle's position based on PSO velocity update equations.
        
        :param particle_idx: Index of the particle to update
        :return: Updated particle (Individual)
        '''
        current_particle = self.population.individuals[particle_idx]
        personal_best = self.personal_best[particle_idx]
        
        # Create new particle by combining current, personal best, and global best
        new_particle = current_particle.clone()
        
        # PSO update: combine influence from current position, personal best, and global best
        r1 = random.random()
        r2 = random.random()
        
        # Determine which chromosome to use based on PSO weights
        # w: inertia (keep current), c1*r1: cognitive (move toward personal best), c2*r2: social (move toward global best)
        choice = random.random()
        
        if choice < self.w:
            # Keep current position (inertia)
            pass
        elif choice < self.w + self.c1 * r1:
            # Move toward personal best (cognitive component)
            new_particle = self._blend_particles(current_particle, personal_best, 0.5)
        else:
            # Move toward global best (social component)
            new_particle = self._blend_particles(current_particle, self.global_best, 0.5)
        
        # Apply mutation-like perturbation for exploration
        if random.random() < 0.1:  # 10% chance of perturbation
            new_particle = self._perturb_particle(new_particle)
        
        return new_particle

    def _blend_particles(self, particle1, particle2, blend_ratio):
        '''
        Blend two particles by combining their chromosomes.
        
        :param particle1: First particle
        :param particle2: Second particle
        :param blend_ratio: Ratio for blending (0.5 = equal blend)
        :return: Blended particle
        '''
        new_particle = particle1.clone()
        
        # Randomly select genes from both particles
        max_len = min(len(particle1.chromosome), len(particle2.chromosome))
        if max_len > 0:
            new_chromosome = []
            for i in range(max_len):
                if random.random() < blend_ratio:
                    new_chromosome.append(particle1.chromosome[i])
                else:
                    new_chromosome.append(particle2.chromosome[i])
            
            # Add remaining genes from longer chromosome
            if len(particle1.chromosome) > max_len:
                new_chromosome.extend(particle1.chromosome[max_len:])
            elif len(particle2.chromosome) > max_len:
                new_chromosome.extend(particle2.chromosome[max_len:])
            
            new_particle.chromosome = new_chromosome
            new_particle.solution = new_particle.decode()
        
        return new_particle

    def _perturb_particle(self, particle):
        '''
        Apply small random perturbations to a particle for exploration.
        
        :param particle: Particle to perturb
        :return: Perturbed particle
        '''
        perturbed = particle.clone()
        
        if len(perturbed.chromosome) > 0:
            # Randomly modify one gene
            gene_idx = random.randint(0, len(perturbed.chromosome) - 1)
            gene = perturbed.chromosome[gene_idx]
            
            # Randomly select what to perturb
            perturbation_type = random.choice(['amount', 'gaslimit', 'account'])
            
            if perturbation_type == 'amount' and 'amount' in gene:
                gene['amount'] = random.randint(0, 1000)
            elif perturbation_type == 'gaslimit' and 'gaslimit' in gene:
                gene['gaslimit'] = random.randint(100000, 8000000)
            elif perturbation_type == 'account' and 'account' in gene:
                # This would need access to available accounts
                pass
            
            perturbed.solution = perturbed.decode()
        
        return perturbed

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
        A decorator for analysis regsiter.
        '''
        if not issubclass(analysis_cls, OnTheFlyAnalysis):
            raise TypeError('analysis class must be subclass of OnTheFlyAnalysis')

        # Add analysis instance to engine.
        analysis = analysis_cls()
        self.analysis.append(analysis)

    # Functions for fitness scaling.

    def linear_scaling(self, target='max', ksi=0.5):
        '''
        A decorator constructor for fitness function linear scaling.

        :param target: The optimization target, maximization or minimization.
        :type target: str, 'max' or 'min'

        :param ksi: Selective pressure adjustment value.
        :type ksi: float

        Linear Scaling:
            1. arg max f(x), then f' = f - min{f(x)} + ksi;
            2. arg min f(x), then f' = max{f(x)} - f(x) + ksi;
        '''
        def _linear_scaling(fn):
            # For original fitness calculation.
            self.ori_fitness = fn

            @wraps(fn)
            def _fn_with_linear_scaling(indv):
                # Original fitness value.
                f = fn(indv)

                # Determine the value of a and b.
                if target == 'max':
                    f_prime = f - self.ori_fmin + ksi
                elif target == 'min':
                    f_prime = self.ori_fmax - f + ksi
                else:
                    raise ValueError('Invalid target type({})'.format(target))
                return f_prime

            return _fn_with_linear_scaling

        return _linear_scaling

    def dynamic_linear_scaling(self, target='max', ksi0=2, r=0.9):
        '''
        A decorator constructor for fitness dynamic linear scaling.

        :param target: The optimization target, maximization or minimization.
        :type target: str, 'max' or 'min'

        :param ksi0: Initial selective pressure adjustment value, default value
                     is 2
        :type ksi0: float

        :param r: The reduction factor for selective pressure adjustment value,
                  ksi^(k-1)*r is the adjustment value for generation k, default
                  value is 0.9
        :type r: float in range [0.9, 0.999]

        Dynamic Linear Scaling:
            For maximizaiton, f' = f(x) - min{f(x)} + ksi^k, k is generation number.
        '''
        def _dynamic_linear_scaling(fn):
            # For original fitness calculation.
            self.ori_fitness = fn

            @wraps(fn)
            def _fn_with_dynamic_linear_scaling(indv):
                f = fn(indv)
                k = self.current_generation + 1

                if target == 'max':
                    f_prime = f - self.ori_fmin + ksi0*(r**k)
                elif target == 'min':
                    f_prime = self.ori_fmax - f + ksi0*(r**k)
                else:
                    raise ValueError('Invalid target type({})'.format(target))
                return f_prime

            return _fn_with_dynamic_linear_scaling

        return _dynamic_linear_scaling

    def minimize(self, fn):
        '''
        A decorator for minimizing the fitness function.
        '''
        @wraps(fn)
        def _minimize(indv):
            return -fn(indv)
        return _minimize
