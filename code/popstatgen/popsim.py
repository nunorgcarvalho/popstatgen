'''
This file contains classes related to population and statistical genetics.
The classes present are:
- Population: Models a population of individuals with genotypes. Attributes store raw and summary data on such individuals, including traits.
- Trait: Models a trait in a population, including its individual components.
These classes largely contain wrapper methods that call functions from other files. These wrapper methods extract the necessary attributes from the class object and pass them as arguments to the functions.
These wrapper methods often just update the class attributes, instead of returning the results.
'''

# imports
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union
from scipy import sparse
from . import popgen_functions as pop

class Population:
    '''
    Class for a population to simulate. Contains genotype information. Contains methods to simulate change in population over time.
    '''

    ########################
    #### Initialization ####
    ########################

    def __init__(self, N: int, M: int, P: int = 2,
                 p_init: Union[float, np.ndarray] = None,
                 seed: int = None):
        '''
        Initializes a population, simulating initial genotypes.

        Parameters:
            N (int): {opulation size of individuals (not haplotypes).
            M (int): Total number of variants in genome.
            P (int): ploidy of genotpes. Default is 2 (diploid).
            p_init (float or array): Initial allele frequency of variants. If only a single value is provided, it is treated as the initial allele frequency for all variants. Otherwise, must be an array of length M. Default is uniform distribution between 0.05 and 0.95.
            seed (int): Initial seed to use when simulating genotypes (and allele frequencies if necessary).
        '''
        # defines properties of instance
        self.N = N
        self.M = M
        self.P = P
        self.t = 0 # generation
        self.T_breaks = [self.t]
        self.traits = {}
        # stores variants' positions
        self.BPs = np.arange(M)
        # sets seed if specified
        if seed is not None:
            seed = np.random.seed(seed)
        # draws initial allele frequencies from uniform distribution between 0.05 and 0.95 if not specified
        if p_init is None:
            p_init = pop.draw_p_init(M, method = 'uniform', params = (0.05, 0.95))
        elif type(p_init) == float or type(p_init) == int:
            # if only single value is given, all variants have same initial frequency
            p_init = np.full(M, p_init)

        # generates initial genotypes and records allele frequencies
        H = self._generate_unrelated_haplotypes(p_init)
        self._update_obj(H=H)
        self.ps = np.expand_dims(self.p, axis=0)

        # generates recombination rates
        self.R = pop.generate_LD_blocks(M)
    
    ###################################
    #### Storing object attributes ####
    ###################################

    def _update_obj(self, H: np.ndarray = None, K: np.ndarray = None,
                    get_GRM: bool = False, get_corr_matrix: bool = False):
        '''
        Update the population object's attributes.

        Parameters:
            H (3D array): Haplotype array.
            K (2D array): Kinship array.
            GRM (bool): Whether the GRM should computed from the genotype matrix. Default is false.
        '''
        if H is not None:
            self.H = H
            self.G = pop.make_G(self.H)
            self.p = pop.compute_freqs(self.G, self.P)
            self.X = pop.standardize_G(self.G, self.p, self.P, impute=True, std_method='observed')
            if get_GRM:
                self.GRM = self.get_GRM(self.G, self.p, self.P)
        if K is not None:
            self.K = K
        if get_corr_matrix:
            self.corr_matrix = self.get_corr_matrix()
    
    def store_neighbor_matrix(self, LDwindow: float = None) -> sparse.coo_matrix:
        '''
        Stores boolean sparse matrix of variants (`neighbor_matrix`) within the specified LD window distance (`LDwindow`) based on object's variant positions (currently only supports `BPs`). Calls `make_neighbor_matrix()` from `popgen_functions.py`.
        Parameters:
            LDwindow (float): Maximum distance between variants to be considered neighbors. In the same units as the positions used. If not provided, defaults to infinite maximum distance.
        '''
        # by default, uses object's base pair positions
        positions = self.BPs
        self.neighbor_matrix = pop.make_neighbor_matrix(positions=positions, LDwindow=LDwindow)

    def store_LD_matrix(self):
        '''
        Stores the correlation matrix (`corr_matrix`) and its square, the LD matrix (`LD_matrix`). Must have pre-computed `neighbor_matrix` attribute (see `store_neighbor_matrix()`).

        Parameters:
            corr_matrix (sparse 2D matrix (CSR)): M*M correlation matrix between pairs of variants in CSR format. Can be computed with `get_corr_matrix()`.
        Returns:
            LD_matrix (sparse 2D matrix (CSR)): M*M LD matrix (square of correlation) between pairs of variants in CSR format.
        '''
        if not hasattr(self, 'neighbor_matrix'):
            raise Exception('Must have pre-computed neighbor_matrix. Use `Population.get_neighbor_matrix()`.')
        self.corr_matrix = pop.compute_corr_matrix(self.X, self.neighbor_matrix)
        self.LD_matrix = pop.compute_LD_matrix(self.corr_matrix)

    def add_trait(self, name: str, seed: int = None, **kwargs):
        '''
        Initializes and generates trait.
        Parameters:
            name (str): Name of trait.
            seed (int): Seed for random number generation.
            **kwargs: All other arguments are passed to the Trait constructor. See Trait.__init__ for details.
        '''
        if seed is not None:
            np.random.seed(seed)
        self.traits[name] = Trait(self.X, **kwargs)
        # stores actual heritability value
        self.traits[name].h2_true = self.traits[name].get_h2_true()
        self.traits[name].h2_trues = np.expand_dims(self.traits[name].h2_true, axis=0)
    
    def update_traits(self, fixed_h2: bool = True, traits: list = None):
        '''
        Updates all traits by generating based on the current genotype matrix. Random noise components are re-generated. Causal genetic effects remain fixed.

        Parameters:
            fixed_h2 (bool): Whether the variance of the noise component should be updated to maintain the heritability. Default is True.
            traits (list of str): List of trait names to update. If None, updates all traits in the object.
        '''
        if traits is None:
            traits = self.traits.keys()
        # loops through each trait
        for key in self.traits:
            if key not in traits:
                continue
            trait = self.traits[key]
            trait.generate_trait(self.X, fixed_h2=fixed_h2)
            # computes true heritability
            trait.h2_true = trait.get_h2_true()
            # adds to running heritability history
            t_last = trait.h2_trues.shape[0]
            h2_add = np.full(self.t - t_last + 1, np.nan)
            h2_add[-1] = trait.h2_true
            trait.h2_trues = np.concatenate( (trait.h2_trues, h2_add) )

    #######################################
    #### Analysis of object attributes ####
    #######################################

    def get_fixation_t(self, ps: np.ndarray = None) -> np.ndarray:
        '''
        For each variant, finds *first* generation (returned as an index) for which the allele frequency is 0 (loss) or 1 (fixation). A value of -1 means the variant never got fixed

        Parameters:
            ps (2D array): T*M matrix (where T is number of generations) containing allele frequencies over time. Defaults to object's allele frequency history.
        
        Returns:
            t_fix (1D array): Array of length M with the first generation (as an index) for which the respective allele was lost or fixed. If the allele was not fixed by the most recent simulation, a -1 is returned.
        '''
        # uses population's allele frequency history if not specified
        if ps is None:
            ps = self.ps
        # gets mask of whether frequency is 0 or 1
        ps_mask = np.any((ps == 0, ps == 1), axis=0)
        # finds first instance of True for each variant
        t_fix = np.where(ps_mask.any(axis=0), ps_mask.argmax(axis=0), -1)        
        return t_fix
    
    def summarize_ps(self, ps: np.ndarray = None, quantiles: tuple = (0.25, 0.5, 0.75)) -> tuple[np.ndarray, np.ndarray]:
        '''
        Returns the mean as well as the specified quantiles of variants across each generation.

        Parameters:
            ps (2D array): T*M matrix (where T is number of generations) containing allele frequencies over time. Defaults to object's allele frequency history.
            quantiles (tuple): List of quantiles (e.g. 0.99) of allele frequencies across variants at each generation to plot. `summarize` must be set to True. Default is median, lower quartile, and upper quartile.
        
        Returns:
            tuple ((ps_mean, ps_quantile)):
            Where:
            - ps_mean (1D array): Array of length T (where T is the total number of generations) of mean allele frequency at each generation.
            - ps_quantile (2D array): K*T matrix (where K is the number of quantiles specified) of allele frequency for each quantile at each generation.
        '''
        # uses population's allele frequency history if not specified
        if ps is None:
            ps = self.ps
        # computes mean allele frequency over time
        ps_mean = self.ps.mean(axis=1)
        # computes quantiles over time
        ps_quantile = np.quantile(self.ps, quantiles, axis=1)
        return (ps_mean, ps_quantile)
    
    ####################################
    #### Simulating forward in time ####
    ####################################

    def _generate_unrelated_haplotypes(self, p: np.ndarray) -> np.ndarray:
        '''
        Generates 3-dimensional matrix of population haplotypes.

        Parameters:
            p (1D array): Array of allele frequencies to draw alleles from.
        Returns:
            H (3D array): N*M*P array of alleles. First dimension is individuals, second dimension is variants, third dimension is haplotype number (related to ploidy). Each element is either a 0 or a 1.
        '''
        p = p.reshape(1, self.M, 1)
        H = np.random.binomial( 1, p = p, size = (self.N, self.M, self.P))
        return H
        
    def next_generation(self, s: Union[float, np.ndarray] = 0.0,
                        mu: Union[float, np.ndarray] = 0.0):
        '''
        Simulates new generation. Doesn't simulate offspring directly, meaning that future offspring have haplotypes drawn randomly from allele frequencies. Automatically updates object.

        Parameters:
            s (float or 1D array): Selection coefficient, such that an individual with the alternate allele has a (1+s) relative fitness compared to the reference allele. Occurs before mutation. If only a single value is provided, it is treated as the selection coefficient for all variants. Otherwise, must be an array of length M. Default is 0 (no selection).
            mu (float or 1D array): Mutation rate, such that the probability of any individual allele flipping to its alternate in the next generation is given by mu. Occurs after selection (i.e. mutation occurs in germline of current generation). Default is 0 (no mutations).
        '''
        # assigns same selection coefficient/mutation rate to all variants if only single value specified
        if isinstance(s, (float, int)):
            s = np.full(self.M, s)
        if isinstance(mu, (float, int)):
            mu = np.full(self.M, mu)
        # analytical allele frequency from which next generations' alleles are drawn from
        p = self.p
        # effect of selection
        p = p * (1 + s) / (1 + p*s)
        # effect of mutation (in gametes that lead to new generation, i.e. post-selection)
        p = p*(1-mu) + (1-p)*mu
        # effect of genetic drift
        H = self._generate_unrelated_haplotypes(p)

        self.t = self.t + 1
        self._update_obj(H=H)

    def simulate_generations(self, generations: int, related_offspring: bool = False,
                             trait_updates: bool = False, fixed_h2: bool = False,
                             **kwargs):
        '''
        Simulates specified number of generations beyond current generation. Can simulate offspring directly. Automatically updates object. Recombination rates are extracted from object attributes.

        Parameters:
            generations (int): Number of generations to simulate (beyond the current generation).
            related_offspring (bool): Whether the offspring of the next generation should be directly related to parents from previous generation by simulating meiosis and haplotype transfer. Default is False, meaning that future offspring have haplotypes drawn randomly from allele frequencies.
            trait_updates (bool): Whether to update traits after each generation. Default is False, meaning that traits are only updated at the end of the simulation.
            **kwargs: All other arguments are passed to the `next_generation` or `generate_offspring` methods. See those methods for details.
        '''
        # keeps track of allele frequencies over generations if specified
        previous_gens = self.ps.shape[0]
        ps = np.full( (previous_gens + generations, self.M), np.nan)
        ps[0:previous_gens,] = self.ps
        
        # loops through each generation
        for t in range(generations):
            if related_offspring:
                self.generate_offspring(**kwargs)
            else:
                self.next_generation(**kwargs)
            # records metrics
            ps[previous_gens + t,] = self.p
            if trait_updates:
                self.update_traits(fixed_h2=fixed_h2)
        self.T_breaks.append(previous_gens + generations)
        if not trait_updates:
            self.update_traits(fixed_h2=fixed_h2)
        # saves metrics to object
        self.ps = ps

    def _pair_mates(self, AM_r: float = 0, AM_trait: Union[str, np.ndarray] = None,
                    AM_type: str = 'phenotypic') -> np.ndarray:
        '''
        Pairs individuals up monogamously to mate and generate offspring. Population size must be multiple of 2. Allows for assortative mating (AM) if specified.

        Parameters:
            AM_r (float): Desired correlation between AM trait values of spouses. Default is 0 (no assortative mating).
            AM_trait (str or 1D array): If a string, the name of the trait to use for assortative mating (as stored in the object). If an array, the trait values to use for assortative mating. If None, no assortative mating is performed. Default is None.
            AM_type (str): Type of assortative mating to perform. If 'phenotypic', uses the trait values directly. If 'genetic', uses the genetic values (only for object's stored traits). Default is 'phenotypic'.

        Returns:
            tuple ((iM, iP)):
            Where:
            - iMs (1D array): Array of length N/2 containing indices of the mothers.
            - iPs (1D array): Array of length N/2 containing indices of the fathers.
        '''
        # checks for population size
        if self.N % 2 != 0:
            raise Exception('Population size must be multiple of 2.')
        N2 = self.N // 2

        # extracts assortative mating value:
        if AM_r != 0 and AM_trait is not None:
            if type(AM_trait) == np.ndarray:
                AM_values = AM_trait
            elif type(AM_trait) == str:
                if AM_trait not in self.traits:
                    raise Exception(f'Trait {AM_trait} not found in population traits.')
                if AM_type == 'phenotypic':
                    AM_values = self.traits[AM_trait].y
                elif AM_type == 'genetic':
                    AM_values = self.traits[AM_trait].genetic_value
            # standardizes
            AM_values = (AM_values - AM_values.mean()) / AM_values.std()
        else:
            AM_values = np.zeros(self.N)

        # randomly splits up population into maternal (M) and paternal (P) halves 
        # also shuffles their order
        iMs = np.random.choice(self.N, N2, replace=False)
        iPs = np.setdiff1d(np.arange(self.N), iMs)

        # computes mate value
        mate_values = AM_r * AM_values + np.random.normal(scale = np.sqrt(1-AM_r**2), size=self.N)
        # sorts mothers and fathers by mate value
        iMs = iMs[np.argsort(AM_values[iMs])]
        iPs = iPs[np.argsort(mate_values[iPs])]    

        return (iMs, iPs)

    def generate_offspring(self, s: Union[float, np.ndarray] = 0,
                           mu: Union[float, np.ndarray] = 0,
                           **kwargs):
        '''
        Pairs up mates and generates offspring for parents' haplotypes. Only works for diploids. Each pair always has two offspring. Recombination rates are extracted from object attributes.

        Parameters:
            s (float or 1D array): Selection coefficient, such that an individual with the alternate allele has a (1+s) relative fitness compared to the reference allele. Occurs before mutation. If only a single value is provided, it is treated as the selection coefficient for all variants. Otherwise, must be an array of length M. Default is 0 (no selection).
            mu (float or 1D array): Mutation rate, such that the probability of any individual allele flipping to its alternate in the next generation is given by mu. Occurs after selection (i.e. mutation occurs in germline of current generation). Default is 0 (no mutations).
            **kwargs: All other arguments (related to assortative mating) are passed to the `_pair_mates` method. See that method for details.
        
        Returns:
            H (3D array): N*M*P array of offspring haplotypes. First dimension is individuals, second dimension is variants, and third dimension is haplotype number (related to ploidy). Each element is either a 0 or a 1.
        '''
        # checks ploidy
        if self.P != 2:
            raise Exception('Offspring generation only works for diploids.')

        # Assortative Mating ####        
        # pairs up mates
        iMs, iPs = self._pair_mates()

        # Selection ####
        if isinstance(s, (float, int)):
            s = np.full(self.M, s)
        # computes individuals' breeding weight (assumes linear additive fitness effects)
        W = np.exp( (np.log(1 + self.G * s[None, :])).sum(axis=1) )
        # computes each pair's breeding weight
        W_pair = W[iMs] * W[iPs]
        # computs probability of each parent pair being chosen to mate for one offspring
        P_mate = W_pair / W_pair.sum()
        # determines population size of next generation (currently maintains population size)
        N_offspring = self.N
        # draws indices of parents for each offspring
        i_mate = np.random.choice(np.arange(len(iMs)), size=N_offspring, p=P_mate)
        parents = np.stack((iMs[i_mate], iPs[i_mate]), axis=1)
        
        # Drift + Recombination ####
        # generates variants for which a crossover event happens for each parent of each offspring
        crossover_events = np.random.binomial(n=1, p=self.R.reshape(1, self.M, 1),
                                              size=(N_offspring, self.M, 2))
        # determines the shift in haplotype phase for each parent's haplotype at each variant
        haplo_phase = np.cumsum(crossover_events, axis=1)
        # randomly chooses haplotype to start with for each parent of each offspring
        haplo_k0 = np.random.binomial(n=1, p=1/self.P, size = (N_offspring ,2))
        # adds the starting shift and gets the modulo so that haplotype phase acts as an index
        haplo_ks = (haplo_k0[:, None, :] + haplo_phase) % 2

        H = np.empty((N_offspring, self.M, self.P), dtype=int)
        for i in np.arange(N_offspring):
            iM = parents[i,0]
            iP = parents[i,1]
            # extract allele from correct haplotype of each parent
            haploM = self.H[iM, np.arange(self.M), haplo_ks[i, :, 0]]
            haploP = self.H[iP, np.arange(self.M), haplo_ks[i, :, 1]]
            haplos = np.stack((haploM, haploP), axis = 1)
            # shuffles haplotypes around
            haplos = haplos[:,np.random.choice(self.P, size=self.P, replace=False)]
            H[i,:,:] = haplos

        # Mutations ####
        if isinstance(mu, (float, int)):
            mu = np.full(self.M, mu)
        mutations = np.random.binomial(n=1, p=mu[None,:,None], size = (N_offspring, self.M, self.P))
        # Apply mutations by flipping alleles (0 to 1 or 1 to 0) based on the mutation matrix
        H = (H + mutations) % 2

        self.t = self.t + 1
        self._update_obj(H=H)

    #######################
    #### Visualization ####
    #######################

    def plot_over_time(self,
                       metrics: np.ndarray,
                       ts: np.ndarray = None,
                       aes: dict = {'title': None, 'xlabel': None, 'ylabel': None},
                       aes_line: dict = {'color': None, 'ls': None, 'labels': None},
                       vlines: np.ndarray = None,
                       legend: bool = True):
        '''
        General function for plotting some metric over time.

        Parameters:
            metric (1D or 2D array): Metric(s) to plot over time. If a T*K 2D matrix, each column is treated as a different line to plot.
            ts (1D array): Array of time points (generations) corresponding to the metric. If not specified, defaults to the range of generations in the metric.
            aes (dict): Dictionary of aesthetic parameters for the plot.
            aes_line (dict): Dictionary of aesthetic parameters for the lines in the plot.
            vlines (1D array): Array of time points at which to draw vertical lines. Default is None, meaning no vertical lines are drawn.
            legend (bool): Whether to include a legend in the plot for each line. Default is False.
        '''
        if metrics.ndim == 1:
            metrics = metrics.reshape(-1, 1)
        K = metrics.shape[1]
        # fills out aes_line settings
        # color
        if aes_line['color'] is None:
            aes_line['color'] = self._get_default_colors(K)
        if type(aes_line['color']) == str:
            aes_line['color'] = [aes_line['color']] * K
        colors = aes_line['color']
        # line style
        if aes_line['ls'] is None:
            aes_line['ls'] = '-'
        if type(aes_line['ls']) == str:
            aes_line['ls'] = [aes_line['ls']] * K
        ls = aes_line['ls']
        # labels
        if aes_line['labels'] is None:
            aes_line['labels'] = [f'Line {j}' for j in range(K)] # not shown
            legend=False
        labels = aes_line['labels']

        if ts is None:
            ts = np.arange(metrics.shape[0])

        # plotting
        plt.figure(figsize=(8, 5))
        # plots lines
        for j in range(K):
            plt.plot(ts, metrics[:, j],
                     color=colors[j],
                     ls=ls[j],
                     label=labels[j])
        # vertical lines
        if vlines is not None:
            for t in vlines:
                plt.axvline(t, ls='--', color='black')
        # labels
        if aes['xlabel'] is not None:
            plt.xlabel(aes['xlabel'])
        if aes['ylabel'] is not None:
            plt.ylabel(aes['ylabel'])
        if aes['title'] is not None:
            plt.title(aes['title'])
        plt.xlim(ts.min(), ts.max())
        plt.ylim(0, 1)
        if legend:
            plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_freq_over_time(self, ps: np.ndarray = None, j_keep: tuple = None,
                            legend=False, last_generations: int = None,
                            summarize: bool = False, quantiles: tuple = (0.25, 0.5, 0.75)):
        '''
        Plots variant allele frequencies over time.

        Parameters:
            ps (2D array): T*M matrix (where T is number of generations) containing allele frequencies over time. Defaults to object's allele frequency history.
            j_keep (tuple): Variant indices to include when plotting. Defaults to all variants.
            legend (bool): Whether to include a legend in the plot for each line. Default is False.
            last_generations (int): Number specifying the number of most recent generations to plot. Defaults to all generations since beginning.
            summarize (bool): If true, instead of plotting individual variant trajectories, it plots the mean and specified quantiles of allele frequencies across variants at each generation. Default is False.
            quantiles (tuple): List of quantiles (e.g. 0.99) of allele frequencies across variants at each generation to plot. `summarize` must be set to True. Default is median, lower quartile, and upper quartile.
        '''
        # plots all variants if not specified
        if j_keep is None:
            j_keep = tuple( range(self.M) )
        # uses population's allele frequency history if not specified
        if ps is None:
            ps = self.ps
        # plots all generations if not specified
        if last_generations is None:
            t_start = 0
        else:
            t_start = max(0,ps.shape[0] - last_generations)
        t_keep = tuple( range(t_start, ps.shape[0]))
        # subsets to specified variants
        ts = np.arange(t_start, ps.shape[0])
        ps = ps[np.ix_(t_keep, j_keep)]
        
        # if True, gets mean and quartiles for variants over time, which are plotted instead
        if summarize:
            ps_mean, ps_quantile = self.summarize_ps(ps, quantiles)
            metrics = np.column_stack((ps_mean, ps_quantile.T))
            aes_line = {'color': ['deepskyblue'] + ['lightskyblue'] * len(quantiles),
                        'ls': ['--'] + [':'] * len(quantiles),
                        'labels': ['Mean'] + [f'{q*100}% percentile' for q in quantiles]}
        else:
            # gets metrics to plot
            metrics = ps
            aes_line = {'color': None, 'ls': '-',
                        'labels': [f'Variant {j_keep[j]}' for j in range(len(j_keep))]}
        # plot aesthetics
        aes = {'title': 'Allele Frequency Trajectories Over Time',
               'xlabel': 'Generation',
               'ylabel': 'Allele Frequency'}

        self.plot_over_time(metrics, ts, aes=aes, aes_line=aes_line, vlines = self.T_breaks, legend=legend)
        
    def plot_LD_matrix(self, LD_matrix: sparse.csr_matrix = None,
                      plot_range: Tuple[int, int] = None, type: str = 'LD',
                      omit_mono: bool = False):
        '''
        Plots LD/correlation between variants.

        Parameters:
            LD_matrix (sparse 2D matrix): Either an LD or correlation M*M matrix of variants. If not provided, uses object's LD/correlation matrix.
            plot_range (tuple): A tuple of length 2 containing the range of variant indices to plot. If not provided, plots all variants.
            type (str): Uses color scheme for either 'LD' (default) or 'corr' matrix.
            omit_mono (bool): Whether variants that are monomorphic (p = 0 or 1) should be skipped over when plotting. Default is False.
        '''
        if LD_matrix is None:
            if type == 'LD':
                LD_matrix = self.LD_matrix
            elif type == 'corr':
                LD_matrix = self.corr_matrix
        if plot_range is not None:
            start, stop = plot_range
        else:
            start = 0
            stop = LD_matrix.shape[0]
        # Convert the sparse matrix to a dense array
        LD_matrix_dense = LD_matrix[start:stop, start:stop].toarray()
        
        # Skips plotting monomorphic variants
        if omit_mono:
            j_mono = (self.p == 0) | (self.p == 1)
            LD_matrix_dense = np.delete(LD_matrix_dense, j_mono, axis=0)
            LD_matrix_dense = np.delete(LD_matrix_dense, j_mono, axis=1)

        # Create the heatmap
        plt.figure(figsize=(10, 8))
        if type == 'LD':
            plt.imshow(LD_matrix_dense, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
        elif type == 'corr':
            plt.imshow(LD_matrix_dense, cmap='seismic', interpolation='nearest', vmin=-1, vmax=1)
        plt.colorbar(label='LD Value')
        plt.title(f'{type} Matrix Heatmap')
        plt.xlabel('Variant Index')
        plt.ylabel('Variant Index')
        plt.show()

    ########################
    #### Static Methods ####
    ########################

    @staticmethod
    def _get_default_colors(n_lines):
        '''
        Returns a list of colors for plotting lines, cycling through matplotlib's default color cycle.
        Parameters:
            n_lines (int): Number of lines to generate colors for.
        Returns:
            colors (list): List of colors for plotting lines.
        '''
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = [c['color'] for c in prop_cycle]
        # Repeat colors if n_lines > length of color cycle
        return [colors[i % len(colors)] for i in range(n_lines)]

class Trait:
    '''
    Class for a trait belonging to a Population object.
    '''

    def __init__(self, G: np.ndarray, M_causal: int = None,
                 var_G: float = 1.0, var_Eps: float = 0.0):
        '''
        Initializes and generates trait.

        Parameters:
            G (2D array): N*M NON-standardized genotype matrix.
            M_causal (int): Number of causal variants (variants with non-zero effect sizes). Default is all variants.
            var_G (float): Total expected variance contributed by per-standardized-allele genetic effects. Default is 1.0.
            var_Eps (float): Total expected variance contributed by random noise.
        '''
        # stores trait properties
        self.M = G.shape[1]
        self.var = {}
        self.var['G'] = var_G
        self.var['Eps'] = var_Eps
        # computes expected heritability
        self.h2 = var_G / (var_G + var_Eps)
        # computes causal effects
        self.effects, self.j_causal = self.generate_causal_effects(self.M, M_causal, self.var['G'])
        self.effects_per_allele = self.effects / G.std(axis=0) 
        self.M_causal = len(self.j_causal)
        # computes trait 
        self.generate_trait(G)

    @staticmethod
    def generate_causal_effects(M: int, M_causal: int = None, var_G: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        '''
        Generates variant effect sizes for some trait. Default interpretation is per-standardized-allele effect sizes. Non-causal variant effects are set to 0. Causal effects are drawn from normal distribution with mu=0 and sd=`var_G`/`M_causal`.

        Parameters:
            M (int): Total number of variants (causal and non-causal).
            M_causal (int): Number of causal variants (variants with non-zero effect sizes). Default is all variants
            var_G (float): Total expected variance contributed by per-standardized-allele genetic effects. Default is 1.0.
        Returns:
            causal_effects (1D array): Array of length M containing the effect sizes of all variants.
            j_causal (1D array): Array of length M_causal containing the variant indices of causal variants.
        '''
        if M_causal is None:
            M_causal = M
        causal_effects = np.zeros(M)
        j_causal = np.random.choice(M, M_causal, replace=False) 
        causal_effects[j_causal] = np.random.normal(0, np.sqrt(var_G/M_causal), M_causal)
        return causal_effects, j_causal
    
    @staticmethod
    def compute_genetic_value(G: np.ndarray, effects: np.ndarray) -> np.ndarray:
        '''
        Computes the genetic value/score given a (standardized) genotype matrix and (per-standardized-allele) effect sizes.

        Parameters:
            G (2D array): N*M NON-standardized genotype matrix.
            effects (1D array): Array of length M containing causal and non-causal genetic effects.
        Returns:
            y (1D array): Array of length N containing trait values.
        '''
        # dot product
        y = G @ effects
        return y
    
    @staticmethod
    def generate_noise_value(N: int, var_Eps: float = 0.0) -> np.ndarray:
        '''
        Generates noise component of trait drawn randomly from Normal distribution.

        Parameters:
            N (int): Number of individuals to generate noise component for.
            var_Eps (float): Variance of the noise component. Default is 0.
        Returns:
            noise_value (1D array): Array of length N containing noise component values.
        '''
        noise_value = np.random.normal(loc=0, scale=np.sqrt(var_Eps), size = N)
        return noise_value

    def generate_trait(self, G: np.ndarray, fixed_h2: bool = True):
        '''
        Generates/updates trait using stored genetic effects and recomputing other components. Automatically updates object's attributes, instead of returning trait values.

        Parameters:
            G (2D array): N*M NON-standardized genotype matrix.
            fixed_h2 (bool): Whether the variance of the noise component should be updated to maintain the heritability. Genetic component must be non-zero. Default is True.
        '''
        N = G.shape[0]
        # genetic component (using per-allele effects)
        self.genetic_value = self.compute_genetic_value(G, self.effects_per_allele)

        var_Eps = self.var['Eps']
        y_nonEps = self.genetic_value
        # recomputes non-noise component to get needed var_Eps to maintain heritability
        if fixed_h2:
            var_Eps = (y_nonEps.var() / self.h2) - y_nonEps.var()

        # random noise component
        self.noise_value = self.generate_noise_value(N,var_Eps)

        # computes actual trait as additive of individual components
        self.y = y_nonEps + self.noise_value

    def get_h2_true(self) -> float:
        '''
        Returns the true narrow-sense heritability of the trait, which is variance of the genetic component divided by the variance of the trait.
        
        Returns:
            h2_true (float): True heritability of the trait.
        '''
        h2_true = self.genetic_value.var() / self.y.var()
        return h2_true
        