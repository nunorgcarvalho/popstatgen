import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union
from scipy import sparse

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
        self.T_breaks = [0]
        # stores variants' positions
        self.BPs = np.arange(M)
        # sets seed if specified
        if seed is not None:
            seed = np.random.seed(seed)
        # draws initial allele frequencies from uniform distribution between 0.05 and 0.95 if not specified
        if p_init is None:
            p_init = self._draw_p_init(method = 'uniform', params = (0.05, 0.95))
        elif type(p_init) == float or type(p_init) == int:
            # if only single value is given, all variants have same initial frequency
            p_init = np.full(self.M, p_init)

        # generates initial genotypes and records allele frequencies
        H = self._generate_unrelated_haplotypes(p_init)
        self._update_obj(H=H)
        self.ps = np.expand_dims(self.p, axis=0)

        # generates initial kinship matrix
        self.K = np.diag(np.full(self.N, 1))

        # generates recombination rates
        self.R = self.generate_LD_blocks(M)

    def _draw_p_init(self, method: str, params: list) -> np.ndarray:
        '''
        Returns array of initial allele frequencies to simulate genotypes from.

        Parameters:
            method (str): Method of randomly drawing allele frequencies. Options:
                uniform: 'Draws from uniform distribution with lower and upper bounds specified by `params`.'
            params (list): Method-specific parameter values.
        Returns:
            p_init (1D array): Array of length M containing allele frequencies.
        '''
        # uniform sampling is default
        if method == 'uniform':
            p_init = np.random.uniform(params[0], params[1], self.M)
            return p_init
        else:
            return np.full(self.M, np.nan)
    
    @staticmethod
    def generate_LD_blocks(M: int, N_blocks: int = None,
                           hotspot_lambda: float = None, block_R: float = None):
        '''
        Generates an array of recombination rates that produces LD blocks when simulated. Base recombination rate is kept low, with a few recombination hotspots scattered throughout with much higher recombination rates. Hotspot recombination rates are drawn from an exponential distribution.

        Parameters:
            M (int): Number of variants
            N_blocks (int): Number of LD blocks. If not specified, defaults to whichever is higher between 2 and M // 10 (i.e. LD block every 10 variants on average). If 0 or 1, no recombination hotspots are formed.
            hotspot_lambda (float): Lambda parameter for exponential distribution for which hotspot recombination rates are drawn from. The mean recombination rate is 1/hotspot_lambda, and the variance is 1/hotspot_lambda^2. If not specified, defaults to M, meaning that ~1 crossover event is expected per genome per generation.
            block_R (float): Recombination rate for non-hotspot variants. If not specified, defaults to 1 / (100*M).
        Returns:
            R [1D array] Array of length M specifying recombination rates.
        '''
        if N_blocks is None:
            N_blocks = np.max([2, M // 10])
        N_hotspots = N_blocks - 1
        if block_R is None:
            block_R = (1 / M) / 100
        if hotspot_lambda is None:
            hotspot_lambda = M
        
        R = np.full(M, block_R)
        if N_blocks > 1:
            j_hotspots = np.random.choice(M, size=N_hotspots, replace=False)
            R[j_hotspots] = np.random.exponential(scale = 1 / hotspot_lambda, size = N_hotspots)
        return R


    ###################################
    #### Setting object attributes ####
    ###################################
    '''
    For a lot of these methods, if certain main inputs are not provided, the method will pull the input from its object attributes. Then, it also save the resulting output as an object attribute, instead of returning it as an object.
    '''

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
            self._get_G()
            self.get_freq()
            self.X = self.standardize_G(self.G, self.p, self.P)
            if get_GRM:
                self.GRM = self.get_GRM(self.G, self.p, self.P)
        if K is not None:
            self.K = K
        if get_corr_matrix:
            self.corr_matrix = self.get_corr_matrix()

    def _get_G(self, H: np.ndarray = None) -> np.ndarray:
        '''
        Collapses haplotypes into genotypes. If the first parameter is provided, output is stored as an attribute, instead of being returned.

        Parameters:
            H (3D array): N*M*P array of alleles
        Returns:
            G (2D array): N*M array of genotypes. First dimension is individuals, second dimension is variants. Each element is an integer ranging from 0 to P (the ploidy).
        '''
        replace = False
        if H is None:
            replace = True
            H = self.H
        G = H.sum(axis=2)
        if replace:
            self.G = G
        else:
            return G

    def get_freq(self, G: np.ndarray = None, P: int = None) -> np.ndarray:
        '''
        Computes array of allele frequencies for current genotypes. If the first parameter is provided, output is stored as an attribute, instead of being returned.

        Parameters:
            G (2D array): N*M array of genotypes. First dimension is individuals, second dimension is variants. Each element is an integer ranging from 0 to P (the ploidy).
            P (int): Ploidy of genotype matrix. If not specified, assumes to be same as object it was called from.
        Returns:
            p (1D array): Array of allele frequencies.
        '''
        replace = False
        if G is None:
            replace = True
            G = self.G
        if P is None:
            P = self.P
        p = G.mean(axis=0) / P

        if replace:
            self.p = p
        else:
            return p
        
    def center_G(self,G: np.ndarray = None, p: np.ndarray = None, P: int = None) -> np.ndarray:
        '''
        Centers genotype matrix so that the mean of each column is 0 (or approximately)
        Parameters:
            G (2D array): Genotype matrix. If not specified, assumes to be from object it was called from.
            p (1D array): Array of allele frequencies from which to center genotypes. If not specified, assumes to be from object it was called from.
            P (int): Ploidy of genotype matrix. If not specified, assumes to be same as object it was called from.
        Returns:
            G_centered (2D array): Centered genotype matrix
        '''
        if G is None:
            G = self.G
            p = self.p
            P = self.P
        else:
            if p is None:
                p = self.get_freq(G)
            if P is None:
                P = self.P
        G = G - P*p[None,:]
        return G # now centered, so actually X - mu

    def standardize_G(self, G: np.ndarray, p: np.ndarray = None, P: int = None,
                      impute: bool = True, std_method: str = 'observed'):
        '''
        Standardizes genotype matrix so that each column has mean 0 and standard deviation of 1 (or approximately). If the first parameter is provided, output is stored as an attribute, instead of being returned.

        Parameters:
            G (2D array): Genotype matrix. If not specified, assumes to be from object it was called from.
            p (1D array): Array of allele frequencies from which to center genotypes. If not specified, assumes to be from object it was called from.
            P (int): Ploidy of genotype matrix. If not specified, assumes to be same as object it was called from.
            impute (bool): If genotype matrix is a masked array, missing values are filled with the mean genotype value. Default is True.
            std_method (str): Method for calculating genotype standard deviations. If 'observed' (default), then the actual mathematical standard deviation is used. If 'binomial', then the expected standard deviation based on binomial sampling of the allele frequency is used.
        Returns:
            X (2D array): Standardized genotype matrix
        '''
        replace = False
        if G is None:
            replace = True
            G = self.G
            p = self.p
            P = self.P
        else:
            if p is None:
                p = self.get_freq(G)
            if P is None:
                P = self.P
        G = self.center_G(G, p, P)
        
        if np.ma.isMaskedArray(G) and impute:
            G[G.mask] = 0
            G = G.data

        if std_method == 'binomial':
            var_G = P * p * (1 - p) 
        elif std_method == 'observed': # Ensures Var[G] = 1
            var_G = G.var(axis=0)
        
        # replaces monomorph variances with 1 so no divide by 0 error
        var_G[var_G == 0] = 1
        
        X = G / np.sqrt(var_G)[None,:]

        if replace:
            self.X = X
        else:
            return X
    
    def get_GRM(self, G: np.ndarray = None, p: np.ndarray = None, P: int = None) -> np.ndarray:
        '''
        Computes the genetic relationship matrix (GRM). If the first parameter is provided, output is stored as an attribute, instead of being returned.

        Parameters:
            G (2D array): Genotype matrix. If not specified, assumes to be from object it was called from.
            p (1D array): Array of allele frequencies from which to center genotypes. If not specified, assumes to be from object it was called from.
            P (int): Ploidy of genotype matrix. If not specified, assumes to be same as object it was called from.
        Returns:
            GRM (2D array): An N*N genetic relationship matrix. Each element is the mean covariance of standardized genotypes across all variants.
        '''
        standardize = True
        replace = False
        if G is None:
            replace = True
            if hasattr(self, 'X'):
                X = self.X
                standardize = False
            else:
                G = self.G
                p = self.p
                P = self.P
        else:
            if p is None:
                p = self.get_freq(G)
            if P is None:
                P = self.P
        # get standardized genotype matrix (forced to have Var=1 per column)
        if standardize:
            X = self.standardize_G(G,p,P, std_method='observed')
        M = X.shape[1]
        # computes GRM
        GRM = (X @ X.T) / M

        if replace:
            self.GRM = GRM
        else:
            return GRM
    
    def get_neighbor_matrix(self, positions: np.ndarray = None, LDwindow: float = None) -> sparse.coo_matrix:
        '''
        Gets boolean sparse matrix of variants within the specified LD window distance. If the first parameter is provided, output is stored as an attribute, instead of being returned.

        Parameters:
            positions (array): Array of length M containing physical positions of variants. Positions must already be in ascending order. If not provided, defaults to object's position map.
            LDwindow (float): Maximum distance between variants to be considered neighbors. In the same units as that provided in `positions`. If not provided, defaults to infinite maximum distance.
        Returns:
            neighbor_matrix (sparse 2D matrix (COO)): An M*M scipy sparse matrix with boolean values, where a 1 at (i,j) indicates that variant i and j are within `LDwindow` of each other, and 0 if not. Returned in COO sparse format.
        '''
        # Uses object's position map if not provided
        replace = False
        if positions is None:
            replace = True
            positions = self.BPs
        if LDwindow is None:
            LDwindow = positions[-1] - positions[0]
        # Initialize lists to store row and column indices of non-zero entries
        rows = []
        cols = []
        # Iterate through each position
        for i in range(len(positions)):
            # Find indices of positions within LDwindow of positions[i]
            start_idx = np.searchsorted(positions, positions[i] - LDwindow, side='left')
            end_idx = np.searchsorted(positions, positions[i] + LDwindow, side='right')
            # Add the indices to the rows and cols lists
            for j in range(start_idx, end_idx):
                rows.append(i)
                cols.append(j)
        # Create a sparse matrix in COO format
        data = np.ones(len(rows), dtype=bool)  # All non-zero entries are True
        neighbor_matrix = sparse.coo_matrix((data, (rows, cols)),
                                          shape=(len(positions), len(positions)))
        if replace:
            self.neighbor_matrix = neighbor_matrix
        else:
            return neighbor_matrix
    
    def get_corr_matrix(self, X: np.ndarray = None,
                        neighbor_matrix: sparse.coo_matrix = None):
        '''
        Computes the correlation between neighboring pairs of variants. The square of this matrix is the LD matrix. If the first parameter is provided, output is stored as an attribute, instead of being returned, and inputs are taken from the object.

        Parameters:
            X (2D array): N*M standardized genotype matrix where for every column, the mean is 0 and the variance is 1. If not provided, defaults to the object's standardized genotype matrix.
            neighbor_matrix (sparse 2D matrix (COO)): An M*M scipy sparse matrix with boolean values, where True indicates that the correlation between variants i and j is to be computed. If not provided, uses object's (pre-computed) neighbor_matrix attribute.
        Returns:
            corr_matrix (sparse 2D matrix (CSR)): M*M correlation matrix between pairs of variants. Returned in CSR sparse format.
        '''
        replace = False
        if X is None:
            replace = True
            X = self.X
            
        if neighbor_matrix is None:
            if hasattr(self, 'neighbor_matrix'):
                neighbor_matrix = self.neighbor_matrix
            else:
                raise Exception('Must have pre-computed neighbor_matrix. Use `Population.get_neighbor_matrix()`.')
        # Lists to store row indices, column indices, and data for the sparse matrix
        rows = []
        cols = []
        data = []
        # Iterate over non-zero entries in neighbors_mat
        for i, j in zip(neighbor_matrix.row, neighbor_matrix.col):
            if i < j:  # Only compute for upper triangle (including diagonal)
                # Compute the dot product for (i, j)
                #corr_value = (X[:, i] * X[:, j]).mean()
                corr_value = X[:, i].dot(X[:, j]) / X.shape[0]
                # Add (i, j) and (j, i) to the sparse matrix
                rows.append(i)
                cols.append(j)
                data.append(corr_value)
                if i != j:  # Avoid duplicating the diagonal
                    rows.append(j)
                    cols.append(i)
                    data.append(corr_value)
        # Add diagonal entries (set to 1)
        M = X.shape[1]
        for i in range(M):
            rows.append(i)
            cols.append(i)
            data.append(1.0)
        # Create the sparse matrix in COO format
        corr_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(M, M))
        
        if replace:
            self.corr_matrix = corr_matrix
        else:
            return corr_matrix

    def get_LD_matrix(self, corr_matrix: sparse.csr_matrix = None):
        '''
        Computes LD matrix by just taking the square of a correlation matrix. If the first parameter is provided, output is stored as an attribute, instead of being returned.

        Parameters:
            corr_matrix (sparse 2D matrix (CSR)): M*M correlation matrix between pairs of variants in CSR format. Can be computed with `get_corr_matrix()`.
        Returns:
            LD_matrix (sparse 2D matrix (CSR)): M*M LD matrix (square of correlation) between pairs of variants in CSR format.
        '''
        replace = False
        if corr_matrix is None:
            replace = True
            corr_matrix = self.corr_matrix
        
        LD_matrix = corr_matrix.multiply(corr_matrix)  # Element-wise square
        if replace:
            self.LD_matrix = LD_matrix
        else:
            return LD_matrix
    
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
        
    def next_generation(self, s: Union[float, np.ndarray] = 0, mu: float = 0):
        '''
        Simulates new generation. Doesn't simulate offspring directly, meaning that future offspring have haplotypes drawn randomly from allele frequencies. Automatically updates object.

        Parameters:
            s (float or 1D array): Selection coefficient, such that an individual with the alternate allele has a (1+s) relative fitness compared to the reference allele. Occurs before mutation. If only a single value is provided, it is treated as the selection coefficient for all variants. Otherwise, must be an array of length M. Default is 0 (no selection).
            mu (float): Mutation rate, such that the probability of any individual allele flipping to its alternate in the next generation is given by mu. Occurs after selection (i.e. mutation occurs in germline of current generation). Default is 0 (no mutations).
        '''
        # assigns same selection coefficient/mutation rate to all variants if only single value specified
        if type(s) == float:
            s = np.full(self.M, s)
        if type(mu) == float:
            mu = np.full(self.M, mu)
        # analytical allele frequency from which next generations' alleles are drawn from
        p = self.p
        # effect of selection
        p = p * (1 + s) / (1 + p*s)
        # effect of mutation (in gametes that lead to new generation, i.e. post-selection)
        p = p*(1-mu) + (1-p)*mu
        # effect of genetic drift
        H = self._generate_unrelated_haplotypes(p)
        self._update_obj(H=H)

    def simulate_generations(self, generations: int, related_offspring: bool = False,
                             mu: float = 0, s: Union[float, np.ndarray] = 0, R: Union[float, np.ndarray] = None,
                             record_history: bool = True):
        '''
        Simulates specified number of generations beyond current generation. Can simulate offspring directly. Automatically updates object. Recombination rates are extracted from object attributes.

        Parameters:
            generations (int): Number of generations to simulate (beyond the current generation).
            related_offspring (bool): Whether the offspring of the next generation should be directly related to parents from previous generation by simulating meiosis and haplotype transfer. Default is False, meaning that future offspring have haplotypes drawn randomly from allele frequencies.
            mu (float): Mutation rate, such that the probability of any individual allele flipping to its alternate in the next generation is given by mu. Occurs after selection (i.e. mutation occurs in germline of current generation). Default is 0 (no mutations).
            s (float or 1D array): Selection coefficient, such that an individual with the alternate allele has a (1+s) relative fitness compared to the reference allele. Occurs before mutation. If only a single value is provided, it is treated as the selection coefficient for all variants. Otherwise, must be an array of length M. Default is 0 (no selection).
            record_history (bool): Determines if allele frequencies at each generation are saved to a matrix belonging to the object. Default is True.
        '''
        # keeps track of allele frequencies over generations if specified
        if record_history:
            # checks if previous generations have already been generated or not
            previous_gens = self.ps.shape[0]
            ps = np.full( (previous_gens + generations, self.M), np.nan)
            ps[0:previous_gens,] = self.ps
        
        # loops through each generation
        for t in range(generations):
            if related_offspring:
                self.generate_offspring()
            else:
                self.next_generation(mu=mu, s=s)
            # records allele frequency
            if record_history:
                ps[previous_gens + t,] = self.p
        self.T_breaks.append(previous_gens + generations)
        # saves allele freqs. at each generation (keeps old generations)
        if record_history:
            self.ps = ps

    def _pair_mates(self) -> np.ndarray:
        '''
        Pairs individuals up monogamously to mate and generate offspring. Population size must be multiple of 2. In the future will allow for assortative mating (phenotypic and genetic). 

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
        # randomly splits up population into maternal (M) and paternal (P) halves 
        # also shuffles their order
        iMs = np.random.choice(self.N, N2, replace=False)
        iPs = np.setdiff1d(np.arange(self.N), iMs)

        return (iMs, iPs)

    def generate_offspring(self, replace: bool = True):
        '''
        Pairs up mates and generates offspring for parents' haplotypes. Only works for diploids. Each pair always has two offspring. Recombination rates are extracted from object attributes.

        Parameters:
            replace (bool): Whether the offspring replace the current generation. Default is True.
        
        Returns:
            H (3D array): N*M*P array of offspring haplotypes. First dimension is individuals, second dimension is variants, and third dimension is haplotype number (related to ploidy). Each element is either a 0 or a 1.
        '''
        # checks ploidy
        if self.P != 2:
            raise Exception('Offspring generation only works for diploids.')
                
        # pairs up mates
        iMs, iPs = self._pair_mates()
        iMs = np.concatenate([iMs, iMs])
        iPs = np.concatenate([iPs, iPs])

        # determines population size of next generation (currently maintains population size)
        N_offspring = self.N
        
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
        parents = np.full((N_offspring,2), -1)
        for i in np.arange(N_offspring):
            iM = iMs[i]
            iP = iPs[i]
            # extract allele from correct haplotype of each parent
            haploM = self.H[iM, np.arange(self.M), haplo_ks[i, :, 0]]
            haploP = self.H[iP, np.arange(self.M), haplo_ks[i, :, 1]]
            haplos = np.stack((haploM, haploP), axis = 1)
            # shuffles haplotypes around
            haplos = haplos[:,np.random.choice(self.P, size=self.P, replace=False)]
            H[i,:,:] = haplos
            # stores parental information
            parents[i,:] = [iM, iP]

        if replace:
            self._update_obj(H=H)
        else:
            return H

    #######################
    #### Visualization ####
    #######################

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
        
        # plotting
        plt.figure(figsize=(8, 5))
        # allele frequency lines
        if not summarize:
            for j in range(ps.shape[1]):
                plt.plot(ts, ps[:, j], label=f'Variant {j_keep[j]}')
        else:
            plt.plot(ts, ps_mean, color='deepskyblue', label = 'Mean', ls='--')
            for j in range(len(quantiles)):
                plt.plot(ts, ps_quantile[j,:], label=f'{quantiles[j]*100}% percentile', color = 'lightskyblue', ls=':')

        # vertical lines denoting simulation batches
        for t in self.T_breaks:
            plt.axvline(t, ls='--', color='black')
        # labels
        plt.xlabel('Generation')
        plt.ylabel('Allele Frequency')
        plt.title('Allele Frequency Trajectories Over Time')
        plt.xlim(ts.min(), ts.max())
        plt.ylim(0, 1)
        if legend:
            plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_LD_matrix(self, LD_matrix: sparse.csr_matrix = None,
                      plot_range: Tuple[int, int] = None, type: str = 'LD'):
        '''
        Plots LD/correlation between variants.

        Parameters:
            LD_matrix (sparse 2D matrix): Either an LD or correlation M*M matrix of variants. If not provided, uses object's LD/correlation matrix.
            plot_range (tuple): A tuple of length 2 containing the range of variant indices to plot. If not provided, plots all variants.
            type (str): Uses color scheme for either 'LD' (default) or 'corr' matrix.
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