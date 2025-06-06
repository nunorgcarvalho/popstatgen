'''
This file contains functions related to population genetics.
The `popsim.py` file contains classes that contain wrapper methods that call these functions, providing the class object's attributes as arguments.
Alternatively, these functions can be called directly with the appropriate arguments.
The functions here contain the documentation for the arguments and return values, which is not repeated in the class methods for the most part.
'''

# imports
import numpy as np
from scipy import sparse

################################################
#### Genotype handling and basic statistics ####
################################################
def make_G(H: np.ndarray) -> np.ndarray:
    '''
    Collapses haplotypes into genotypes.
    Parameters:
        H (3D array): N*M*P array of alleles.
    Returns:
        G (2D array): N*M array of genotypes. First dimension is individuals, second dimension is variants. Each element is an integer ranging from 0 to P (the ploidy).
    '''
    G = H.sum(axis=2).astype(np.uint8)
    return G

def compute_freqs(G: np.ndarray, P: int) -> np.ndarray:
    '''
    Computes allele frequencies in genotypes.
    Parameters:
        G (2D array): N*M matrix of genotypes. First dimension is individuals, second dimension is variants. Each element is an integer ranging from 0 to P (the ploidy).
        P (int): Ploidy of genotype matrix.
    Returns:
        p (1D array): Array of length M containing allele frequencies.
    '''
    p = G.mean(axis=0) / P
    return p

def center_G(G: np.ndarray, p: np.ndarray, P: int) -> np.ndarray:
    '''
    Centers genotype matrix so that the mean of each column is 0 (or approximately).
    Parameters:
        G (2D array): N*M matrix of genotypes. First dimension is individuals, second dimension is variants. Each element is an integer ranging from 0 to P (the ploidy).
        p (1D array): Array of length M containing allele frequencies from which to center genotypes.
        P (int): Ploidy of genotype matrix.
    Returns:
        G_centered (2D array): Centered genotype matrix.
    '''
    G_centered = G - P*p[None,:]
    return G_centered

def standardize_G(G: np.ndarray, p: np.ndarray, P: int,
                      impute: bool = True, std_method: str = 'observed'):
    '''
    Standardizes genotype matrix so that each column has mean 0 and standard deviation of 1 (or approximately).
    Parameters:
        G (2D array): *M matrix of genotypes. First dimension is individuals, second dimension is variants. Each element is an integer ranging from 0 to P (the ploidy).
        p (1D array): Array of length M containing allele frequencies from which to center genotypes. Also used to scale genotypes.
        P (int): Ploidy of genotype matrix.
        impute (bool): If genotype matrix is a masked array, missing values are filled with the mean genotype value. Default is True.
        std_method (str): Method for calculating genotype standard deviations. If 'observed' (default), then the actual mathematical standard deviation is used. If 'binomial', then the expected standard deviation based on binomial sampling of the allele frequency is used.
    Returns:
        X (2D array): N*M standardized genotype matrix.
    '''
    # centers genotype matrix
    G = center_G(G, p, P)
    # imputes missing values if specified
    if np.ma.isMaskedArray(G) and impute:
        G[G.mask] = 0
        G = G.data
    # determines standard deviation of genotypes
    if std_method == 'binomial':
        var_G = P * p * (1 - p) 
    elif std_method == 'observed': # Ensures Var[G] = 1
        var_G = G.var(axis=0)
    # replaces monomorph variances with 1 so no divide by 0 error
    var_G[var_G == 0] = 1
    # standardizes genotype matrix
    X = G / np.sqrt(var_G)[None,:]
    return X

def compute_GRM(X: np.ndarray) -> np.ndarray:
    '''
    Computes the genetic relationship matrix (GRM) from a standardized genotype matrix (X).
    Parameters:
        X (2D array): N*M standardized genotype matrix.
    Returns:
        GRM (2D array): An N*N genetic relationship matrix. Each element is the mean covariance of standardized genotypes across all variants for the two individuals.
    '''
    M = X.shape[1]
    # computes GRM
    GRM = (X @ X.T) / M
    return GRM

#####################################
#### Linkage Disequilibrium (LD) ####
#####################################
def make_neighbor_matrix(positions: np.ndarray, LDwindow: float = None) -> sparse.coo_matrix:
    '''
    Makes boolean sparse matrix of variants within the specified LD window distance.
    Parameters:
        positions (array): Array of length M containing physical positions of variants. Positions must already be in ascending order.
        LDwindow (float): Maximum distance between variants to be considered neighbors. In the same units as that provided in `positions`. If not provided, defaults to infinite maximum distance.
    Returns:
        neighbor_matrix (sparse 2D matrix (COO)): An M*M scipy sparse matrix with boolean values, where a 1 at (i,j) indicates that variant i and j are within `LDwindow` of each other, and 0 if not. Returned in COO sparse format.
    '''
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
    return neighbor_matrix

def compute_corr_matrix(X: np.ndarray, neighbor_matrix: sparse.coo_matrix):
    '''
    Computes the correlation between neighboring pairs of variants. The square of this matrix is the LD matrix.
    Parameters:
        X (2D array): N*M standardized genotype matrix where for every column, the mean is 0 and the variance is 1.
        neighbor_matrix (sparse 2D matrix (COO)): An M*M scipy sparse matrix with boolean values, where True indicates that the correlation between variants i and j is to be computed.
    Returns:
        corr_matrix (sparse 2D matrix (CSR)): M*M correlation matrix between pairs of variants. Returned in CSR sparse format.
    '''
    N, M = X.shape
    # Lists to store row indices, column indices, and data for the sparse matrix
    rows = []
    cols = []
    data = []
    # Iterate over non-zero entries in neighbors_mat
    for i, j in zip(neighbor_matrix.row, neighbor_matrix.col):
        if i < j:  # Only compute for upper triangle (including diagonal)
            # Compute the dot product for (i, j)
            corr_value = X[:, i].dot(X[:, j]) / N
            # Add (i, j) and (j, i) to the sparse matrix
            rows.append(i)
            cols.append(j)
            data.append(corr_value)
            if i != j:  # Avoid duplicating the diagonal
                rows.append(j)
                cols.append(i)
                data.append(corr_value)
    # Add diagonal entries (set to 1)
    for i in range(M):
        rows.append(i)
        cols.append(i)
        data.append(1.0)
    # Create the sparse matrix in COO format
    corr_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(M, M))
    return corr_matrix

def compute_LD_matrix(corr_matrix):
        '''
        Computes LD matrix by just taking the square of a correlation matrix.
        Parameters:
            corr_matrix (sparse 2D matrix (CSR)): M*M correlation matrix between pairs of variants in CSR format. Can be computed with `get_corr_matrix()`.
        Returns:
            LD_matrix (sparse 2D matrix (CSR)): M*M LD matrix (square of correlation) between pairs of variants in CSR format.
        '''
        LD_matrix = corr_matrix.multiply(corr_matrix)  # Element-wise square
        return LD_matrix

#############################
#### Forward simulations ####
#############################

def draw_binom_haplos(p: np.ndarray, N: int, P: int = 2) -> np.ndarray:
    '''
    Generates 3-dimensional matrix of population haplotypes.
    Parameters:
        p (1D array): Array of length M containing allele frequencies from which haplotypes are drawn.
        N (int): Number of individuals in the population.
        P (int): Ploidy of the population. Default is 2 (diploid).
    Returns:
        H (3D array): N*M*P array of alleles. First dimension is individuals, second dimension is variants, third dimension is haplotype number (related to ploidy). Each element is either a 0 or a 1.
    '''
    M = p.shape[0]
    p = p.reshape(1, M, 1)
    H = np.random.binomial( 1, p = p, size = (N, M, P)).astype(np.uint8)
    return H

########################################
#### Simulating realistic genotypes ####
########################################
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
        R (1D array) Array of length M specifying recombination rates.
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

def draw_p_init(M, method: str, params: list) -> np.ndarray:
    '''
    Returns array of initial allele frequencies to simulate genotypes from.
    Parameters:
        method (str): Method of randomly drawing allele frequencies. Options:
            uniform: 'Draws from uniform distribution with lower and upper bounds specified by `params`.'
        params (list): Method-specific parameter values.
    Returns:
        p_init (1D array): Array of length M containing allele frequencies.
    '''
    # uniform sampling is default (and only currently supported method)
    if method == 'uniform':
        p_init = np.random.uniform(params[0], params[1], M)
        return p_init
    else:
        return np.full(M, np.nan)

