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
    Generates 3-dimensional matrix of population haplotypes (allele dosages).
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

########################################
#### Analysis of allele frequencies ####
########################################

def get_fixation_t(ps: np.ndarray = None) -> np.ndarray:
    '''
    For each variant, finds *first* generation (returned as an index) for which the allele frequency is 0 (loss) or 1 (fixation). A value of -1 means the variant never got fixed
    Parameters:
        ps (2D array): T*M matrix (where T is number of generations) containing allele frequencies over time.
    Returns:
        t_fix (1D array): Array of length M with the first generation (as an index) for which the respective allele was lost or fixed. If the allele was not fixed by the most recent simulation, a -1 is returned.
    '''
    # gets mask of whether frequency is 0 or 1
    ps_mask = np.any((ps == 0, ps == 1), axis=0)
    # finds first instance of True for each variant
    t_fix = np.where(ps_mask.any(axis=0), ps_mask.argmax(axis=0), -1)        
    return t_fix

def summarize_ps(ps: np.ndarray = None, quantiles: tuple = (0.25, 0.5, 0.75)) -> tuple[np.ndarray, np.ndarray]:
    '''
    Returns the mean as well as the specified quantiles of variants across each generation.

    Parameters:
        ps (2D array): T*M matrix (where T is number of generations) containing allele frequencies over time.
        quantiles (tuple): List of quantiles (e.g. 0.99) of allele frequencies across variants at each generation to plot. `summarize` must be set to True. Default is median, lower quartile, and upper quartile.
    Returns:
        tuple ((ps_mean, ps_quantile)):
        Where:
        - ps_mean (1D array): Array of length T (where T is the total number of generations) of mean allele frequency at each generation.
        - ps_quantile (2D array): K*T matrix (where K is the number of quantiles specified) of allele frequency for each quantile at each generation.
    '''
    # computes mean allele frequency over time
    ps_mean = ps.mean(axis=1)
    # computes quantiles over time
    ps_quantile = np.quantile(ps, quantiles, axis=1)
    return (ps_mean, ps_quantile)

###########################
#### Handling families ####
###########################

def initialize_relations(N: int, N1: int = None):
    '''
    Initializes a dictionary of relations for a population of N individuals.
    Parameters:
        N (int): Number of individuals in the population.
        N1 (int): Number of individuals in the parental generation. If not specified, defaults to N.
    '''
    if N1 is None:
        N1 = N
    relations = {
        'parents': np.zeros((N, N1), dtype=np.uint8),
        'full_sibs': np.zeros((N, N), dtype=np.uint8),
        'spouses': np.zeros((N, N), dtype=np.uint8),
        'household': np.zeros((N, N), dtype=np.uint8),
        'ped': {}
    }
    return relations

##################################
#### IBD: Identity by Descent ####
##################################

def get_true_IBD1(chr_i: np.ndarray, chr_j: np.ndarray) -> np.ndarray:
    '''
    Computes true IBD1 segments between two haplotypes.
    Parameters:
        chr_i (1D array): Array of length M containing haplotype IDs for haplotype i.
        chr_j (1D array): Array of length M containing haplotype IDs for haplotype j.
    Returns:
        IBD1 (1D boolean array): Array of length M where True indicates that the two haplotypes are IBD1 at that variant.
    '''
    IBD1 = (chr_i == chr_j)
    # the -1 identifier is used to denote unrelated haplotypes
    unrelated_mask = (chr_i == -1) | (chr_j == -1)
    IBD1[unrelated_mask] = False
    return IBD1

def get_true_IBD_tensor(haplos_i:np.ndarray, haplos_j:np.ndarray) -> np.ndarray:
    '''
    Returns a 3D (M*P*P) tensor indicating IBD1 status for each pair of haplotypes between two individuals. Genotypes must be diploid (P=2).
    Parameters:
        haplos_i (2D array): M*2 array containing haplotype IDs for individual i.
        haplos_j (2D array): M*2 array containing haplotype IDs for individual j.
    Returns:
        IBD_tensor (3D boolean array): M*P*P array where each element at (m, p_i, p_j) indicates whether haplotype p_i of individual i and haplotype p_j of individual j are IBD1 at variant m.
    '''
    # extracts relevant genotype dimensions (# of variants and ploidy)
    (M, P) = haplos_i.shape # assumes same dimensions for both individuals
    # checks if diploid
    if P != 2:
        raise ValueError("Genotypes must be diploid to compute IBD2.")
    # makes empty list that is P*P*M
    IBD_tensor = np.zeros((M,P,P), dtype=bool)
    # computes IBD1 status for each pair of haplotypes
    for (i_haplo, j_haplo) in [(0,0), (0,1), (1,0), (1,1)]:
        IBD_hi_hj = get_true_IBD1(haplos_i[:, i_haplo], haplos_j[:, j_haplo])
        IBD_tensor[:, i_haplo, j_haplo] = IBD_hi_hj
    
    return IBD_tensor

def get_true_IBD_arr(haplos_i:np.ndarray, haplos_j:np.ndarray, return_tensor: bool = False) -> np.ndarray:
    '''
    Computes the IBD state (0,1,2) between two individuals based on their haplotype IDs. Genotypes must be diploid (P=2). Assumes that an individuals two haplotypes are different. So, e.g., if comparing (A,A) and (A,B), this is treated as IBD1, not IBD2.
    Parameters:
        haplos_i (2D array): M*2 array containing haplotype IDs for individual i.
        haplos_j (2D array): M*2 array containing haplotype IDs for individual j.
        return_tensor (bool): If True, also returns the full IBD tensor (M*P*P array) indicating IBD1 status for each haplotype pair.
    Returns:
        tuple ((IBD_arr, IBD_tensor)):
        Where:
        - IBD_arr (1D array): Array of length M where each element is 0, 1, or 2 indicating the IBD state between the two individuals at that variant.
        - IBD_tensor (3D array, optional): If return_tensor is True, also returns the full IBD tensor (see get_true_IBD_tensor).
    '''
    
    IBD_tensor = get_true_IBD_tensor(haplos_i, haplos_j)
    (M, P, _) = IBD_tensor.shape
    # sums IBD1 statuses to get IBD2 status
    # by assuming that an individual's two haplotypes are different, it means IBD2 state can only occur when two distinct pairs of haplotype indices are IBD1 (e.g. (0,1) and (1,0), or (0,0) and (1,1), NOT (0,0) and (0,1))
    IBD_arr = np.zeros(M, dtype=np.uint8)
    # IBD1 status is first given to any variant where any haplotype pair is IBD1
    IBD_arr[np.any(IBD_tensor, axis=(1,2))] = 1
    # IBD2 status is then given to any variant where either (0,0) and (1,1) are both IBD1, or (0,1) and (1,0) are both IBD1
    IBD2_mask = ( (IBD_tensor[:,0,0] & IBD_tensor[:,1,1]) |
                  (IBD_tensor[:,0,1] & IBD_tensor[:,1,0]) )
    IBD_arr[IBD2_mask] = 2

    if return_tensor:
        return (IBD_arr, IBD_tensor)
    else:
        return (IBD_arr)
    
def get_coeff_kinship(haplos_i:np.ndarray, haplos_j:np.ndarray, return_arr: bool = False) -> float:
    '''
    Computes the coefficient of kinship between two individuals based on their haplotype IDs.
    Parameters:
        haplos_i (2D array): M*2 array containing haplotype IDs for individual i.
        haplos_j (2D array): M*2 array containing haplotype IDs for individual j.
        return_arr (bool): If True, also returns an array with the coefficient of kinship between the two individuals at each variant.
    Returns:
        tuple ((coeff_kinship, coeff_kinship_arr)):
        Where:
        - coeff_kinship (float): Coefficient of kinship between the two individuals.
        - coeff_kinship_arr (1D array, optional): If return_arr is True, also returns an array with the coefficient of kinship between the two individuals at each variant.
    '''
    IBD_tensor = get_true_IBD_tensor(haplos_i, haplos_j).astype(np.uint8)
    P = IBD_tensor.shape[1]
    # sums IBD1 statuses across all haplotype pairs
    total_IBD1_arr = IBD_tensor.sum(axis=(1,2))
    # computes kinship coefficient
    coeff_kinship_arr = total_IBD1_arr / (P * P)
    coeff_kinship = coeff_kinship_arr.mean()
    if return_arr:
        return (coeff_kinship, coeff_kinship_arr)
    else:
        return coeff_kinship

def get_coeff_inbreeding(haplos_i:np.ndarray, return_arr: bool = False) -> float:
    '''
    Computes the coefficient of inbreeding for an individual based on their IBD tensor.
    Parameters:
        haplos_i (2D array): M*2 array containing haplotype IDs for individual i.
        return_arr (bool): If True, also returns an array with the coefficient of inbreeding at each variant.
    Returns:
        tuple ((coeff_inbreeding, coeff_inbreeding_arr)):
        Where:
        - coeff_inbreeding (float): Coefficient of inbreeding for the individual.
        - coeff_inbreeding_arr (1D array, optional): If return_arr is True, also returns an array with the coefficient of inbreeding (0 or 1) at each variant.
    '''
    IBD_tensor = get_true_IBD_tensor(haplos_i, haplos_i).astype(np.uint8)
    # inbreeding coefficient is the IBD1 status between the two distinct haplotypes
    coeff_inbreeding_arr = IBD_tensor[:,0,1] # guaranteed to be symmetric, so either off-diagonal works
    coeff_inbreeding = coeff_inbreeding_arr.mean()
    if return_arr:
        return (coeff_inbreeding, coeff_inbreeding_arr)
    else:
        return coeff_inbreeding
    
def get_coeff_relatedness(haplos_i:np.ndarray, haplos_j:np.ndarray, return_arr: bool = False) -> float:
    '''
    Returns the coefficient of relatedness between two individuals based on their haplotype IDs. Simply twice the coefficient of kinship, see get_coeff_kinship().
    Parameters:
        haplos_i (2D array): M*2 array containing haplotype IDs for individual i.
        haplos_j (2D array): M*2 array containing haplotype IDs for individual j.
        return_arr (bool): If True, also returns an array with the coefficient of relatedness between the two individuals at each variant.
    Returns:
        tuple ((coeff_relatedness, coeff_relatedness_arr)):
        Where:
        - coeff_relatedness (float): Coefficient of relatedness between the two individuals.
        - coeff_relatedness_arr (1D array, optional): If return_arr is True, also returns an array with the coefficient of relatedness between the two individuals at each variant.
    '''
    (coeff_kinship, coeff_kinship_arr) = get_coeff_kinship(haplos_i, haplos_j, return_arr=True)
    coeff_relatedness = 2 * coeff_kinship
    if return_arr:
        coeff_relatedness_arr = 2 * coeff_kinship_arr
        return (coeff_relatedness, coeff_relatedness_arr)
    else:
        return coeff_relatedness
    

def compute_K_IBD(Haplos: np.ndarray, standardize: bool = False) -> np.ndarray:
    '''
    Computes the kinship matrix based on true IBD between all pairs of individuals in the population.
    Parameters:
        Haplos (3D array): N*M*P array of haplotype IDs. First dimension is individuals, second dimension is variants, third dimension is haplotype number (related to ploidy).
        standardize (bool): If True, standardizes the kinship matrix according to [Young et al. 2018 NatGen]. The mean value in the matrix becomes 0. Default is False.
    Returns:
        K_IBD (2D array): N*N kinship matrix based on true IBD between all pairs of individuals in the population. Element (i,j) is the coefficient of *relatedness* (twice that of kinship) between individuals i and j.
    '''
    N = Haplos.shape[0]
    K_IBD = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            coeff_relatedness = get_coeff_relatedness(Haplos[i, :, :], Haplos[j, :, :])
            K_IBD[i, j] = coeff_relatedness
            K_IBD[j, i] = coeff_relatedness
    if standardize:
        K0 = np.mean(K_IBD) # mean relatedness
        K_IBD = (K_IBD - K0) / (1 - K0)
    return K_IBD