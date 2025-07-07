'''
This file contains functions related to statistical genetics.
The `popsim.py` file contains classes that contain wrapper methods that call these functions, providing the class object's attributes as arguments.
Alternatively, these functions can be called directly with the appropriate arguments.
The functions here contain the documentation for the arguments and return values, which is not repeated in the class methods for the most part.
'''

# imports
import numpy as np
from typing import Tuple, Union

##########################
#### Trait generation ####
##########################

def generate_causal_effects(M: int, M_causal: int = None, var_G: float = 1.0, dist: str = 'normal') -> tuple[np.ndarray, np.ndarray]:
    '''
    Generates variant effect sizes for some trait. Default interpretation is per-standardized-allele effect sizes. Non-causal variant effects are set to 0. Causal effects are drawn from a specified distribution (default is Normal).
    Parameters:
        M (int): Total number of variants (causal and non-causal).
        M_causal (int): Number of causal variants (variants with non-zero effect sizes). Default is all variants.
        var_G (float): Total expected variance contributed by per-standardized-allele genetic effects. Default is 1.0.
        dist (str): Distribution to draw causal effects from. Options are:
            - 'normal': Normal distribution (default).
            - 'constant': All effect sizes are the same.
    Returns:
        tuple ((causal_effects, j_causal)):
        Where:
        - causal_effects (1D array): Array of length M containing the effect sizes of all variants.
        - j_causal (1D array): Array of length M_causal containing the variant indices of causal variants.
    '''
    if M_causal is None:
        M_causal = M
    causal_effects = np.zeros(M)
    j_causal = np.random.choice(M, M_causal, replace=False)
    if dist == 'normal':
        causal_effects[j_causal] = np.random.normal(0, np.sqrt(var_G/M_causal), M_causal)
    elif dist == 'constant':
        causal_effects[j_causal] = np.sqrt(var_G/M_causal)
    return (causal_effects, j_causal)

def compute_genetic_value(G: np.ndarray, effects: np.ndarray) -> np.ndarray:
    '''
    Computes the genetic value/score given a (standardized) genotype matrix and (per-standardized-allele) effect sizes.
    Parameters:
        G (2D array): N*M NON-standardized genotype matrix.
        effects (1D array): Array of length M containing causal and non-causal genetic effects.
    Returns:
        y_G (1D array): Array of length N containing genetic component values.
    '''
    # dot product
    y_G = G @ effects
    return y_G

def generate_noise_value(N: int, var_Eps: float = 0.0) -> np.ndarray:
    '''
    Generates noise component of trait drawn randomly from Normal distribution.
    Parameters:
        N (int): Number of individuals to generate noise component for.
        var_Eps (float): Variance of the noise component. Default is 0.
    Returns:
        y_Eps (1D array): Array of length N containing noise component values.
    '''
    y_Eps = np.random.normal(loc=0, scale=np.sqrt(var_Eps), size = N)
    return y_Eps

def get_G_std_for_effects(G: np.ndarray, p_min: float = 0.05, P: int = None, ) -> np.ndarray:
    '''
    Computes the standard deviation of each column in the genotype matrix used for converting between per-allele and per-standardized-allele effects. Has safe handling for monomorphic alleles.
    Parameters:
        G (2D array): N*M NON-standardized genotype matrix.
        p_min (float): Minimum allele frequency to consider an allele polymorphic. Default is 0.05.
        P (int): Ploidy level. If not specified, it is estimated as the maximum value in G.
    Returns:
        G_std (1D array): Array of length M containing the standard deviations of each column in G.
    '''
    # gets observed standard deviation of the genotype matrix
    G_std = G.std(axis=0)
    # if an allele is monomorphic, it uses the standard deviation one would get for a binomial variable with p = p_min
    if P is None:
        P = G.max() # estimates ploidy to be the maximum value in G, which may not always be true
    G_std[G_std == 0] = np.sqrt(P * p_min * (1 - p_min)) # sets standard deviation for monomorphic alleles
    return G_std

def get_standardized_effects(effects: np.ndarray, G_std: np.ndarray, std2allelic: bool = True) -> np.ndarray:
        '''
        Converts between per-allele and per-standardized-allele effects.
        Parameters:
            effects (1D array): M-length array of effects. Can be per-allele or per-standardized-allele.
            G_std (1D array): Array of length M containing genotype standard deviations to scale the effects by. Cannot have zero values. See `get_G_std_for_effects` for how to compute this safely.
            std2allelic (bool): If True (default), converts from per-standardized-allele to per-allele effects. If False, converts from per-allele to per-standardized-allele effects.
        '''
        if std2allelic:
            effects_output = effects / G_std # per-allele effects
        else:
            effects_output = effects * G_std # per-standardized-allele effects
        return effects_output

def run_HE_regression(A: np.ndarray, y:np.ndarray, se: str = 'jackknife') -> Union[float, Tuple[float, float]]:
    '''
    Performs Haseman-Elston regression to estimate heritability.
    Parameters:
        A (2D array): N*N matrix of pairwise genetic relatedness coefficients. Can be ei
        y (1D array): N-length array of trait values. Should be standardized.
    '''
    y = (y - y.mean()) / y.std() # standardizes trait values
    N = A.shape[0]
    numerator = []
    denominator = []
    jk_indices = [[] for _ in range(N)]
    jk=0 # pair index
    for j in range(N):
        for k in range(N):
            if j >= k: # since matrix is symmetrical along diagonal
                continue
            numerator.append( A[j,k] * y[j] * y[k] )
            denominator.append( A[j,k]**2 )
            # keeps track of individual pairs for jackknife
            if se == 'jackknife':
                jk_indices[j].append(jk)
                jk_indices[k].append(jk)
                jk += 1
    # computes h2
    sum_numerator = sum(numerator)
    sum_denominator = sum(denominator)
    h2g_HE = sum_numerator / sum_denominator

    # computes standard error if requested
    if se == 'jackknife':
        h2_jackknife = []
        for j in range(N):
            # compute sum of num/denom for the the pairs containing individual j
            j_num = sum(numerator[i] for i in jk_indices[j])
            j_denom = sum(denominator[i] for i in jk_indices[j])
            # subtracts out individual j's contribution to the numerator and denominator
            num_minus_j = sum_numerator - j_num
            denom_minus_j = sum_denominator - j_denom
            # recomputes h2 without individual j
            h2_jackknife.append(num_minus_j / denom_minus_j)
        # computes standard error of h2 using jackknife
        h2_jackknife = np.array(h2_jackknife)
        h2_jackknife_mean = h2_jackknife.mean()
        h2g_HE_se = np.sqrt( ((N-1) / N) * sum((h2_jackknife - h2_jackknife_mean)**2))
        return h2g_HE, h2g_HE_se
    else:
        return h2g_HE