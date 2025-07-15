'''
This file contains functions related to statistical genetics.
The `popsim.py` file contains classes that contain wrapper methods that call these functions, providing the class object's attributes as arguments.
Alternatively, these functions can be called directly with the appropriate arguments.
The functions here contain the documentation for the arguments and return values, which is not repeated in the class methods for the most part.
'''

# imports
import numpy as np
import matplotlib.pyplot as plt
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

def get_random_effects(Zs: list[np.ndarray], variances: list[float], names: list[str] = None) -> dict:
    '''
    Computes random effects of clusters for a mixed model.
    Parameters:
        Zs (list): List of design matrices for random effects. Each matrix should be N*N_i, where N is the number of individuals and N_i is the number of clusters for that random effect.
        variances (list): List of variances for each random effect. Random effects are drawn from a normal distribution with mean 0 and variance given by the corresponding element in this list.
        names (list): List of names for each random effect. Default is None, meaning names are not used (instead the index is used).
    Returns:
        random_effects (dict): Dictionary of random effects, with names as keys and (Z, var) tuples as values.
    '''
    # names components if not given
    if names is None:
        names = [f"RE_{i}" for i in range(len(Zs))]
    # generates random effects for each cluster for each component
    us = []
    for i in range(len(Zs)):
        u = np.random.normal(loc=0, scale=np.sqrt(variances[i]), size=Zs[i].shape[1])
        us.append(u)
    # creates a dictionary of random effects
    random_effects = {
        'name': names,
        'var': variances,
        'Z': Zs,
        'u': us
    }
    return random_effects

#################################
#### Heritability Estimation ####
#################################

#### Haseman-Elston Methods ####
def run_HE_regression(A: np.ndarray, y:np.ndarray, se: str = 'jackknife') -> Union[float, Tuple[float, float]]:
    '''
    Performs Haseman-Elston regression to estimate heritability.
    Parameters:
        A (2D array): N*N matrix of pairwise genetic relatedness coefficients. Can be ei
        y (1D array): N-length array of trait values. Should be standardized.
    Returns:
        h2g_HE (float): Estimated heritability from Haseman-Elston regression.
        h2g_HE_se (float): Standard error of the heritability estimate if `se` is 'jackknife'. Otherwise, this value is not returned.
    '''
    # implementation based on Slide 66 of Week 8 of Alkes Price's EPI 511 course lectures
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
    
def plot_HE_regression(A: np.ndarray, y: np.ndarray, bins: int = 5) -> None:
    '''
    Plots the Haseman-Elston regression line and the data points.
    Parameters:
        A (2D array): N*N matrix of pairwise genetic relatedness coefficients.
        y (1D array): N-length array of trait values. Should be standardized.
        bins (int): Number of bins to use for grouping relatedness coefficients. 95% confidence interval for each bin's mean is shown. Default is 5 bins. If 0, no binning is performed.
    '''
    relatedness = []
    sq_diff = []

    y = (y - y.mean()) / y.std() # standardizes trait values
    N = A.shape[0]
    # computes the squared difference between pairwise trait values
    # and the pairwise genetic relatedness coefficients
    for j in range(N):
        for k in range(N):
            if j >= k: # since matrix is symmetrical along diagonal
                continue
            relatedness.append( A[j,k] )
            sq_diff.append( (y[j] - y[k])**2 )
    
    relatedness = np.array(relatedness)
    sq_diff = np.array(sq_diff)
    # binning
    if bins > 1:
        # defines bin edges
        bin_edges = np.linspace(np.min(relatedness), np.max(relatedness), bins + 1)
        # assigns each x_val to a bin index
        x_bin_i = np.digitize(relatedness, bin_edges) - 1 # -1 to make indices 0-based
        x_bin_i[x_bin_i == bins] = bins - 1  # ensure the last bin is inclusive
        # computes mean x and y for each bin
        x_val = np.array([np.mean(relatedness[x_bin_i == i]) for i in range(bins)])
        y_val = np.array([np.mean(sq_diff[x_bin_i == i]) for i in range(bins)])
        # computes 95CI around each point
        y_val_se = np.array([np.std(sq_diff[x_bin_i == i]) / np.sqrt(np.sum(x_bin_i == i)) for i in range(bins)])
    else:
        x_val = relatedness
        y_val = sq_diff

    # plotting
    plt.figure(figsize=(8, 5))
    plt.scatter(x_val, y_val, alpha=0.5)
    if bins > 1:
        plt.errorbar(x_val, y_val, yerr=1.96*y_val_se, fmt='o', color='black', capsize=5, label='95% CI')
    plt.xlabel('Genetic relatedness coefficient')
    plt.ylabel('Squared difference in trait values')
    plt.title('Haseman-Elston Regression')
    plt.show()

#### REstricted Maximum Likelihood (REML) Methods ####

def run_REML(y: np.ndarray, Bs: list[np.ndarray], Zs: list[np.ndarray] = [None], X: np.ndarray = None, init: list[float] = None,
             method: str = 'EM', tol: float = 1e-5, max_iter: int = 1000, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    '''
    Runs REML for estimating variance components while accounting for fixed effects. Different algorithms are available.
    Parameters:
        y (1D array): N-length array of trait values.
        Bs (list of 2D arrays): List containing M random effects covariance matrices that are N_i*N_i. Don't include residual matrix.
        Zs (list of 2D arrays): List containing M random effects design matrices that are N*N_i. Don't include residual matrix. If an element is None, it assumes the identity matrix, but requires the the corresponding B matrix to be N*N.
        X (2D array): N*K design matrix of fixed effects covariates. If None, doesn't incorporate fixed effects.
        init (list): M-length list with initial guess for the variance components. Default is random initialization.
        method (str): Method to use for REML estimation. Options are:
            - 'EM': Expectation-Maximization algorithm (default). Slow but stable.
            - 'NR': Newton-Raphson algorithm. Faster than EM.
            - 'FS': Fisher Scoring algorithm (similar to NR). Even faster than NR, but perhaps less stable.
            - 'AI': Average Information algorithm (similar to NR). Even faster than FS and more stable. Used by GCTA.
        tol (float): Tolerance for convergence. Default is 1e-5.
        max_iter (int): Maximum number of iterations. Default is 1000.
        verbose (bool): If True, prints convergence information per iteration. Default is True.
    Returns:
        output (dict): Dictionary containing the following keys:
            - 'var_components': Estimated variance components for each random effect, including residual.
            - 'var_components_se': Standard errors of the variance components.
            - 'var_components_vcov': Variance-covariance matrix of the variance components
            - 'fixed_effects': Estimated fixed effects coefficients (beta). If X is None, this is None.
            - 'fixed_effects_se': Standard errors of the fixed effects coefficients. If X is None, this is None.
            - 'fixed_effects_vcov': Variance-covariance matrix of the fixed effects. If X is None, this is None.
            - 'log_likelihood': Log-likelihood of the model.
    '''
    N = y.shape[0] # number of individuals
    y = (y - y.mean()) / y.std()  # standardizes y to have mean 0 and variance 1

    # adds residual matrix to Zs and Bs and initial guess
    Zs.append(np.identity(N))  # residual design matrix (identity)
    Bs.append(np.identity(N)) # residual covariance matrix (identity)
    # fills out Z matrices if None
    for i in range(len(Zs)):
        if Zs[i] is None:
            Zs[i] = np.identity(N)
            if Bs[i].shape[0] != N or Bs[i].shape[1] != N:
                raise ValueError(f"Zs[{i}] is None, but the corresponding B matrix is not a square matrix of size {N}.")
    # randomly generates initial guesses for variance components if not provided
    if init is None:
        init = np.random.uniform(0, 1, len(Bs))
        init = init / np.sum(init)  # normalizes to sum to 1
    else:
        init_resid = y.var() - np.sum(init)
        if init_resid < 0:
            raise ValueError("Initial guesses for variance components exceed total variance of y.")
        init.append(init_resid)  # initial guess for residual variance component

    # gets variance matrices
    M = len(Zs)
    Vs_i = [Zs[i] @ Bs[i] @ Zs[i].T for i in range(M)]

    if method == 'EM':
        N_i = [Z.shape[1] for Z in Zs] # number of clusters in each random effect
        vars_i = _run_REML_EM(y, Vs_i, N_i, X, init, tol, max_iter, verbose)
    elif method in ['NR', 'FS', 'AI']:
        vars_i = _run_REML_quad(y, Vs_i, X, init, method=method, tol=tol, max_iter=max_iter, verbose=verbose)
    else:
        raise ValueError(f"Method '{method}' is not implemented. See documentation.")

    # final V matrix after convergence
    (V, V_inv) = _get_V_components(Vs_i, vars_i)  # computes V, and V_inv

    # gets variance-covariance matrix for variance components
    P = _get_P_matrix(V_inv, X)  # computes P matrix
    F = _get_F_matrix(P, Vs_i)  # computes Fisher's Information Matrix (expected Hessian)
    F_inv = np.linalg.pinv(F)  # inverse of Fisher's Information Matrix
    # computes standard errors of variance components
    vars_i_se = np.sqrt(np.diag(F_inv))  # inverse

    # fixed effect estimation
    if X is None:
        beta = None
        beta_vcov = None
        beta_se = None
        XB = np.zeros(N)
    else:
        beta = np.linalg.inv(X.T @ V_inv @ X) @ (X.T @ V_inv @ y)
        XB = X @ beta
        # gets variance-covariance matrix for fixed effects
        beta_vcov = np.linalg.inv(X.T @ V_inv @ X) # Equation 27.23
        beta_se = np.sqrt(np.diag(beta_vcov))  # standard errors of fixed effects
    

    # log-likelihood calculation
    logdetV = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(V)))) # for numerical stability
    LL = -0.5*N*np.log(2 * np.pi) - 0.5*logdetV - 0.5*(y - XB).T @ V_inv @ (y - XB)

    # returns
    output = {
        'var_components': vars_i,
        'var_components_se': vars_i_se,
        'var_components_vcov': F_inv,
        'fixed_effects': beta,
        'fixed_effects_se': beta_se,
        'fixed_effects_vcov': beta_vcov,
        'log_likelihood': LL.item(),  # .item() to convert single-element array to float
    }
    return output

def _get_V_components(Vs_i: list[np.ndarray], vars_i: np.ndarray,
                      jitter = 1e-6) -> np.ndarray:
    '''
    Computes the total variance matrix V from the random effects design matrices and covariance matrices.
    Parameters:
        Vs_i (list of 2D arrays): List of M category-specific variance N*N matrices.
        vars_i (1D array): Estimated variance components for each random effect.
        jitter (float): Small value added to diagonal of variance matrices to ensure positive definiteness. Default is 1e-6.
    Returns:
        tuple (V, V_inv):
        Where:
        - V (2D array): Total variance matrix.
        - V_inv (2D array): Inverse of the total variance matrix.
    '''
    M = len(Vs_i)
    V = sum(vars_i[i] * Vs_i[i] for i in range(M))  # total variance matrix
    if jitter > 0:
        V += np.eye(V.shape[0]) * jitter # adds jitter to diagonal for numerical stability
    V_inv = np.linalg.inv(V)
    return (V, V_inv)

def _get_P_matrix(V_inv: np.ndarray, X: np.ndarray = None) -> np.ndarray:
    '''
    Computes the P matrix used in REML estimation.
    Parameters:
        V_inv (2D array): Inverse of the total variance N*N matrix.
        X (2D array): N*K design matrix of fixed effects covariates. If None, P is just V_inv.
    Returns:
        P (2D array): The N*N P matrix used in REML estimation.
    '''
    if X is None:
        P = V_inv
    else:
        P = V_inv - V_inv @ X @ np.linalg.inv(X.T @ V_inv @ X) @ X.T @ V_inv
    return P

def _get_F_matrix(P: np.ndarray, Vs_i: list[np.ndarray]) -> np.ndarray:
    '''
    Computes Fisher's Information Matrix used in REML estimation. Its inverse is also used to compute the standard errors of the variance component estimates. It is the information matrix used under the Fisher Scoring REML method.
    Parameters:
        P (2D array): The N*N P matrix used in REML estimation.
        Vs_i (list of 2D arrays): List of M category-specific variance N*N matrices.
    Returns:
        F (2D array): The N*N F matrix used in REML estimation.
    '''
    M = len(Vs_i)
    F = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
                if i > j:
                    continue # skips double computation
                # Equation 27.35b of textbook
                F_ij = 0.5 * np.trace(P @ Vs_i[i] @ P @ Vs_i[j])
                F[i, j] = F_ij 
                F[j, i] = F_ij # symmetric matrix
    return F

def _get_AI_matrix(P: np.ndarray, Vs_i: list[np.ndarray], y: np.ndarray) -> np.ndarray:
    '''
    Computes the AI matrix used in REML estimation. It is the information matrix used under the AI REML method.
    Parameters:
        P (2D array): The N*N P matrix used in REML estimation.
        Vs_i (list of 2D arrays): List of M category-specific variance N*N matrices.
        y (1D array): N-length array of trait values.
    Returns:
        AI (2D array): The N*N AI matrix used in REML estimation.
    '''
    M = len(Vs_i)
    AI = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            if i > j:
                continue # skips double computation
            # Equation 27.34 of textbook (second term, but halved)
            AI_ij = 0.5 * y.T @ P @ Vs_i[i] @ P @ Vs_i[j] @ P @ y
            AI[i, j] = AI_ij 
            AI[j, i] = AI_ij # symmetric matrix
    return AI

def _run_REML_EM(y: np.ndarray, Vs_i: list[np.ndarray], N_i: list[int] = [None], X: np.ndarray = None, init: list[float] = None,
                 tol: float = 1e-5, max_iter: int = 1000, verbose: bool = True) -> np.ndarray:
    '''
    Runs REML using the Expectation-Maximization (EM) algorithm to estimate variance components. See `run_REML` for parameter descriptions. Returns variane components.
    '''
    # Based on pg 797 of Bruce Walsh and Michael Lynch's "Genetics and Analysis of Quantitative Traits" (1998)
    
    M = len(Vs_i) # number of random effects (technically M+1, since we added the residual matrix)
    vars_i = np.array(init) # initial variance components

    for iter in range(max_iter):
        # V matrix
        (_, V_inv) = _get_V_components(Vs_i, vars_i)  # computes V, and V_inv
        # P matrix
        P = _get_P_matrix(V_inv, X)  # computes P matrix

        offsets = np.zeros(M) # offsets for parameter estimate
        for i in range(M):
            var_i = vars_i[i] # variance component            
            V_i = Vs_i[i] # category-specific variance matrix

            # E-step: compute expected values of random effects given current estimates
            # Equation 27.37a of textbook (modified)
            offset = var_i**2 * ( y.T @ P @ V_i @ P @ y - np.trace(P @ V_i)) / N_i[i]
            offsets[i] = offset

            # M-step: update variance components
            # Equation 27.36a of textbook (modified)
            vars_i[i] = var_i + offset
            vars_i = np.maximum(vars_i, 1e-6) # ensures variance components are positive

            if verbose and i < M - 1:  # don't print for residual variance component
                print(f"Iteration {iter+1}, Random Effect {i+1}: Updated variance component = {vars_i[i]:.6f}, Offset = {offset:.6f}")
        
        # check convergence
        if np.max(np.abs(offsets)) < tol:
            print(f"Converged after {iter+1} iterations.")
            break
        if iter == max_iter - 1:
            print(f"Reached maximum iterations ({max_iter}) without convergence.")
            break

    return vars_i

def _run_REML_quad(y: np.ndarray, Vs_i: list[np.ndarray], X: np.ndarray = None, init: list[float] = None,
                   method: str = 'FS', tol: float = 1e-5, max_iter: int = 1000, verbose: bool = True) -> np.ndarray:
            
    '''
    Runs REML using quadratic iterative methods to estimate variance components: Newton-Raphson, Fisher-Scoring, Average Information. See `run_REML` for parameter descriptions. Returns variane components.
    '''
    # Based on pg 794 of Bruce Walsh and Michael Lynch's "Genetics and Analysis of Quantitative Traits" (1998)

    M = len(Vs_i) # number of random effects (technically M+1, since we added the residual matrix)
    vars_i = np.array(init) # initial variance components
    for iter in range(max_iter):
        # V matrix
        (_, V_inv) = _get_V_components(Vs_i, vars_i)  # computes V, and V_inv
        # P matrix
        P = _get_P_matrix(V_inv, X)  # computes P matrix

        # Score Matrix (gradient of the log-likelihood)
        dLs = np.zeros(M)
        for i in range(M):
            # Equation 27.33 of textbook
            dL = -0.5*np.trace(P @ Vs_i[i]) + 0.5*(y.T @ P @ Vs_i[i] @ P @ y)
            dLs[i] = dL
        
        # Information Matrix (second derivative of the log-likelihood)
        if method in ['NR', 'FS']:
            # this is the expected Hessian matrix (Fisher's Information Matrix)
            I_exp = _get_F_matrix(P, Vs_i)
        if method in ['NR','AI']:
            # this is the average of the expected and observed Hessian matrices
            I_avg = _get_AI_matrix(P, Vs_i, y)
        if method == 'FS':
            # Fisher Scoring method uses the expected Hessian
            H = I_exp
        elif method == 'AI':
            # AI method uses the average of the expected and observed Hessian
            H = I_avg
        elif method == 'NR':
            # Newton-Raphson method uses the observed Hessian
            H = -1 * (I_exp - 2*I_avg)

        # computes offset for parameter estimates
        # Equation 27.32 of textbook
        offsets = np.linalg.pinv(H) @ dLs # pinv for numerical stability
        # updates parameter estimates
        vars_i += offsets
        vars_i = np.maximum(vars_i, 1e-6) # ensures variance components are positive

        for i in range(M):
            if verbose and i < M - 1:  # don't print for residual variance component
                print(f"Iteration {iter+1}, Random Effect {i+1}: Updated variance component = {vars_i[i]:.6f}, Offset = {offsets[i]:.6f}")

        # check convergence
        if np.max(np.abs(offsets)) < tol:
            print(f"Converged after {iter+1} iterations.")
            break
        if iter == max_iter - 1:
            print(f"Reached maximum iterations ({max_iter}) without convergence.")
            break
    
    return vars_i