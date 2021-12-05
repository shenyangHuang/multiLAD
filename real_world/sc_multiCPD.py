# sc_power_mean.py

# standard library dependencies
from typing import List, Tuple, Callable
from numbers import Number
from math import inf, log

# external dependencies
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from tqdm import tqdm 

# local dependencies
from multiCPD_multiview_aggregators import (
    sum_,
    mean_,
    max_,
    min_,
    median_,
    scalar_power_mean
)

def difference_score(vector: np.ndarray) -> np.ndarray:
    """Simple wrapper around np.diff.

    Parameters
    ----------
    vector : np.ndarray
        Input vector.

    Returns
    -------
    np.ndarray
        A 1D NumPy vector whose first element is the first element of vector
        but the remaining items are taken from applying np.diff on said vector.
    
    Examples
    --------
    >>> import numpy as np
    >>> def brute_force_difference_score(z_scores: np.ndarray) -> np.ndarray:
    ...     z = []
    ...     for i in range(len(z_scores)):
    ...         if (i==0):
    ...             z.append(z_scores[0])
    ...         else:
    ...             z.append(z_scores[i] - z_scores[i-1])
    ...     return np.array(z)
    ... 
    >>> v = np.arange(10)
    >>> assert all(brute_force_difference_score(v) == difference_score(v))
    >>> w = np.arange(-10,1)
    >>> assert all(brute_force_difference_score(w) == difference_score(w))
    >>> x = np.random.shuffle(np.hstack(v,w))
    >>> assert all(brute_force_difference_score(x) == difference_score(x))

    """
    vector = vector.flatten()
    difference_vector = np.copy(vector)
    difference_vector[1:] = np.diff(difference_vector)
    return difference_vector

def compute_Z_score(cur_vec: np.ndarray, typical_vec: np.ndarray) -> np.ndarray:
    """Returns the Z-score obtained by applying Akoglu and Faloutsos's
    formula (Z = 1 - (u.T)r) to cur_vec and typical_vec.

    Parameters
    ----------
    cur_vec : np.ndarray
        Vector representing the current "signature" vector.
    typical_vec : np.ndarray
        Vector representing the reference "typical signature" vector.

    Returns
    -------
    np.ndarray
        Vector derived using by applying Akoglu and Faloutsos's
        formula to cur_vec and typical_vec

    Raises
    ------
    AssertionError: 
        if cur_vec and typical_vec do not have the same shape
    """

    assert cur_vec.shape == typical_vec.shape
    cur_vec = cur_vec.flatten()
    typical_vec = typical_vec.flatten()
    cosine_similarity = abs(np.dot(cur_vec, typical_vec) / np.linalg.norm(cur_vec) / np.linalg.norm(typical_vec))
    z = (1 - cosine_similarity)
    return z

def get_typical_behavior_vector(context_vecs: np.ndarray, principal: bool = True) -> np.ndarray:
    """
    Computes and returns a vector representing the "normal" or "typical" behavior in context_vecs.
    If principal = True, this vector is taken as the principal eigenvector of context_vecs.T.
    If principal = False, this vector's ith entry is the mean of context_vecs[:,i].

    Parameters
    ----------
    context_vecs : np.ndarray
        # timepoint to consider-by-# eigen NumPy array whose rows each correspond to 
        a previous timepoint.
    principal : bool, optional
        Boolean indicating whether to return the principal eigenvector (if True) or the 
        column means (if False) as the typical behavior vector, by default True.

    Returns
    -------
    [type]
        A m-shaped vector representing the "normal" or "typical" behavior in context_vecs
        (where m is the number of columns in context_vecs)

    References
    ----------
    This function combines the 'principal_vec_typical_behavior' and 'average_typical_behavior' functions.

    """
    assert context_vecs.ndim == 2
    assert min(context_vecs.shape) > 0

    if principal:
        activity_matrix = context_vecs.T
        u, s, vh = np.linalg.svd(activity_matrix, full_matrices=False)
        return u[:,0]
    else:
        return np.mean(context_vecs, axis=0)

def compute_Zs_on_many_windows( spectra: np.ndarray,
                                windows:List[int],
                                principal: bool = True,
                                difference: bool = True,
                                percent_ranked: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert len(windows) > 0
    assert all([window_length > 0 for window_length in windows])
    windows.sort()
    assert 0.0 < percent_ranked <= 1.0
    assert spectra.ndim == 2
    assert min(spectra.shape) > 0
    initial_window = max(windows)
    num_timepoints, num_eigen = spectra.shape
    z_scores = np.zeros((len(windows), num_timepoints))
    for timepoint in range(initial_window, num_timepoints):
        for window_index, window_length in enumerate(windows):
            typical_vector = get_typical_behavior_vector(
                spectra[timepoint - window_length:timepoint],
                principal = principal
            )
            current_vector = spectra[timepoint]
            z_scores[window_index, timepoint] = compute_Z_score(current_vector, typical_vector)
    if difference:
        for window_index in range(len(windows)):
            z_scores[window_index] = difference_score(z_scores[window_index])
    z_overall = np.zeros((num_timepoints,))
    for timepoint in range(initial_window+1, len(spectra)):
            z_overall[timepoint] = z_scores[:, timepoint].max()

    num_ranked = round(num_timepoints * percent_ranked)
    outliers = z_overall.argsort()[-num_ranked:][::-1]
    outliers.sort()
    return (z_overall, z_scores, outliers)

# NOTE: this isn't actually used anywhere
def find_anomalies(z_scores, initial_window_length: int, percent_ranked: float = 0.05) -> np.ndarray:
    assert initial_window_length > 0
    assert 0.0 < percent_ranked <= 1.0
    z_scores = np.array(z_scores)
    for i in range(initial_window+1):
        z_scores[i] = 0        #up to initial window + 1 are not considered anomalies. +1 is because of difference score
    num_ranked = round(len(z_scores) * percent_ranked)
    outliers = z_scores.argsort()[-num_ranked:][::-1]
    outliers.sort()
    return outliers

def get_Laplacian_matrix(   G: nx.Graph, 
                            num_nodes: int,
                            normalized_laplacian: bool = True,
                            weight: str = 'weight') -> np.matrix:
    """Convenience function used to return the Laplacian matrix
    of a graph according to some parameter specifications. 

    Parameters
    ----------
    G : nx.Graph
        The graph whose Laplacian matrix will be computed and returned.
    num_nodes : int
        The number of nodes this graph must contain.
        "Dummy" nodes are added to graphs with fewer nodes.
    normalized_laplacian : bool, optional
        Boolean specifying whether to return the normalized (if True)
        or unnormalized (if False) Laplacian matrix, by default True.
    weight : str, optional
        Optional keyword argument indicating whether to use edge weights
        in the computation of the Laplacian (if 'weight') or whether to 
        consider the graph as being unweighted (if None), by default 'weight'.

    Returns
    -------
    np.ndarray
        Laplacian matrix of G according to the parameter specifications.

    Raises
    ------
        AssertionError: if num_nodes is <= 0, or if G has more nodes than num_nodes.

    """
    assert num_nodes > 0
    for missing_node in range(len(G), num_nodes): 
        # add empty node with no connectivity (zero padding)
        G.add_node(-1 * missing_node)
    
    if normalized_laplacian: 
        try:
            L = nx.linalg.normalized_laplacian_matrix(G, weight=weight)
            L = L.todense()
        except nx.NetworkXNotImplemented as graph_is_directed:
            L = np.matrix(nx.linalg.directed_laplacian_matrix(G, weight=weight))
            L = np.asarray(L)
    else:
        try:
            L = nx.linalg.laplacian_matrix(G, weight=weight)
            L = L.todense()
        except nx.NetworkXNotImplemented as graph_is_directed:
            out_degree_matrix = np.diag( [ out_degree for node_int_id, out_degree in G.out_degree() ] )
            directed_adjacency_matrix = nx.adjacency_matrix(G, weight=weight)
            L = directed_adjacency_matrix - out_degree_matrix
        
    return L

def get_context_matrix( G_times: List[nx.Graph], 
                        num_eigen: int, 
                        p: float = None, 
                        max_size: int = None,
                        add_shift: bool = True,
                        top: bool = True,
                        normalized_laplacian: bool = True,
                        weight: str = 'weight') -> Tuple[np.ndarray, List[np.ndarray]]:
    """Returns the context matrix of the graph timecourse (a.k.a. a time-evolving graph)
    G_times. 

    Parameters
    ----------
    G_times : List[nx.Graph]
        A list of graphs, each graph being a timepoint in the timecourse.
    num_eigen : int
        Integer specifying the number of eigenvalues and eigenvectors to compute.
    p : float, optional
        Float used to parameterize the scalar-power mean operation.
        Is included because it also parameterizes the 'add_diagonal_shift'
        function used when the context matrix returned by the current function
        will be aggregated with other context matrices using the scalar power mean
        aggregator. 
        None by default (indicating that the scalar power mean aggregator won't be used).
    max_size : int, optional
        The number of nodes this graph must contain.
        "Dummy" nodes are added to graphs with fewer nodes.
        None by default (and gets calculated dynamically from the graphs in G_times).
    add_shift : bool, optional
        Boolean indicating whether to use the 'add_diagonal_shift' function
        when context matrix returned by the current function will be aggregated 
        with other context matrices using the scalar power mean aggregator.  
        True by default, but 'add_diagonal_shift' is only used if add_shift is True
        and p is not None.
    top : bool, optional
        Boolean indicating whether to pass "LM" (if True) or "SM" (if False) to
        NumPy's SVD functions, by default True.
    normalized_laplacian : bool, optional
        Boolean specifying whether to return the normalized (if True)
        or unnormalized (if False) Laplacian matrix, by default True.
    weight : str, optional
        Optional keyword argument indicating whether to use edge weights
        in the computation of the Laplacian (if 'weight') or whether to 
        consider the graph as being unweighted (if None), by default 'weight'.

    Returns
    -------
    Tuple[np.ndarray, List[np.ndarray]]
        Tuple of a NumPy array and a list of NumPy arrays.
        Temporal_eigenvalues: A # timepoints-by-num_eigen shaped NumPy array
            representing the context matrix for the graph timecourse G_times.
            Its ith rows contain the eigenvalues of the ith graph's Laplacian matrix.
        
        activity_vecs: A list of 

    References
    ----------
    This function was previously-called 'find_eigenvalues'
    
    """
    assert num_eigen > 0
    assert weight in ('weight', None)

    which = "LM" if top else "SM"

    if max_size is None:
        max_size = max([len(G) for G in G_times])
    
    assert max_size > 0
    try:
        assert max_size >= num_eigen 
    except AssertionError as ae:
        print(f"Warning: max_size={max_size} < num_eigen={num_eigen};\nSetting max_size and num_eigen to min({max_size} - 1, {num_eigen})")
        num_eigen = max_size = min(num_eigen, max_size - 1)

    num_timepoints = len(G_times)

    Temporal_eigenvalues = np.zeros((num_timepoints, num_eigen))
    activity_vecs = []  

    for timepoint_index, G in tqdm(enumerate(G_times), total=num_timepoints, desc="Processing timepoints"):
        L: np.matrix = get_Laplacian_matrix(
            G, 
            max_size, 
            normalized_laplacian = normalized_laplacian, 
            weight = weight
        )

        if (add_shift and p is not None):
            L = add_diagonal_shift(L, p)

        L = csr_matrix(L).asfptype()
        vecs, vals, vh = svds(L, k=num_eigen, which=which)
        max_index = list(vals).index(max(list(vals)))
        activity_vecs.append(np.asarray(vecs[max_index]))
        Temporal_eigenvalues[timepoint_index, :] = np.asarray(vals)

    return (Temporal_eigenvalues, activity_vecs)

# leave as-is
def add_diagonal_shift(A: np.ndarray, p: float) -> np.ndarray:
    '''
    this is required for positive definiteness of negative powers
    directly add to the eigenvalues
    1d array of eigenvalues
    p -- power
    for sparse matrix
    following the values in "The Power Mean Laplacian for Multilayer Graph Clustering" arXiv:1803.00491v1
    '''
    assert A.ndim == 2
    assert min(A.shape) > 0
    #add a small shift for 0
    if p == 0:
        shift = 1e-6
        np.fill_diagonal(A, A.diagonal() + shift)

    # do not shift for positive power
    elif p > 0:
        return A

    # shift for negative power
    else:
        shift = log(1 + abs(p))
        np.fill_diagonal(A, A.diagonal() + shift)

    return A 

# will be scrapped
def find_power_mean_eigenvalues(viewwise_G_times: List[List[nx.Graph]], 
                                num_eigen: int, 
                                p: float,
                                add_diagonal_shift: bool,
                                top: bool = True) -> List[List[float]]:

    assert num_eigen > 0
    num_views = len(viewwise_G_times)
    num_timepoints = inf
    max_num_nodes = 0
    for view_G_times in viewwise_G_times:
        max_num_nodes = max(
            max_num_nodes,
            max([len(G) for G in view_G_times])
        )
        num_timepoints = min(num_timepoints, len(view_G_times))
    
    viewwise_eigenvalues = np.zeros(
        (num_views, num_timepoints, num_eigen)
    )

    for v, view_G_times in enumerate(viewwise_G_times):
        Temporal_eigenvalues, activity_vecs = get_context_matrix(
            view_G_times,
            num_eigen,
            p,
            max_size = max_num_nodes,
            add_shift = add_diagonal_shift,
            top = top
        )
        viewwise_eigenvalues[v,...] = Temporal_eigenvalues
    
    timewise_scalar_power_mean_spectra = []

    for timepoint in range(num_timepoints):
        v_s: List[np.ndarray] = []
        for view in range(num_views):
            v_s.append(viewwise_eigenvalues[view][timepoint])
        timewise_scalar_power_mean_spectra.append(scalar_power_mean(v_s, p))
    
    return timewise_scalar_power_mean_spectra

def multiCPD(   viewwise_G_times: List[List[nx.Graph]], 
                windows: List[int],
                num_eigen: int = 499,
                p: float = None,
                max_size: int = None,
                add_diagonal_shift: bool = True,
                top: bool = True,
                principal: bool = True,
                difference: bool = True,
                percent_ranked: float = 0.05,
                normalized_laplacian: bool = True,
                weight: str = 'weight',
                multiview_agg_fn: Callable = None,
                context_matrix_norm: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    assert num_eigen > 0
    assert p is None or isinstance(p, Number)
    assert max_size is None or max_size > 0
    assert weight in ('weight', None)
    assert len(windows) > 0
    assert min(windows) > 0
    windows.sort()
    assert 0.0 < percent_ranked <= 1.0
    assert multiview_agg_fn in (sum_, mean_, max_, min_, median_, scalar_power_mean, None)
    
    if p is not None and multiview_agg_fn is not None:
        assert multiview_agg_fn in (sum_, mean_, max_, min_, median_, scalar_power_mean)
    
    if context_matrix_norm is not None:
        assert context_matrix_norm in ("l1", "l2")
    
    viewwise_context_matrices = [
        get_context_matrix(
            G_times,
            num_eigen,
            p = p,
            max_size = max_size,
            add_shift = add_diagonal_shift,
            top = top, 
            normalized_laplacian = normalized_laplacian,
            weight = weight
        )[0] 
        for G_times in viewwise_G_times
    ]
    
    if len(viewwise_context_matrices) == 1:
        context_matrix = viewwise_context_matrices[0]
    else:
        assert multiview_agg_fn is scalar_power_mean
        viewwise_context_matrices = np.array(viewwise_context_matrices)
        context_matrix = multiview_agg_fn(viewwise_context_matrices, p) 
    
    if context_matrix_norm is not None:
        context_matrix = normalize(context_matrix, context_matrix_norm)

    initial_window = max(windows)

    z_overall, z_scores, anomalies = compute_Zs_on_many_windows(
        context_matrix,
        windows,
        principal = principal,
        difference = difference,
        percent_ranked = percent_ranked
    )

    return context_matrix, z_overall, z_scores, anomalies

def plot_scores_and_spectra(context_matrix: np.ndarray, 
                            z_scores_array: np.ndarray, 
                            labels: List[str] = None,
                            x_axis_labels: List = None, 
                            colors: List[str] = None,
                            anomalous_timepoints: List[int] = None,
                            figsize: Tuple[int,int] = None,
                            title: str = None):

    assert context_matrix.ndim == 2
    assert min(context_matrix.shape) > 0
    assert z_scores_array.ndim == 2
    assert min(z_scores_array.shape) > 0
    assert z_scores_array.shape[1] == context_matrix.shape[0]
    assert len(labels) == z_scores_array.shape[0]
    
    no_spectra_fig, no_spectra_ax = plt.subplots(nrows = 1, ncols = 1, sharex=True, figsize=(10, 4) if figsize is None else figsize)
    spectra_fig, spectra_ax = plt.subplots(nrows = 2, ncols = 1, sharex=True, figsize=(10, 6) if figsize is None else figsize, gridspec_kw={'height_ratios': [1, 1]})
    
    
    spectra_ax[1].imshow(context_matrix.T, interpolation='nearest', aspect='auto')
    
    if colors is None:
        colors = ["#08519c", "#fc8d59", "#78c679", "#e34a33", "#31a354", "#b30000", "#006837"]
        if len(colors) < z_scores_array.shape[0]:
            print("Warning: some colors will be used more than once. You can avoid this by specifying the list of colors when calling this function.")

    if x_axis_labels is None:
        x_axis_labels = np.arange(z_scores_array.shape[1])
    xs = np.arange(z_scores_array.shape[1]) 
    if labels is None:
        labels = list(map(str, list(range(z_scores_array.shape[0]))))
    for i, (z_score_row, label) in enumerate(zip(z_scores_array, labels)):
        spectra_ax[0].plot(xs, z_score_row/1e-5, 'k-', label=label, color=colors[i%len(colors)])
        no_spectra_ax.plot(xs, z_score_row/1e-5, 'k-', label=label, color=colors[i%len(colors)])
    
    if anomalous_timepoints is not None:
        assert all([int(x) == x and x >= 0 for x in anomalous_timepoints])
        anomalous_timepoints = [int(x) for x in anomalous_timepoints]
        relevant_xs = [xs[i] for i in anomalous_timepoints]
        relevant_ys = z_scores_array.max(axis=0)[anomalous_timepoints]/1e-5
        relevant_labels = [x_axis_labels[i] for i in anomalous_timepoints]
        spectra_ax[0].scatter(relevant_xs, relevant_ys)
        no_spectra_ax.scatter(relevant_xs, relevant_ys)
        for (x,y,label) in zip(relevant_xs, relevant_ys, relevant_labels):
            spectra_ax[0].annotate(
                label,  
                xy=(x, y),
                rotation=45,
                textcoords="offset points", # how to position the text
                xytext=(8,0), # distance from text to points (x,y)
                ha='center',
                fontsize=16,
                color="black",
                path_effects=[patheffects.withStroke(linewidth=3, foreground='white', capstyle="round")]
            )
            spectra_ax[0].scatter(x, y, marker='v')
            no_spectra_ax.annotate(
                label,  
                xy=(x, y),
                rotation=45,
                textcoords="offset points", # how to position the text
                xytext=(8,0), # distance from text to points (x,y)
                ha='center',
                fontsize=16,
                color="black",
                path_effects=[patheffects.withStroke(linewidth=3, foreground='white', capstyle="round")]
            )
            no_spectra_ax.scatter(x, y, marker='v')

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    spectra_ax[1].set_xlabel('Date', fontsize=25)
    no_spectra_ax.set_xlabel('Date', fontsize=25)
    spectra_ax[0].set_ylabel('anomaly score\n(1e-5)', fontsize=25)
    no_spectra_ax.set_ylabel('anomaly score\n(1e-5)', fontsize=25)

    spectra_ax[0].tick_params(axis='both', length=8, width=3)
    spectra_ax[1].tick_params(axis='both', length=8, width=3)
    no_spectra_ax.tick_params(axis='both', length=8, width=3)

    plt.xticks(xs, x_axis_labels, fontsize=16)
    spectra_ax[0].set_xticks([tick_location for t, tick_location in enumerate(xs) if t%7 == 0], minor=False)
    spectra_ax[1].set_xticks([tick_location for t, tick_location in enumerate(xs) if t%7 == 0], minor=False)
    spectra_ax[1].set_xticklabels([x_axis_labels[t] for t, tick_location in enumerate(xs) if t%7 == 0], minor=False)

    no_spectra_ax.set_xticks([tick_location for t, tick_location in enumerate(xs) if t%7 == 0], minor=False)
    no_spectra_ax.set_xticklabels([x_axis_labels[t] for t, tick_location in enumerate(xs) if t%7 == 0], minor=False)
    spectra_fig.autofmt_xdate()
    no_spectra_fig.autofmt_xdate()
    if title:
        spectra_ax[0].set_title(title)
        no_spectra_ax.set_title(title)
    return spectra_fig, no_spectra_fig


def PCA_context_matrix( context_matrix: np.ndarray, 
                        point_labels: List[str] = None,
                        plot: bool = False,
                        verbose: bool = False):

    pca = PCA(n_components=2, svd_solver='full')
    pca.fit(context_matrix)
    if verbose: 
        print("explained variance ratio: ", pca.explained_variance_ratio_)
        print("singular values: ", pca.singular_values_)
    
    context_matrix_transformed = pca.transform(context_matrix)

    if plot:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].scatter(
            context_matrix_transformed[:,0], 
            context_matrix_transformed[:,1], 
            c=np.arange(context_matrix_transformed.shape[0]), 
            cmap="Reds"
        )
        ax[1].scatter(
            context_matrix_transformed[:,0], 
            context_matrix_transformed[:,1], 
            c=np.arange(context_matrix_transformed.shape[0]), 
            cmap="Reds"
        )
        ax[1].plot(
            context_matrix_transformed[:,0], 
            context_matrix_transformed[:,1], 
            'k-'
        )

        for (x,y,label) in zip(context_matrix_transformed[:,0], context_matrix_transformed[:,1], point_labels):
            ax[0].annotate(
                label,  
                xy=(x, y),
                rotation=45
            )
            ax[1].annotate(
                label,  
                xy=(x, y),
                rotation=45
            )

        return context_matrix_transformed, fig
    return context_matrix_transformed, None