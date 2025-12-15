import numpy as np
from ..definitions import phase_param_name

# =============================================================================
# Data Analysis & Clustering
# =============================================================================

def clustering(data, freqDerivOrder, cluster_nSpacing=4.0):
    """
    Clusters outliers based on spatial proximity in phase parameter space.
    
    Parameters:
        data (astropy.table.Table): The outlier data.
        freqDerivOrder (int): Order of f-derivatives.
        cluster_nSpacing (float): Clustering threshold (grid units).
    Returns:
        centers_idx (numpy.ndarray): Indices of cluster centers.
        cluster_size (numpy.ndarray): Sizes of each cluster.
        cluster_member (list of numpy.ndarray): Members of each cluster.
    """
    # Extract phase parameter names
    fn, dfn = phase_param_name(freqDerivOrder)
    
    # Create arrays for coordinates and spacings
    _data = np.column_stack([data[key] for key in fn])
    _spacing = np.column_stack([data[key] for key in dfn])

    # Retrieve loudness
    loudness = data['mean2F']

    # Sort descending by loudness
    sorted_indices = np.argsort(-loudness)
    sorted_coords = _data[sorted_indices]
    sorted_spacing = _spacing[sorted_indices]

    centers_idx = []
    cluster_size = []
    cluster_member = []
    processed_indices = set()

    # Loop over sorted samples
    for i, (center, gridsize) in enumerate(zip(sorted_coords, sorted_spacing)):
        if sorted_indices[i] in processed_indices:
            continue

        within_dim_indices = []

        # Check distance in every dimension
        for dim in range(freqDerivOrder+1):
            r0 = cluster_nSpacing * gridsize[dim]
            distances_dim = np.abs(_data[:, dim] - center[dim])
            within_dim = np.where(distances_dim <= r0)[0]
            within_dim_indices.append(within_dim)

        # Intersection of all dimensions
        within_r0_indices = within_dim_indices[0]
        for dim_indices in within_dim_indices[1:]:
            within_r0_indices = np.intersect1d(within_r0_indices, dim_indices)

        processed_indices.update(within_r0_indices)
        centers_idx.append(sorted_indices[i])
        cluster_size.append(len(within_r0_indices))
        cluster_member.append(within_r0_indices)

    centers_idx = np.array(centers_idx)
    cluster_size = np.array(cluster_size)

    print(f'{len(data)} outliers are grouped to {len(centers_idx)} clusters.')
    return centers_idx, cluster_size, cluster_member
