"""Clustering of outliers in the phase parameter space."""

import numpy as np
from jaxtyping import Int, Shaped

from ..definitions import phase_param_name


def clustering(
    data: Shaped[np.ndarray, " n_outlier"],
    spacing: dict[str, float],
    cluster_n_spacing: float = 4.0,
) -> tuple[
    Int[np.ndarray, " n_cluster"],
    Int[np.ndarray, " n_cluster"],
    list[Int[np.ndarray, " _"]],
]:
    """Clusters outliers by proximity in the phase parameter space: from the loudest outlier down, every outlier
    within cluster_n_spacing grid spacings (spacing: {df, df1dot, ...}) in every dimension joins its cluster.

    Returns (index of each cluster's center, number of outliers in each cluster, indices of each cluster's members).
    """
    # Extract phase parameter names
    freq_deriv_order = len(spacing) - 1
    fn, dfn = phase_param_name(freq_deriv_order)

    # Create arrays for coordinates and spacings
    _data = np.column_stack([data[key] for key in fn])
    gridsize = np.array([spacing[key] for key in dfn])

    # Retrieve loudness
    loudness = data["mean2F"]

    # Sort descending by loudness
    sorted_indices = np.argsort(-loudness)
    sorted_coords = _data[sorted_indices]

    centers_idx = []
    cluster_size = []
    cluster_member = []
    processed_indices = set()

    # Loop over sorted samples
    for i, center in enumerate(sorted_coords):
        if sorted_indices[i] in processed_indices:
            continue

        within_dim_indices = []

        # Check distance in every dimension
        for dim in range(freq_deriv_order + 1):
            r0 = cluster_n_spacing * gridsize[dim]
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

    print(f"{len(data)} outliers are grouped to {len(centers_idx)} clusters.")
    return centers_idx, cluster_size, cluster_member
