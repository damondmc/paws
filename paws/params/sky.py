import numpy as np


def sky_sampler(target_ra, target_dec, sky_radius, n_samples):
    """
    Samples sky locations uniformly within a circular cap of radius sky_radius
    (radians) around (target_ra, target_dec).
    """
    z_min = np.cos(sky_radius)
    z = np.random.uniform(z_min, 1.0, n_samples)
    theta = np.arccos(z)
    phi = np.random.uniform(0, 2 * np.pi, n_samples)

    sin_d, cos_d = np.sin(target_dec), np.cos(target_dec)
    sin_a, cos_a = np.sin(target_ra), np.cos(target_ra)

    k = np.array([cos_d * cos_a, cos_d * sin_a, sin_d])

    if abs(k[2]) < 0.99:
        u = np.cross([0, 0, 1], k)
    else:
        u = np.cross([0, 1, 0], k)
    u /= np.linalg.norm(u)
    v = np.cross(k, u)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)

    vec = (
        x[None, :] * u[:, None]
        + y[None, :] * v[:, None]
        + cos_theta[None, :] * k[:, None]
    )

    delta = np.arcsin(vec[2])
    alpha = np.arctan2(vec[1], vec[0]) % (2 * np.pi)
    return alpha, delta


def hexagonal_grid(num_layers):
    """
    Generates a hexagonal grid up to num_layers rings around the origin.

    Returns a dict mapping layer index -> list of (q, r, i, j) tuples where
    (q, r) are axial hex coordinates and (i, j) are 2D offsets in hex units:
        i = q + r/2
        j = r * sqrt(3)/2

    Layer 0 always contains exactly one point: the origin (0, 0, 0, 0).
    Multiply (i, j) by (spacing_alpha, spacing_delta) to get sky offsets (d_alpha, d_delta).
    """
    sqrt3_2 = np.sqrt(3) / 2.0
    layer_points = {layer: [] for layer in range(num_layers + 1)}

    for q in range(-num_layers, num_layers + 1):
        for r in range(-num_layers, num_layers + 1):
            if -q - r in range(-num_layers, num_layers + 1):
                layer = max(abs(q), abs(r), abs(-q - r))
                i = q + (r / 2.0)
                j = r * sqrt3_2
                layer_points[layer].append((q, r, i, j))

    return layer_points


def sky_hex_offsets(sky_radius, spacing_alpha, spacing_delta):
    """
    Returns (d_alpha, d_delta) offset arrays for a hex grid truncated to sky_radius.

    Generates enough hex rings to cover sky_radius, then keeps only points whose
    Euclidean offset sqrt((i*spacing_alpha)^2 + (j*spacing_delta)^2) <= sky_radius.
    The origin (0, 0) is always first; remaining points are ordered layer by layer.

    Parameters
    ----------
    sky_radius : float
        Radius of the sky region to tile (radians). Points outside are discarded.
    spacing_alpha : float
        Step size per hex unit in the alpha direction (radians).
    spacing_delta : float
        Step size per hex unit in the delta direction (radians).

    Returns
    -------
    d_alpha, d_delta : ndarray
        Offset arrays; length depends on how many hex points fall within sky_radius.
    """
    if spacing_alpha is None or spacing_delta is None:
        raise ValueError("spacing_alpha and spacing_delta must both be provided")
    if spacing_alpha > spacing_delta:
        print(
            f"Warning: spacing_alpha ({spacing_alpha}) > spacing_delta ({spacing_delta}). "
            "Typically spacing_alpha <= spacing_delta for equal sensitivity loss per sky point."
        )

    # Generate enough layers to fully cover sky_radius, plus one buffer ring
    min_spacing = min(spacing_alpha, spacing_delta)
    num_layers = int(np.ceil(sky_radius / min_spacing)) + 1

    grid = hexagonal_grid(num_layers)
    all_points = [pt for layer in sorted(grid) for pt in grid[layer]]

    # Keep only points within sky_radius
    filtered = [
        (i, j)
        for _, _, i, j in all_points
        if np.sqrt((i * spacing_alpha) ** 2 + (j * spacing_delta) ** 2) <= sky_radius
    ]

    d_alpha = np.array([i * spacing_alpha for i, j in filtered])
    d_delta = np.array([j * spacing_delta for i, j in filtered])
    return d_alpha, d_delta
