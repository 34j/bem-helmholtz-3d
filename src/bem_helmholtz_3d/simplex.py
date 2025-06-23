from array_api._2024_12 import Array
from array_api_compat import array_namespace


def simplex_volume[TArray: Array](simplex_vertices: TArray, /) -> TArray:
    """
    Compute the volume of a simplex given its vertices.

    Parameters
    ----------
    simplex_vertices : TArray
        Vertices of the simplex of shape (..., n, d), where d is the dimension
        and n is the number of vertices (d + 1 for a d-dimensional simplex).

        n should satisfy 0 < n <= d + 1.

    Returns
    -------
    TArray
        Volume of the simplex of shape (...)

    References
    ----------
    Contributors to Wikimedia projects. (2025, April 22).
    Cayley–Menger determinant - Wikipedia.
    Retrieved from
    https://en.wikipedia.org/w/index.php?title=Cayley-Menger_determinant&oldid=1286928466

    """
    xp = array_namespace(simplex_vertices)
    n = simplex_vertices.shape[-2]  # number of vertices
    # (n, d) -> (n, n, d) -> (n, n)
    dists = xp.linalg.vector_norm(
        simplex_vertices[..., :, None, :] - simplex_vertices[..., None, :, :], axis=-1
    )
    mat = xp.concat(
        [
            xp.concat([dists, xp.ones_like(dists[..., 0, :])], axis=-2),
            xp.concat(
                [xp.ones_like(dists[..., :, 0]), xp.zeros_like(dists[..., 0, 0])],
                axis=-2,
            ),
        ],
        axis=-1,
    )
    coef = (-1) ** (n + 1) / (xp.prod(xp.arange(n + 1))) ** 2 / 2**n
    return coef * xp.linalg.det(mat)
