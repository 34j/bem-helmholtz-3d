from collections.abc import Callable

import quadpy
from array_api._2024_12 import Array
from array_api_compat import array_namespace

from .special import fundamental_solution


def bem_matrix[TArray: Array](
    vertices: TArray,
    simplex: TArray,
    uin: Callable[[TArray], TArray],
    k: TArray,
    quadrature_points_and_weights: tuple[TArray, TArray] | None = None,
) -> tuple[TArray, TArray]:
    """
    Boundary element method based on collocation method and P0 element.

    The collocation points are the centers of the simplices.

    Parameters
    ----------
    vertices : TArray
        The vertices of the mesh of shape (..., n_vertices, d).
    simplex : TArray
        The indices of the vertices that form simplex of shape (..., n_simpex, d).
    uin : Callable[[TArray], TArray]
        The incident wave function which takes
        x of shape (..., d) and returns a value of shape (...,).
    k : float
        Wavenumber of the Helmholtz equation of shape (...,).
    quadrature_points_and_weights : tuple[TArray, TArray] | None, optional
        Quadrature points and weights for the integration, by default None.

    Returns
    -------
    Left-hand side and right-hand side matrices.

    """
    if vertices.shape[-1] != simplex.shape[-1]:
        raise ValueError(
            "The last dimension of vertices and simplex must match."
            f"Got {vertices.shape[-1]} and {simplex.shape[-1]}."
        )
    d = simplex.shape[-1]
    # n_vertices = vertices.shape[0]
    # n_simplex = simplex.shape[0]
    xp = array_namespace(vertices, simplex)
    # (..., n_simplex, d (coordinates), d (vertices))
    simplex_vertices = vertices[..., simplex, :]
    # (..., n_simplex, d (coordinates))
    centers = xp.mean(simplex_vertices, axis=-1)
    if quadrature_points_and_weights is None:
        scheme = quadpy.tn.grundmann_moeller(d, 2)
        points, weights = scheme.points, scheme.weights
    else:
        # (n_quadrature, d (vertices)), (n_quadrature,)
        points, weights = quadrature_points_and_weights
        if points.shape[-1] != simplex.shape[-1]:
            raise ValueError(
                "The last dimension of points must match the simplex dimension."
                f"Got {points.shape[-1]} and {simplex.shape[-1]}."
            )
        if weights.shape != points.shape[:-1]:
            raise ValueError(
                "Weights must have the same shape as points except for the last dimension."
            )
    # (..., n_simplex, 1, d (coordinates))
    centers = centers[..., :, None, :]
    # (..., n_simplex, d (coordinates), d (vertices)),
    # (n_quadrature, d (vertices)) ->
    # (..., n_quadrature, 1, n_simplex, d (coordinates))
    points_simplex = xp.vecdot(
        simplex_vertices[..., None, :, :, :], points[:, None, None, :], axis=-1
    )
    # (..., n_quadrature, n_simplex (x), n_simplex (y))
    lhs = xp.asarray(1 / 2) + xp.sum(
        fundamental_solution(xp.asarray(d), centers[None, ...] - points_simplex, k),
        axis=-1,
    )
    # (..., n_simplex, 1)
    rhs = uin(centers)
    return lhs, rhs
