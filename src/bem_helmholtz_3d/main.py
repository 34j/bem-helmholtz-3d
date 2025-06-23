from collections.abc import Callable

import quadpy
from array_api._2024_12 import Array
from array_api_compat import array_namespace


def bem_matrix[TArray: Array](
    vertices: TArray,
    simplex: TArray,
    uin: Callable[[TArray], TArray],
) -> tuple[TArray, TArray]:
    """
    Boundary element method based on collocation method
    and P0 element.

    The collocation points are the centers of the simplices.

    Parameters
    ----------
    vertices : TArray
        The vertices of the mesh of shape (n_vertices, d).
    simplex : TArray
        The indices of the vertices that form simplex of shape (n_simpex, d + 1).

    Returns
    -------
    Left-hand side and right-hand side matrices.

    """
    d = simplex.shape[-1] - 1
    n_vertices = vertices.shape[0]
    n_simplex = simplex.shape[0]
    # (n_simplex, d, d + 1)
    xp = array_namespace(vertices, simplex)
    simplex_vertices = vertices[simplex, :]
    # (n_simplex, d)
    centers = xp.mean(simplex_vertices, axis=-1)
    scheme = quadpy.tn.grundmann_moeller(d, 2)
    lhs = scheme.integrate(
        lambda x: uin(x),
        simplex_vertices,
    )
    rhs = uin(centers)
    return lhs, rhs
