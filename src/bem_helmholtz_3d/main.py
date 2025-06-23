from collections.abc import Callable

import attrs
import quadpy
from array_api._2024_12 import Array
from array_api_compat import array_namespace

from .simplex import simplex_volume
from .special import fundamental_solution


def single_layer_potential[TArray: Array](
    *,
    x: TArray,
    simplex_vertices: TArray,
    k: TArray,
    quadrature_points_and_weights: tuple[TArray, TArray] | None = None,
    sum_all_elements: bool = True,
    fx: TArray | None = None,
) -> TArray:
    """
    Single layer potential for the boundary element method.

    Parameters
    ----------
    x : TArray
        The point at which to evaluate the single layer potential of shape (..., d (coordinates)).
    simplex_vertices : TArray
        The vertices of the simplices of shape (..., n_simplex, d (vertices), d (coordinates)).
    k : TArray
        Wavenumber of the Helmholtz equation of shape (...,).
    quadrature_points_and_weights : tuple[TArray, TArray] | None, optional
        Quadrature points and weights for the integration, by default None.
        points are of shape (n_quadrature, d (vertices)) (barycentric coordinates),
        weights are of shape (n_quadrature,).
    fx : TArray, optional
        The function values at the (P0) element of shape (..., n_simplex).
    sum_all_elements : bool, optional
        Whether to sum over all simplices or return the value for each simplex separately, by default True.

    Returns
    -------
    TArray
        The value of the single layer potential at x of shape (...,) if sum_all_elements is True,
        (..., n_simplex) if sum_all_elements is False.

    """
    xp = array_namespace(x, simplex_vertices)
    if simplex_vertices.shape[-2] != simplex_vertices.shape[-1]:
        raise ValueError(
            f"The last two dimensions of simplex_vertices must match.Got {simplex_vertices.shape=}."
        )
    d = simplex_vertices.shape[-1]
    if d is None or d < 1:
        raise ValueError(f"The last dimension of simplex_vertices must be at least 1, got {d}.")
    if quadrature_points_and_weights is None:
        scheme = quadpy.tn.grundmann_moeller(d, 2)
        points, weights = scheme.points, scheme.weights
    else:
        # (n_quadrature, d (vertices)), (n_quadrature,)
        points, weights = quadrature_points_and_weights
        if points.shape[-1] != d:
            raise ValueError(
                f"The last dimension of points must match the simplex dimension {d}, "
                f"got {points.shape[-1]}."
            )
        if weights.shape != points.shape[:-1]:
            raise ValueError(
                "Weights must have the same shape as points except for the last dimension."
            )
    # (..., n_simplex, d (coordinates), d (vertices)) -> (..., n_simplex, n_quadrature, d (coordinates), d (vertices))
    # (n_quadrature, d (vertices)) -> (1, n_quadrature, 1, d (vertices))
    # (..., n_simplex, n_quadrature, d (coordinates))
    points_simplex = xp.vecdot(
        simplex_vertices[..., :, None, :, :], points[None, :, None, :], axis=-1
    )
    # (..., n_simplex, n_quadrature)
    fundamental_sol = fundamental_solution(xp.asarray(d), x[..., None, None, :] - points_simplex, k)
    # (..., n_simplex)
    vol = simplex_volume(simplex_vertices)
    # (..., n_simplex)
    result = xp.vecdot(
        fundamental_sol,
        weights * vol[..., None],
        axis=-1,
    )
    if sum_all_elements:
        if fx is None:
            result = xp.sum(result, axis=-1)
        else:
            result = xp.vecdot(result, fx, axis=-1)
    return result


@attrs.frozen(kw_only=True)
class BEMCalculator[TArray: Array]:
    simplex_vertices: TArray
    """The vertices of the simplices of shape (..., n_simplex, d (vertices), d (coordinates))."""
    uin: Callable[[TArray], TArray]
    """The incident wave function which takes x of shape (..., d) and returns a value of shape (...,)."""
    sol: TArray
    """The neumann or dirichlet data of shape (..., n_simplex)."""
    k: TArray
    """Wavenumber of the Helmholtz equation of shape (...,)."""
    quadrature_points_and_weights: tuple[TArray, TArray] | None = None
    """Quadrature points and weights for the integration, by default None.
    points are of shape (n_quadrature, d (vertices)) (barycentric coordinates),
    weights are of shape (n_quadrature,)."""

    def uscat(self, x: TArray, /) -> TArray:
        """
        Scattered wave function at the point x.

        Parameters
        ----------
        x : TArray
            The point at which to evaluate the scattered wave function of shape (..., d (coordinates)).

        Returns
        -------
        TArray
            The value of the scattered wave function at x of shape (...,).

        """
        return -single_layer_potential(
            x=x,
            simplex_vertices=self.simplex_vertices,
            k=self.k,
            quadrature_points_and_weights=self.quadrature_points_and_weights,
            fx=self.sol,
            sum_all_elements=True,
        )

    def utotal(self, x: TArray, /) -> TArray:
        """
        Total wave function at the point x, which is the sum of the incident and scattered wave functions.

        Parameters
        ----------
        x : TArray
            The point at which to evaluate the total wave function of shape (..., d (coordinates)).

        Returns
        -------
        TArray
            The value of the total wave function at x of shape (...,).

        """
        return self.uin(x) + self.uscat(x)


def bem_matrix[TArray: Array](
    simplex_vertices: TArray,
    uin: Callable[[TArray], TArray],
    k: TArray,
    quadrature_points_and_weights: tuple[TArray, TArray] | None = None,
    return_matrix: bool = False,
) -> BEMCalculator[TArray] | tuple[TArray, TArray]:
    """
    Boundary element method based on collocation method and P0 element.

    The collocation points are the centers of the simplices.

    Parameters
    ----------
    simplex_vertices : TArray
        The vertices of the simplices of shape (..., n_simplex, d (vertices), d (coordinates)).
    uin : Callable[[TArray], TArray]
        The incident wave function which takes
        x of shape (..., d) and returns a value of shape (...,).
    k : TArray
        Wavenumber of the Helmholtz equation of shape (...,).
    quadrature_points_and_weights : tuple[TArray, TArray] | None, optional
        Quadrature points and weights for the integration, by default None.
        points are of shape (n_quadrature, d (vertices)) (barycentric coordinates),
        weights are of shape (n_quadrature,).

    Returns
    -------
    BEMCalculator[TArray]
        An object containing the boundary element method matrix and the solution.
        If `return_matrix` is True, it returns the left-hand side and right-hand side matrices.
        Otherwise, it returns the solution of the linear system.

    """
    xp = array_namespace(simplex_vertices, k)
    # (..., n_simplex (x), d (coordinates))
    centers = xp.mean(simplex_vertices, axis=-1)
    # (..., n_simplex (x), 1, d (coordinates))
    centers = centers[..., :, None, :]
    # (..., n_simplex (x), n_simplex (y))
    lhs = xp.asarray(1 / 2) + single_layer_potential(
        x=centers,
        simplex_vertices=simplex_vertices[..., None, :, :, :],
        k=k[..., None],
        quadrature_points_and_weights=quadrature_points_and_weights,
    )
    # (..., n_simplex (x), 1)
    rhs = uin(centers)
    if return_matrix:
        return lhs, rhs
    sol = xp.linalg.solve(lhs, rhs)
    return BEMCalculator(
        simplex_vertices=simplex_vertices,
        uin=uin,
        sol=sol,
        quadrature_points_and_weights=quadrature_points_and_weights,
        k=k,
    )
