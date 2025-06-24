from collections.abc import Callable
from typing import Literal, overload

import attrs
import numpy as np
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
        Whether to sum over all simplices or return the value for each simplex separately,
        by default True.

    Returns
    -------
    TArray
        The value of the single layer potential at x of shape (...,) if sum_all_elements is True,
        (..., n_simplex) if sum_all_elements is False.

    """
    xp = array_namespace(x, simplex_vertices)
    if not (x.shape[-1] == simplex_vertices.shape[-2] == simplex_vertices.shape[-1]):
        raise ValueError(
            f"The last dimension of x and last two dimensions of simplex_vertices must match, "
            f"got {x.shape=} and {simplex_vertices.shape=}."
        )
    if fx is not None and fx.shape[-1] != simplex_vertices.shape[-3]:
        raise ValueError(
            f"The last dimension of fx must match the number of simplices, "
            f"got {fx.shape=} and {simplex_vertices.shape=}."
        )
    extra_shapes = [x.shape[:-1], simplex_vertices.shape[:-3], k.shape]
    if fx is not None:
        extra_shapes.append(fx.shape[:-1])
    if xp.unique_values([len(s) for s in extra_shapes]).size != 1:
        raise ValueError(
            "The shapes of x, simplex_vertices, k, and fx must be compatible. "
            f"Got {x.shape=}, {simplex_vertices.shape=}, {k.shape=}"
            + (f", {fx.shape=}" if fx is not None else "")
            + f", {extra_shapes=}."
        )
    np.broadcast_shapes(*[tuple(s) for s in extra_shapes])
    d = simplex_vertices.shape[-1]
    if d is None or d < 1:
        raise ValueError(f"The last dimension of simplex_vertices must be at least 1, got {d}.")
    if quadrature_points_and_weights is None:
        scheme = quadpy.tn.grundmann_moeller(d - 1, 2)
        points, weights = scheme.points.T, scheme.weights
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
    # (..., n_simplex, d (vertices), d (coordinates)) ->
    # (..., n_simplex, n_quadrature, d (vertices), d (coordinates))
    # (n_quadrature, d (vertices)) -> (1, n_quadrature, d (vertices), 1)
    # (..., n_simplex, n_quadrature, d (coordinates))
    points_simplex = xp.vecdot(
        simplex_vertices[..., :, None, :, :], points[None, :, :, None], axis=-2
    )
    # print(xp.linalg.vector_norm(x[..., None, None, :] - points_simplex, axis=-1).sum(axis=-1))
    # (..., n_simplex, n_quadrature)
    fundamental_sol = fundamental_solution(xp.asarray(d), x[..., None, None, :] - points_simplex, k)
    fundamental_sol = np.nan_to_num(fundamental_sol, nan=0.0)
    if xp.any(xp.isnan(fundamental_sol)):
        raise ValueError(
            "The fundamental solution contains NaN values. "
            "This may happen if x is too close to the center of the simplex. "
            f"NaN rate: {xp.sum(xp.isnan(fundamental_sol)) / fundamental_sol.size:.2%}."
        )
    # (..., n_simplex)
    vol = simplex_volume(simplex_vertices)
    # (..., n_simplex)
    result = xp.vecdot(
        fundamental_sol,
        weights,
        axis=-1,
    )
    result *= vol
    if sum_all_elements:
        if fx is None:
            result = xp.sum(result, axis=-1)
        else:
            result = xp.vecdot(result, fx, axis=-1)
    return result


@attrs.frozen(kw_only=True)
class BEMCalculator[TArray: Array]:
    """Calculate fields from the solution of the boundary element method."""

    simplex_vertices: TArray
    """The vertices of the simplices of shape (..., n_simplex, d (vertices), d (coordinates))."""
    uin: Callable[[TArray], TArray]
    """The incident wave function which takes x of shape (..., d) and returns a value
    of shape (...,)."""
    sol: TArray
    """The neumann or dirichlet data of shape (..., n_simplex)."""
    k: TArray
    """Wavenumber of the Helmholtz equation of shape (...,)."""
    quadrature_points_and_weights: tuple[TArray, TArray] | None = None
    """Quadrature points and weights for the integration, by default None.
    points are of shape (n_quadrature, d (vertices)) (barycentric coordinates),
    weights are of shape (n_quadrature,)."""

    @property
    def extra_dim(self) -> int:
        """The number of extra dimensions ..."""
        return self.sol.ndim - 1

    def uscat(self, x: TArray, /) -> TArray:
        """
        Scattered wave function at the point x.

        Parameters
        ----------
        x : TArray
            The point at which to evaluate the scattered wave function
            of shape (...(x), d (coordinates)).

        Returns
        -------
        TArray
            The value of the scattered wave function at x of shape (...(x),...(sol)).

        """
        extra_dim = x.ndim - 1
        return -single_layer_potential(
            x=x[(...,) + (None,) * self.extra_dim + (slice(None),)],
            simplex_vertices=self.simplex_vertices[(None,) * extra_dim + (...,)],
            k=self.k[(None,) * extra_dim + (...,)],
            quadrature_points_and_weights=self.quadrature_points_and_weights,
            fx=self.sol[(None,) * extra_dim + (...,)],
            sum_all_elements=True,
        )

    def utotal(self, x: TArray, /) -> TArray:
        """
        Total wave function at the point x.

        The sum of the incident and scattered wave functions.

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


@overload
def bem(
    *,
    simplex_vertices: Array,
    uin: Callable[[Array], Array],
    k: Array,
    quadrature_points_and_weights: tuple[Array, Array] | None = None,
    return_matrix: Literal[False] = ...,
) -> BEMCalculator[Array]: ...
@overload
def bem(
    *,
    simplex_vertices: Array,
    uin: Callable[[Array], Array],
    k: Array,
    quadrature_points_and_weights: tuple[Array, Array] | None = None,
    return_matrix: Literal[True] = ...,
) -> tuple[Array, Array]: ...
def bem[TArray: Array](
    *,
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
    return_matrix : bool, optional
        If True, return the left-hand side and right-hand side matrices of the linear system,
        otherwise return BEMCalculator with the solution of the linear system,

    Returns
    -------
    BEMCalculator[TArray] | tuple[TArray, TArray]
        An object containing the boundary element method matrix and the solution.
        If `return_matrix` is True, it returns the left-hand side and right-hand side matrices.
        Otherwise, it returns the solution of the linear system.

    """
    xp = array_namespace(simplex_vertices, k)
    n_simplex = simplex_vertices.shape[-3]
    # (..., n_simplex (x), d (coordinates))
    centers = xp.mean(simplex_vertices, axis=-2)
    # (..., n_simplex (x), n_simplex (y))
    lhs = 0.5 * xp.eye(n_simplex) + single_layer_potential(
        x=centers,
        simplex_vertices=simplex_vertices[..., None, :, :, :],
        k=k[..., None],
        quadrature_points_and_weights=quadrature_points_and_weights,
        sum_all_elements=False,
    )
    # (..., n_simplex (x), 1)
    rhs = uin(centers)
    if return_matrix:
        return lhs, rhs
    if xp.any(xp.isnan(lhs)):
        raise ValueError(
            "The left-hand side matrix contains NaN values. "
            f"NaN rate: {xp.sum(xp.isnan(lhs)) / lhs.size:.2%}."
        )
    if xp.any(xp.isnan(rhs)):
        raise ValueError(
            "The right-hand side vector contains NaN values. "
            f"NaN rate: {xp.sum(xp.isnan(rhs)) / rhs.size:.2%}."
        )
    sol = xp.linalg.solve(lhs, rhs)
    # print(centers, lhs, rhs, sol)
    return BEMCalculator(
        simplex_vertices=simplex_vertices,
        uin=uin,
        sol=sol,
        quadrature_points_and_weights=quadrature_points_and_weights,
        k=k,
    )
