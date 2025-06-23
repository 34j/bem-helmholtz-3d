from typing import Literal
from typing import TypeVar
from scipy.special import jv, jvp, yv, yvp
from array_api._2024_12 import Array, ArrayNamespace
from array_api_compat import array_namespace
TArray = TypeVar("TArray", bound=Array)
def sjv(
    v: TArray,
    d: TArray,
    z: TArray,
    derivative: bool = False,
) -> TArray:
    """
    hyperspherical Bessel function of the first kind.

    Parameters
    ----------
    v : TArray
        The degree of the hyperspherical Bessel function.
    d : TArray
        The dimension of the hypersphere.
    z : TArray
        The argument of the hyperspherical Bessel function.
    derivative : bool, optional
        Whether to compute the derivative of the
        hyperspherical Bessel function, by default False

    Returns
    -------
    Array
        The hyperspherical Bessel function of the first kind.

    References
    ----------
    McLean, W. (2000). Strongly Elliptic Systems and
    Boundary Integral Equations. p.279

    """
    ivy = array_namespace(v, d, z)
    if ((d > 2) & (v < 0)).any():
        raise ValueError(
            "The hyperspherical Bessel function of "
            "the first kind is not defined for negative degrees."
        )
    if d > 2 and derivative:
        return v / z * sjv(v, d, z) - sjv(v + 1, d, z)
    d_half_minus_1 = d / 2 - 1
    return (
        ivy.sqrt(ivy.pi / 2)
        * ivy.asarray(
            (jvp if derivative else jv)(
                (v + d_half_minus_1), (z)
            )
        )
        / (z**d_half_minus_1)
    )


def syv(
    v: TArray,
    d: TArray,
    z: TArray,
    derivative: bool = False,
) -> TArray:
    """
    hyperspherical Bessel function of the second kind.

    Parameters
    ----------
    v : TArray
        The degree of the hyperspherical Bessel function.
    d : TArray
        The dimension of the hypersphere.
    z : TArray
        The argument of the hyperspherical Bessel function.
    derivative : bool, optional
        Whether to compute the derivative of the
        hyperspherical Bessel function, by default False

    Returns
    -------
    Array
        The hyperspherical Bessel function of the second kind.

    References
    ----------
    McLean, W. (2000). Strongly Elliptic Systems and
    Boundary Integral Equations. p.279

    """
    ivy = array_namespace(v, d, z)
    if ((d > 2) & (v < 0)).any():
        raise ValueError(
            "The hyperspherical Bessel function of "
            "the second kind is not defined for negative degrees."
        )
    if d > 2 and derivative:
        return v / z * syv(v, d, z) - syv(v + 1, d, z)
    d_half_minus_1 = d / 2 - 1
    return (
        ivy.sqrt(ivy.pi / 2)
        * ivy.asarray(
            (yvp if derivative else yv)(
                (v + d_half_minus_1), (z)
            )
        )
        / (z**d_half_minus_1)
    )


def shn1(
    v: TArray,
    d: TArray,
    z: TArray,
    derivative: bool = False,
) -> TArray:
    """
    hyperspherical Hankel function of the first kind.

    Parameters
    ----------
    v : TArray
        The degree of the hyperspherical Hankel function.
    d : TArray
        The dimension of the hypersphere.
    z : TArray
        The argument of the hyperspherical Hankel function.
    derivative : bool, optional
        Whether to compute the derivative of the
        hyperspherical Hankel function, by default False

    Returns
    -------
    Array
        The hyperspherical Hankel function of the first kind.

    """
    return sjv(v, d, z, derivative) + 1j * syv(v, d, z, derivative)


def shn2(
    v: TArray,
    d: TArray,
    z: TArray,
    derivative: bool = False,
) -> TArray:
    """
    hyperspherical Hankel function of the second kind.

    Parameters
    ----------
    v : TArray
        The degree of the hyperspherical Hankel function.
    d : TArray
        The dimension of the hypersphere.
    z : TArray
        The argument of the hyperspherical Hankel function.
    derivative : bool, optional
        Whether to compute the derivative of the
        hyperspherical Hankel function, by default False

    Returns
    -------
    Array
        The hyperspherical Hankel function of the second kind.

    """
    return sjv(v, d, z, derivative) - 1j * syv(v, d, z, derivative)


def szv(
    v: TArray,
    d: TArray,
    z: TArray,
    type: Literal["j", "y", "h1", "h2"],
    derivative: bool = False,
) -> Array:
    """
    Utility function to compute hyperspherical functions.

    Parameters
    ----------
    v : TArray
        The degree of the hyperspherical Hankel function.
    d : TArray
        The dimension of the hypersphere.
    z : TArray
        The argument of the hyperspherical Hankel function.
    type : Literal["j", "y", "h1", "h2"]
        The type of the hyperspherical function.
    derivative : bool, optional
        Whether to compute the derivative of the
        hyperspherical Hankel function, by default False

    Returns
    -------
    TArray
        The hyperspherical function.

    """
    if type == "j":
        return sjv(v, d, z, derivative)
    if type == "y":
        return syv(v, d, z, derivative)
    if type == "h1":
        return shn1(v, d, z, derivative)
    if type == "h2":
        return shn2(v, d, z, derivative)
    raise ValueError(f"Invalid type {type}.")