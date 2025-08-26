from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from array_api._2024_12 import Array, ArrayNamespace
from array_api_compat import array_namespace

from bem_helmholtz_3d.main import bem, single_layer_potential
from bem_helmholtz_3d.special import potential_coef


def circle_mesh[TArray: Array](N: int, /, xp: ArrayNamespace[TArray, Any, Any] = np) -> TArray:
    # (N, 2)
    vertices = xp.exp(2j * xp.pi * xp.arange(N) / N)
    vertices = xp.stack([vertices.real, vertices.imag], axis=-1)
    # (N, 2)
    simplex = xp.stack([xp.arange(N), (xp.arange(N) + 1) % N], axis=-1)
    simplex_vertices = vertices[simplex, :]
    return simplex_vertices


def test_circle_mesh():
    N = 20
    xp = array_namespace(np.array(0))
    simplex_vertices = circle_mesh(N, xp=xp)
    fig, ax = plt.subplots()
    for i, simplex_ in enumerate(simplex_vertices):
        ax.plot(simplex_[:, 0], simplex_[:, 1])
        ax.text(
            xp.mean(simplex_[:, 0]),
            xp.mean(simplex_[:, 1]),
            str(i),
            fontsize=8,
            ha="center",
            va="center",
        )
    ax.set_title("Simplex vertices")
    fig.savefig("tests/circle_mesh.png")
    assert simplex_vertices.shape == (N, 2, 2)


def test_slp():
    N = 20
    xp = array_namespace(np.array(0))
    k = 1
    simplex_vertices = circle_mesh(N)
    centers = xp.mean(simplex_vertices, axis=-1)
    single_layer = single_layer_potential(
        x=centers,
        simplex_vertices=simplex_vertices[None, ...],
        k=xp.asarray(k)[None, ...],
    )
    assert xp.allclose(single_layer, potential_coef(0, 2, xp.asarray(k), y_abs=1, x_abs=1))
    print(single_layer)


def test_bem():
    k = 1.0
    N = 20
    xp = array_namespace(np.array(0))

    def uin[TArray: Array](x: TArray) -> TArray:
        xp = array_namespace(x)
        return xp.exp(1j * k * x[..., 0])

    simplex_vertices = circle_mesh(N, xp=xp)
    calc = bem(simplex_vertices=simplex_vertices, k=xp.asarray(k), uin=uin)
    x, y = xp.meshgrid(xp.linspace(-4, 4, 100), xp.linspace(-4, 4, 100), indexing="ij")
    points = xp.stack([x, y], axis=-1)
    for name, u in [
        ("uin", calc.uin(points)),
        ("uscat", calc.uscat(points)),
        ("utotal", calc.utotal(points)),
    ]:
        u[x**2 + y**2 < 1] = 0  # zero inside the unit circle
        # heatmap
        fig, ax = plt.subplots()
        c = ax.pcolormesh(x, y, xp.real(u), shading="auto", cmap="bwr", vmin=-1, vmax=1)
        ax.set_title(f"Re({name})")
        fig.colorbar(c, ax=ax)
        fig.savefig(f"tests/{name}.png")
