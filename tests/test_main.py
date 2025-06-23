import matplotlib.pyplot as plt
import numpy as np
from array_api._2024_12 import Array
from array_api_compat import array_namespace

from bem_helmholtz_3d.main import bem


def test_bem():
    k = 1.0
    N = 20
    xp = array_namespace(np.array(0))

    def uin[TArray: Array](x: TArray) -> TArray:
        xp = array_namespace(x)
        return xp.exp(1j * k * x[..., 0])

    # (N, 2)
    vertices = xp.exp(2j * xp.pi * xp.arange(N) / N)
    vertices = xp.stack([vertices.real, vertices.imag], axis=-1)
    # (N, 2)
    simplex = xp.stack([xp.arange(N), (xp.arange(N) + 1) % N], axis=-1)
    simplex_vertices = vertices[simplex, :]

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
    fig.savefig("tests/simplex.png")
    calc = bem(simplex_vertices=simplex_vertices, k=xp.asarray(k), uin=uin)

    x, y = xp.meshgrid(xp.linspace(-4, 4, 20), xp.linspace(-4, 4, 20), indexing="ij")
    points = xp.stack([x, y], axis=-1)
    for name, u in [
        ("uin", calc.uin(points)),
        ("uscat", calc.uscat(points)),
        ("utotal", calc.utotal(points)),
    ]:
        # heatmap
        fig, ax = plt.subplots()
        c = ax.pcolormesh(x, y, xp.real(u), shading="auto")
        ax.set_title("Real part of the total potential")
        fig.colorbar(c, ax=ax)
        fig.savefig(f"tests/{name}.png")
