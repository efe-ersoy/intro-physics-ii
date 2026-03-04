import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Required: numpy, matplotlib


def E_field_point_charges(X, Y, charges):
    Ex, Ey = np.zeros_like(X), np.zeros_like(Y)
    for charge in charges:
        dx = X - charge["pos"][0]
        dy = Y - charge["pos"][1]
        r2 = dx**2 + dy**2
        r3 = np.maximum(r2**1.5, 1e-10)  # Avoid division by zero
        Ex += charge["q"] * dx / r3
        Ey += charge["q"] * dy / r3
    return Ex, Ey


if __name__ == "__main__":
    # To use, simply define the point charges below. Each charge is a dictionary with keys "q" for charge magnitude and "pos" for position (x, y).
    # Then choose which quantities to visualize: field lines, field direction, or field magnitude.
    #
    charges = [
        {"q": 1.0, "pos": np.array([-1.0, -1.0])},
        {"q": 1.0, "pos": np.array([-1.0, 1.0])},
        {"q": -1.0, "pos": np.array([1.0, 1.0])},
        {"q": -1.0, "pos": np.array([1.0, -1.0])},
    ]

    show_field_lines = True
    show_field_direction = True
    show_field_magnitude = True

    x = np.linspace(-3, 3, 400)
    y = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(x, y)

    Ex, Ey = E_field_point_charges(X, Y, charges)

    angle = np.arctan2(Ey, Ex)  # range [-pi, pi]

    E = np.sqrt(Ex**2 + Ey**2)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))

    for charge in charges:
        ax.scatter(
            *charge["pos"],
            c="red" if charge["q"] > 0 else "blue",
            s=150,
            edgecolors="k",
            linewidths=2,
        )
    title_str = "Electric field from point charges"
    if show_field_lines:
        ax.streamplot(X, Y, Ex, Ey, color="k", density=2, linewidth=0.5)
        title_str += "Field Lines\n"
    if show_field_direction:
        im = ax.imshow(angle, extent=[-3, 3, -3, 3], origin="lower", cmap="hsv")
        plt.colorbar(im, ax=ax, label="Field Direction (radians)")
        title_str += "\n with field direction"
    elif show_field_magnitude:
        im = ax.imshow(
            E,
            extent=[-3, 3, -3, 3],
            origin="lower",
            cmap="inferno",
            norm=LogNorm(vmin=1e-3, vmax=E.max()),
        )
        plt.colorbar(im, ax=ax, label="Field Magnitude")
        title_str += "\n with field magnitude"

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title_str)
    plt.show()
