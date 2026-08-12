"""Grad-Shafranov diagnostic plotting utilities."""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib import colors, cm, ticker, colormaps
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LightSource


# ----------------------------------------------------------------------------------------------------------------------
def plot_surface(ax, plot_data, label_data=None, view_data=None, cmap="autumn_r", lw=0.5, r_stride=1, c_stride=1):
    """
    Draw one tensor-backed `(R, Z, value)` surface on an existing 3D axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Existing 3D axis on which to draw the surface.
    plot_data : Mapping[str, Tensor]
        Mapping containing `x_data`, `y_data`, and `z_data` tensors of matching shape.
    label_data : Mapping[str, str] | None
        Optional axis labels and `subtitle` for the panel.
    view_data : Mapping[str, float] | None
        Optional camera `elev`, `azim`, and `roll` values.
    cmap : str
        Matplotlib color-map name.
        Optional. Default: "autumn_r".
    lw : float
        Surface line width.
    r_stride, c_stride : int
        Row and column sampling strides for the surface.

    Returns
    -------
    matplotlib.cm.ScalarMappable
        Scalar mappable for an optional color bar.

    """

    if view_data is None:
        view_data = {}

    z_data = plot_data["z_data"].detach().cpu().numpy()
    ls = LightSource(  # [90, 45]
        azdeg=view_data.get("ls_azdeg", 315),  # azimuth (0-360, degrees clockwise from N). Default: 315 (from NW).
        altdeg=view_data.get("ls_altdeg", 0),  # altitude (0-90, degrees up from horizontal). Default: 45.
        hsv_min_val=0,
        hsv_max_val=0.5,
    )
    illuminated_surface = ls.shade(
        data=z_data,
        cmap=colormaps[cmap],
        blend_mode="soft",  # hsv, overlay, soft
        vert_exag=0.01,
    )

    ax.plot_surface(
        X=plot_data["x_data"].detach().cpu().numpy(),
        Y=plot_data["y_data"].detach().cpu().numpy(),
        Z=z_data,
        cmap=cmap,
        antialiased=True,
        lw=lw,
        rstride=r_stride,
        cstride=c_stride,
        # lightsource=ls,
        facecolors=illuminated_surface,
        alpha=0.75,
    )
    norm_z = colors.Normalize(vmin=plot_data["z_data"].min(), vmax=plot_data["z_data"].max())
    map_ax = cm.ScalarMappable(norm=norm_z, cmap=cmap)

    if label_data is not None:
        if "x_label" in label_data:
            ax.set_xlabel(label_data["x_label"])
        if "y_label" in label_data:
            ax.set_ylabel(label_data["y_label"])
        if "z_label" in label_data:
            ax.set_zlabel(label_data["z_label"])
        if "subtitle" in label_data:
            ax.set_title(label_data["subtitle"], pad=35)

    ax.view_init(
        elev=view_data.get("elev", 30),
        azim=view_data.get("azim", -60),
        roll=view_data.get("roll", 0),
    )

    return map_ax


# ----------------------------------------------------------------------------------------------------------------------
def plot_gs_surface(
    base_fig_data,
    surface_data,
    view_data,
    plot_title="",
    include_colorbar=False,
    fail_silently=False,
):
    """
    Add one Grad-Shafranov field panel to a figure and optionally attach its color bar.

    Parameters
    ----------
    base_fig_data : Mapping[str, Any]
        Mapping containing the target `fig`, subplot `grid_spec`, and shared R/Z `grid_data`.
    surface_data : Mapping[str, Any]
        Mapping containing `surface_for_z_data` and its two-value `z_lims` range.
    view_data : Mapping[str, float] | None
        Optional camera `elev`, `azim`, and `roll` values.
    plot_title : str
        Title for this panel.
    include_colorbar : bool
        Whether to attach a color bar to the panel.
    fail_silently : bool
        If true, suppress plotting errors so diagnostic rendering cannot interrupt training.

    Returns
    -------
    None

    """

    ax_ = base_fig_data["fig"].add_subplot(base_fig_data["grid_spec"], projection="3d")
    try:
        map_ax = plot_surface(
            ax=ax_,
            plot_data={
                "x_data": base_fig_data["grid_data"]["R_for_x_data"],
                "y_data": base_fig_data["grid_data"]["Z_for_y_data"],
                "z_data": surface_data["surface_for_z_data"],
            },
            label_data={
                "x_label": "R [m]",
                "y_label": "Z [m]",
                "z_label": surface_data.get("surface_label", ""),
                "subtitle": plot_title,
            },
            view_data=view_data,
        )
        if include_colorbar:
            # REMARK: Inclusion of colorbar may drastically affect the layout of the figure.
            base_fig_data["fig"].colorbar(
                map_ax,
                ax=ax_,
                shrink=0.5,
                pad=0.15,
                # location="bottom",
            )

        ax_.set_zlim(bottom=surface_data["z_lims"][0], top=surface_data["z_lims"][1])
        ax_.ticklabel_format(axis="z", useMathText=True, scilimits=(0, 0))  # style: ['sci', plain],
        ax_.zaxis.set_major_locator(ticker.LinearLocator(3))

        ax_.zaxis.set_ticks_position("upper")
        ax_.zaxis.set_label_position("upper")
        # Alternative: ax_.zaxis._axinfo['juggled'] = (1, 0, 2)  # (0, 2, 1)

    except Exception as ee:
        if fail_silently:
            pass
        else:
            print(f"Plotting error: {ee}")


# ----------------------------------------------------------------------------------------------------------------------
def make_gs_plots(
    plot_data: Mapping[str, Any],
    view_data: Mapping[str, Any] | None = None,
    save_plots: bool = False,
    save_path: Path | str | None = None,
    fail_silently: bool = True,
):
    """
    Make Grad-Shafranov plots.

    Parameters
    ----------
    plot_data : Mapping[str, Any]
        Mapping with plot data of the form:
        {
            "grid_data": {
                "R_for_x_data": R_data,
                "Z_for_y_data": Z_data,
                "z_lims": {
                    "gs_sides": [float, float], "psi": [float, float], "j_tor": [float, float],
                },
            },
            "gs_data": {
                "lhs_pred_data": lhs_pred_data, "lhs_ref_data": lhs_ref_data,
                "rhs_pred_data": rhs_pred_data, "rhs_ref_data": rhs_ref_data,
            },
            "signal_data": {
                "psi_pred_data": psi_pred_data, "psi_ref_data": psi_ref_data,
                "j_tor_pred_data": j_tor_pred_data, "j_tor_ref_data": j_tor_ref_data, "j_tor_case": j_tor_case,
            }
            "title_sufix": title_sufix
        }
    view_data : Mapping[str, Any] | None
        Mapping with view data of the form:
        {
            "elev": elev, "azim": azim, "roll": roll
        }
    save_plots : bool = False
        Boolean flag to save plots. If False, plots are shown.
        Optional. Default: False.
    save_path : Path | str | None
        Complete target image path when saving. Required when `save_plots` is true.
    fail_silently : bool
        Boolean flag to fail silently if True, otherwise print triggered exception.
        Optional. Default: True

    Returns
    -------
    None

    """

    j_tor_case = plot_data["signal_data"].get("j_tor_case", "predicted")

    if view_data is None:
        # Some configurations:
        # - Nice NW, but wrong tokamak orientation: [40, -49- 121]
        # - Nice NE, : [50, 45, 129]
        # - Nice NE, other: [52, 28, 114]
        # - Nice NW: [33, 28, 107]
        view_data = {
            "elev": 33,  # 90 (top view)
            "azim": 28,
            "roll": 107,
        }

    fig = plt.figure(
        figsize=(16, 8),
        layout="tight",  # layout={'constrained', 'compressed', 'tight', 'none', .LayoutEngine, None}
        # tight_layout=True
    )
    title_suffix = "" + plot_data.get("title_sufix", "")
    title = plot_data.get("title", f"Grad-Shafranov related plots\n{title_suffix.strip()}")
    fig.suptitle(t=title, fontsize=16, y=0.99)
    gs = GridSpec(nrows=2, ncols=4, figure=fig)
    fig.subplots_adjust(top=0.7, hspace=0.8, wspace=0.9)

    all_z_lims = plot_data["grid_data"].get("z_lims") or {}
    z_lims_gs_sides = all_z_lims.get("gs_sides", [-2, 2])
    z_lims_psi = all_z_lims.get("psi", [-0.2, 0.2])
    z_lims_j_tor = all_z_lims.get("j_tor", [-0.2 * 1e6, 1e6])

    g_data = plot_data["gs_data"]
    s_data = plot_data["signal_data"]

    default_subplot_titles = [
        r"$\mathrm{LHS}^{pred}:L_{mask}{\cdot}\left(-\Delta*\psi^{pred}\right)$",
        r"$\mathrm{LHS}^{true}:L_{mask}{\cdot}\left(-\Delta*\psi^{true}\right)$",
        r"$\mathrm{RHS}^{pred}:\mu_{0}{\cdot}R{\cdot}J^{" + j_tor_case[:4] + r"}_{\phi}$",
        r"$\mathrm{RHS}^{true}:\mu_{0}{\cdot}R{\cdot}J^{true}_{\phi}$",
        r"$\psi^{pred}$",
        r"$\psi^{true}$",
        r"$J^{" + j_tor_case[:4] + r"}_{\phi}$",
        r"$J^{true}_{\phi}$",
    ]
    subplot_titles = plot_data.get("subplot_titles", default_subplot_titles)
    if len(subplot_titles) != 8:
        raise ValueError("plot_data['subplot_titles'] must contain exactly eight panel titles.")

    panels = [
        (gs[0, 0], subplot_titles[4], s_data.get("psi_pred_data"), z_lims_psi),
        (gs[1, 0], subplot_titles[5], s_data.get("psi_ref_data"), z_lims_psi),
        (gs[0, 1], subplot_titles[6], s_data.get("j_tor_pred_data"), z_lims_j_tor),
        (gs[1, 1], subplot_titles[7], s_data.get("j_tor_ref_data"), z_lims_j_tor),
        (gs[0, 2], subplot_titles[0], g_data.get("lhs_pred_data"), z_lims_gs_sides),
        (gs[1, 2], subplot_titles[1], g_data.get("lhs_ref_data"), z_lims_gs_sides),
        (gs[0, 3], subplot_titles[2], g_data.get("rhs_pred_data"), z_lims_gs_sides),
        (gs[1, 3], subplot_titles[3], g_data.get("rhs_ref_data"), z_lims_gs_sides),
    ]

    for grid_spec, p_title, z_data, z_lims in panels:
        plot_gs_surface(
            base_fig_data={"fig": fig, "grid_data": plot_data["grid_data"], "grid_spec": grid_spec},
            surface_data={"surface_for_z_data": z_data, "z_lims": z_lims},
            view_data=view_data,
            plot_title=p_title,
            fail_silently=fail_silently,
        )

    # ..................................................................................................................
    if save_plots:
        if save_path is None:
            raise ValueError("save_path is required when save_plots is true.")
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(
            fname=save_path,
            transparent=None,
            dpi="figure",
            format=None,
            metadata=None,
            bbox_inches=None,
            pad_inches=0.1,
            facecolor="auto",
            edgecolor="auto",
            backend=None,
        )
    else:
        plt.show()

    plt.close()

    # ..................................................................................................................


# ----------------------------------------------------------------------------------------------------------------------
def make_signal_plots(
    plot_data,
    view_data=None,
    save_plots=False,
    save_dir="output",
    title="Reconstruction: true vs predicted",
    fail_silently: bool = True,
):
    """
    4-panel true-vs-predicted plot for psi and j_tor (loss-agnostic; works for weak/strong).

    plot_data = {
        "grid_data":   {"R_for_x_data": R, "Z_for_y_data": Z},
        "signal_data": {"psi_data": psi_pred, "psi_ref_data": psi_gt,
                        "j_tor_data": j_tor_pred, "j_tor_ref_data": j_tor_gt},
    }
    Each signal field is (n_r, n_z), matching R/Z.

    """

    if view_data is None:
        view_data = {"elev": 40, "azim": -49, "roll": 121}

    R = plot_data["grid_data"]["R_for_x_data"]
    Z = plot_data["grid_data"]["Z_for_y_data"]
    sig = plot_data["signal_data"]

    panels = [
        (sig.get("psi_data"), "Pred. psi"),
        (sig.get("psi_ref_data"), "Real psi"),
        (sig.get("j_tor_data"), "Pred. j_tor"),
        (sig.get("j_tor_ref_data"), "Real j_tor"),
    ]

    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(t=title, fontsize=16, y=0.99)
    gs = GridSpec(nrows=2, ncols=2, figure=fig)
    fig.subplots_adjust(hspace=0.25, wspace=0.15)
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for (z_field, subtitle), (r_, c_) in zip(panels, positions):
        ax = fig.add_subplot(gs[r_, c_], projection="3d")
        try:
            map_ax = plot_surface(
                ax=ax,
                plot_data={"x_data": R, "y_data": Z, "z_data": z_field},
                label_data={"x_label": "", "y_label": "", "z_label": "", "subtitle": subtitle},
                view_data=view_data,
            )
            fig.colorbar(map_ax, ax=ax)
        except Exception as ee:
            if fail_silently:
                pass
            else:
                print(f"[make_signal_plots] Plotting error: {ee}")

    if save_plots:
        os.makedirs(save_dir, exist_ok=True)
        date_tag = f"{datetime.now()}".replace("-", "").replace(":", "").replace(" ", "_")
        filename = f"gs_weak_loss_plots [{date_tag}].png"

        plt.savefig(
            fname=f"{save_dir}/{filename}",
            transparent=None,
            dpi="figure",
            format=None,
            metadata=None,
            bbox_inches=None,
            pad_inches=0.1,
            facecolor="auto",
            edgecolor="auto",
            backend=None,
        )
    else:
        plt.show()

    plt.close()

    # ..................................................................................................................
