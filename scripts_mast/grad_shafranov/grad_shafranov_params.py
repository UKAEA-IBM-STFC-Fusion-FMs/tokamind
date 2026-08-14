"""
grad_shafranov_params.py

TODO: Complete all docstrings
"""

from collections.abc import Callable
import numpy as np
import scipy.sparse as sp
import zarr
from freegs.gradshafranov import GSsparse4thOrder, mu0
from typing import Any
from scipy.sparse.linalg import factorized
from scipy.sparse import csr_matrix, coo_matrix
from dataclasses import dataclass
from skimage.draw import polygon
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch


# ----------------------------------------------------------------------------------------------------------------------
# Constants

GRAD_SHAFRANOV_PARAMS_FILE = "../assets/grad_shafranov/grad_shafranov_params_mast_coo.npz"

# MAST limiter shape
MAST_LIM_SHAPE_NUMPY_RZ = np.array(  # (r, z)
    [
        [1.9, 0.405],
        [1.5551043, 0.405],
        [1.5551043, 0.8225002],
        [1.4079306, 0.8225002],
        [1.4079306, 1.0330003],
        [1.039931, 1.0330003],
        [1.039931, 1.195],
        [1.9, 1.195],
        [1.9, 1.825],
        [0.5649307, 1.825],
        [0.5649307, 1.7280816],
        [0.7835, 1.7280816],
        [0.7835, 1.7155817],
        [0.5825903, 1.547],
        [0.4165, 1.547],
        [0.28, 1.6835],
        [0.28, 1.2290885],
        [0.1952444, 1.0835],
        [0.1952444, -1.0835],
        [0.28, -1.2290885],
        [0.28, -1.6835],
        [0.4165, -1.547],
        [0.5825903, -1.547],
        [0.7835, -1.7155817],
        [0.7835, -1.7280816],
        [0.5649307, -1.7280816],
        [0.5649307, -1.825],
        [1.9, -1.825],
        [1.9, -1.195],
        [1.039931, -1.195],
        [1.039931, -1.0330003],
        [1.4079306, -1.0330003],
        [1.4079306, -0.8225002],
        [1.5551043, -0.8225002],
        [1.5551043, -0.405],
        [1.9, -0.405],
        [1.9, 0.405],
    ]
)

# Base j_tor limiter shape
BASE_J_TOR_LIM_SHAPE_NUMPY_RZ = np.array(  # (r, z)
    [
        [1.9, 0.405],
        [1.5551043, 0.405],
        [1.5551043, 0.8225002],
        [1.4079306, 0.8225002],
        [1.4079306, 1.0330003],
        [1.039931, 1.0330003],
        [1.039931, 1.195],
        [1.0, 1.195],
        [1.0, 1.825],
        [0.5649307, 1.825],
        [0.5649307, 1.7280816],
        [0.7835, 1.7280816],
        [0.7835, 1.7155817],
        [0.5825903, 1.547],
        [0.4165, 1.547],
        [0.28, 1.6835],
        [0.28, 1.2290885],
        [0.1952444, 1.0835],
        [0.1952444, -1.0835],
        [0.28, -1.2290885],
        [0.28, -1.6835],
        [0.4165, -1.547],
        [0.5825903, -1.547],
        [0.7835, -1.7155817],
        [0.7835, -1.7280816],
        [0.5649307, -1.7280816],
        [0.5649307, -1.825],
        [1.0, -1.825],
        [1.0, -1.195],
        [1.039931, -1.195],
        [1.039931, -1.0330003],
        [1.4079306, -1.0330003],
        [1.4079306, -0.8225002],
        [1.5551043, -0.8225002],
        [1.5551043, -0.405],
        [1.9, -0.405],
        [1.9, 0.405],
    ]
)
BASE_J_TOR_LIM_SHAPE_NUMPY_RZ[:, 0] = np.clip(
    a=BASE_J_TOR_LIM_SHAPE_NUMPY_RZ[:, 0],
    a_min=0.3,
    a_max=1.4,
)
BASE_J_TOR_LIM_SHAPE_NUMPY_RZ[:, 1] = np.clip(
    a=BASE_J_TOR_LIM_SHAPE_NUMPY_RZ[:, 1],
    a_min=-1.0,
    a_max=1.0,
)


# ----------------------------------------------------------------------------------------------------------------------
def load_gs_relevant_data_from_mast_shot(shot_path: str) -> dict[str, Any]:
    """Load relevant MAST data for the Grad-Shafranov equation from one shot in Zarr format."""

    out_data: dict[str, Any] = {}

    zarr_store = zarr.open(shot_path)
    equilibrium_data = zarr_store["equilibrium"]
    equilibrium_keys = list(equilibrium_data.keys())

    # ..................................................................................................................
    # psi and j_tor relevant data

    if "major_radius" in equilibrium_keys:
        out_data["r_axis_vector"] = equilibrium_data["major_radius"][:]

    if "z" in equilibrium_keys:
        out_data["z_axis_vector"] = equilibrium_data["z"][:]

    if "time" in equilibrium_keys:
        out_data["time_values_vector"] = equilibrium_data["time"][:]  # -> [t_idx]

    if "j_tor" in equilibrium_keys:
        out_data["j_tor_matrix_over_time"] = equilibrium_data["j_tor"][:]  # -> [:, :, t_idx]

        # Valid time slices: no NaNs in j_tor
        is_j_tor_t_slice_valid_vector = ~np.any(np.isnan(out_data["j_tor_matrix_over_time"]), axis=(0, 1))
        out_data["valid_j_tor_t_slice_idxs_vector"] = np.where(is_j_tor_t_slice_valid_vector)  #  -> [valid_t_idx]

    if "psi" in equilibrium_keys:
        out_data["psi_matrix_over_time"] = equilibrium_data["psi"][:]  # -> [:, :, time_idx]

        # Valid time slices: no NaNs in psi map
        is_psi_t_slice_valid_vector = ~np.any(np.isnan(out_data["psi_matrix_over_time"]), axis=(0, 1))
        out_data["valid_psi_t_slice_idxs_vector"] = np.where(is_psi_t_slice_valid_vector)  #  -> [valid_t_idx]

    if "lcfs_r" in equilibrium_keys:
        out_data["lcfs_r_vector_over_time"] = equilibrium_data["lcfs_r"][:]  # -> [:, time_idx]

    if "lcfs_z" in equilibrium_keys:
        out_data["lcfs_z_vector_over_time"] = equilibrium_data["lcfs_z"][:]  # -> [:, time_idx]

    # ..................................................................................................................
    # pprime and ffprime related

    if "dpressure_dpsi" in equilibrium_keys:
        out_data["pprime_vector_over_time"] = equilibrium_data["dpressure_dpsi"][:]  # -> [:, time_idx]

        # Valid time slices: no NaNs in pprime
        is_pprime_t_slice_valid_vector = ~np.any(np.isnan(out_data["pprime_vector_over_time"]), axis=(0))
        out_data["valid_pprime_t_slice_idxs_vector"] = np.where(is_pprime_t_slice_valid_vector)  #  -> [valid_t_idx]

    if "f_df_dpsi" in equilibrium_keys:
        out_data["ffprime_vector_over_time"] = equilibrium_data["f_df_dpsi"][:]  # -> [:, time_idx]

        # Valid time slices: no NaNs in ffprime
        is_ffprime_t_slice_valid_vector = ~np.any(np.isnan(out_data["ffprime_vector_over_time"]), axis=(0))
        out_data["valid_ffprime_t_slice_idxs_vector"] = np.where(is_ffprime_t_slice_valid_vector)  #  -> [valid_t_idx]

    if "magnetic_axis_r" in equilibrium_keys:
        out_data["mag_axis_r"] = equilibrium_data["magnetic_axis_r"][:]  # -> [t_idx]

    if "magnetic_axis_z" in equilibrium_data:
        out_data["mag_axis_z"] = equilibrium_data["magnetic_axis_z"][:]  # -> [t_idx]

    if "x_point_r" in equilibrium_keys:
        out_data["x_point_r"] = equilibrium_data["x_point_r"][:]  # -> [:, t_idx]

    if "x_point_z" in equilibrium_keys:
        out_data["x_point_z"] = equilibrium_data["x_point_z"][:]  # -> [:, t_idx]

    return out_data


# ======================================================================================================================
@dataclass(frozen=True)
class GSContext:
    """
    Precomputed GS solver context for a fixed (nR, nZ) grid.

    Attributes
    ----------
    r_axis_vector, r_axis_vector : np.ndarray
        1D equilibrium grids.
    GS_op_csr_matrix, GS_op_raw_csr_matrix : csr_matrix
        4th-order Grad-Shafranov (sparse) matrix with Dirichlet boundary rows.
    GS_solver_raw, GS_solver : Mapping[str, callable]
        LU-factorized solvers for raw and fixed operators, produced by scipy.sparse.linalg.factorized.
    R_matrix, Z_matrix : np.ndarray
        (n_r, n_z) R-grid used in the RHS (-mu0 * R_solver_matrix * j_tor), and (n_r, n_z) Z-grid.
    boundary_matrix, boundary_flat : np.ndarray
        Boundary masks in (nR, nZ) and flattened layout.
    dr, dz : float
        Uniform spacings.
    n_r, n_z : int
        Grid sizes.

    """

    r_axis_vector: np.ndarray
    z_axis_vector: np.ndarray

    GS_op_csr_matrix: csr_matrix
    GS_op_raw_csr_matrix: csr_matrix

    GS_solver: Callable
    GS_solver_raw: Callable

    R_matrix: np.ndarray
    Z_matrix: np.ndarray

    boundary_matrix: np.ndarray
    boundary_flat: np.ndarray

    dr: float
    dz: float
    n_r: int
    n_z: int


# ----------------------------------------------------------------------------------------------------------------------
def build_GS_operator(
    n_r,
    n_z,
    r_axis_min,
    r_axis_max,
    z_axis_min,
    z_axis_max,
):
    """
    Build the sparse Grad-Shafranov operator on the fixed R/Z grid.

    TODO: Complete docstrings

    """

    # Preliminaries
    boundary_rz = np.zeros(shape=(n_r, n_z), dtype=bool)
    boundary_rz[0, :] = True
    boundary_rz[-1, :] = True
    boundary_rz[:, 0] = True
    boundary_rz[:, -1] = True

    # Raw operator
    raw_operator = GSsparse4thOrder(
        Rmin=r_axis_min,
        Rmax=r_axis_max,
        Zmin=z_axis_min,
        Zmax=z_axis_max,
    )
    raw_operator = raw_operator(n_r, n_z).tocsr()

    # Fixed operator
    fixed_operator, bad_rows_idxs = ensure_identity_boundary_rows(
        base_operator_matrix=raw_operator,
        boundary_matrix=boundary_rz,
    )

    return raw_operator, boundary_rz, fixed_operator, bad_rows_idxs


# ----------------------------------------------------------------------------------------------------------------------
def build_GS_context(r_axis_vector: np.ndarray, z_axis_vector: np.ndarray) -> GSContext:
    """
    Assemble and factorize the 4th-order GS sparse operator once.

    This is the expensive setup step. Reuse the returned context across all time slices and synthetic samples on the
    same grid.

    """

    # Preliminaries

    n_r = len(r_axis_vector)
    r_axis_vector_min = r_axis_vector.min()
    r_axis_vector_max = r_axis_vector.max()
    dr = (r_axis_vector.max() - r_axis_vector.min()) / (n_r - 1)

    n_z = len(z_axis_vector)
    z_axis_vector_min = z_axis_vector.min()
    z_axis_vector_max = z_axis_vector.max()
    dz = (z_axis_vector.max() - z_axis_vector.min()) / (n_z - 1)

    # Grad-Shafranov operators
    GS_op_raw_csr, boundary_matrix, GS_op_fixed_csr, bad_rows_idxs = build_GS_operator(
        n_r=n_r,
        n_z=n_z,
        r_axis_min=r_axis_vector_min,
        r_axis_max=r_axis_vector_max,
        z_axis_min=z_axis_vector_min,
        z_axis_max=z_axis_vector_max,
    )

    # Grad-Shafranov solvers
    GS_solver_raw = factorized(GS_op_raw_csr)
    GS_solver = factorized(GS_op_fixed_csr)

    # R field for RHS term (-mu0 * R_solver_matrix * j_tor) and Z field.
    Z_matrix, R_matrix = np.meshgrid(z_axis_vector, r_axis_vector)

    return GSContext(
        r_axis_vector=r_axis_vector,
        z_axis_vector=z_axis_vector,
        GS_op_raw_csr_matrix=GS_op_raw_csr,
        GS_op_csr_matrix=GS_op_fixed_csr,
        GS_solver_raw=GS_solver_raw,
        GS_solver=GS_solver,
        R_matrix=R_matrix,
        Z_matrix=Z_matrix,
        boundary_matrix=boundary_matrix,
        boundary_flat=boundary_matrix.flatten(),
        dr=dr,
        dz=dz,
        n_r=len(r_axis_vector),
        n_z=len(z_axis_vector),
    )


# ----------------------------------------------------------------------------------------------------------------------
def ensure_identity_boundary_rows(base_operator_matrix: csr_matrix, boundary_matrix: np.ndarray):
    """
    Return processed operator with identity rows on the rectangular Dirichlet boundary.

    The projection loss sets rhs[boundary] = psi_hat[boundary]. That imposes psi_GS = psi_hat on the boundary only if
    those rows of the operator are identity rows. FreeGS currently uses this convention, but we check it explicitly and
    repair the operator if a future operator does not. TODO: Ask Tobia the origin of this docstring.

    """

    boundary_idxs = np.flatnonzero(boundary_matrix)
    bad_rows_idxs = []

    for idx in boundary_idxs:
        row = base_operator_matrix.getrow(int(idx))
        is_identity = (row.nnz == 1) and (row.indices[0] == idx) and np.isclose(row.data[0], 1.0, rtol=0.0, atol=1e-12)
        if not is_identity:
            bad_rows_idxs.append(int(idx))

    raw_operator_lil = base_operator_matrix.tolil(copy=True)  # LiL: List of lists
    for idx in boundary_idxs:
        raw_operator_lil.rows[int(idx)] = [int(idx)]
        raw_operator_lil.data[int(idx)] = [1.0]

    fixed_operator_matrix = raw_operator_lil.tocsr()

    return fixed_operator_matrix, bad_rows_idxs


# ----------------------------------------------------------------------------------------------------------------------
def save_sparse_array_via_save_npz(sparse_array: np.ndarray, filename: str) -> None:
    """
    Save sparse matrices (e.g., crs, coo) to a file in a single step.

    REMARK: It does not work for dense matrices.

    Parameters
    ----------
    sparse_array : np.ndarray
        Array data to be saved.
    filename : str
        Target filename. It adds the ".npz" extension automatically if not provided.

    Returns
    -------
    None

    """

    sp.save_npz(file=filename, matrix=sparse_array)


# ----------------------------------------------------------------------------------------------------------------------
def save_arrays_via_savez_compressed(filename: str, **kwargs) -> None:
    """
    Examples
    --------

    Include data from csr matrix:
        save_arrays_via_savez_compressed(
            filename=filename,
            csr_matrix_data=sparse_csr_matrix.data,
            csr_matrix_indices=sparse_csr_matrix.indices,
            csr_matrix_indptr=sparse_csr_matrix.indptr,
            csr_matrix_shape=sparse_csr_matrix.shape,
            dense_matrix=dense_matrix,
            dense_vector=dense_vector,
            n=n,
            r=r,
        )

        Include data from coo matrix:
        save_arrays_via_savez_compressed(
            filename=filename,
            coo_matrix_data=AA_coo.data,
            coo_matrix_row=AA_coo.row,
            coo_matrix_col=AA_coo.col,
            coo_matrix_shape=AA_coo.shape,
            dense_matrix=dense_matrix,
            dense_vector=dense_vector,
            n=n,
            r=r,
        )

    Parameters
    ----------
    filename : str
        Target filename.
    kwargs
        Specified kwargs.

    Returns
    -------

    """
    np.savez_compressed(file=filename, **kwargs)


# ----------------------------------------------------------------------------------------------------------------------
def load_arrays_from_savez_compressed(filename: str) -> dict[str, Any]:

    loaded_data = np.load(filename)

    return loaded_data


# ----------------------------------------------------------------------------------------------------------------------
def get_sparse_csr_array_components(
    sparse_csr_array: csr_matrix,
    prefix: str = "",
    suffix: str = "",
) -> dict:

    if prefix:
        prefix = f"{prefix}_"

    if suffix:
        suffix = f"_{suffix}"

    components_dict = {
        f"{prefix}csr_data{suffix}": sparse_csr_array.data,
        f"{prefix}csr_indices{suffix}": sparse_csr_array.indices,
        f"{prefix}csr_indptr{suffix}": sparse_csr_array.indptr,
        f"{prefix}csr_shape{suffix}": sparse_csr_array.shape,
    }

    return components_dict


# ----------------------------------------------------------------------------------------------------------------------
def get_sparse_coo_array_components(
    sparse_coo_array: coo_matrix,
    prefix: str = "",
    suffix: str = "",
):

    if prefix:
        prefix = f"{prefix}_"

    if suffix:
        suffix = f"_{suffix}"

    components_dict = {
        f"{prefix}coo_data{suffix}": sparse_coo_array.data,
        f"{prefix}coo_row{suffix}": sparse_coo_array.row,
        f"{prefix}coo_col{suffix}": sparse_coo_array.col,
        f"{prefix}coo_shape{suffix}": sparse_coo_array.shape,
    }

    return components_dict


# ----------------------------------------------------------------------------------------------------------------------
def build_sparse_csr_matrix_from_components(
    components_dict: dict,
    as_torch_tensor: bool = False,
):
    sparse_csr_matrix = sp.csr_matrix(
        (components_dict["data"], components_dict["indices"], components_dict["indptr"]),
        shape=components_dict["shape"],
    )

    if as_torch_tensor:
        sparse_csr_matrix = sparse_crs_to_tensor(sparse_csr_matrix=sparse_csr_matrix)

    return sparse_csr_matrix


# ----------------------------------------------------------------------------------------------------------------------
def sparse_crs_to_tensor(sparse_csr_matrix):
    sparse_coo_matrix = sparse_csr_matrix.tocoo()
    sparse_coo_matrix_tensor = sparse_coo_to_tensor(sparse_coo_matrix=sparse_coo_matrix)
    sparse_csr_matrix_tensor = sparse_coo_matrix_tensor.to_sparse_csr()

    return sparse_csr_matrix_tensor


# ----------------------------------------------------------------------------------------------------------------------
def sparse_coo_to_tensor(sparse_coo_matrix):
    torch.sparse.check_sparse_tensor_invariants.disable()

    sparse_coo_matrix_tensor = torch.sparse_coo_tensor(
        indices=torch.LongTensor(np.array(sparse_coo_matrix.nonzero())),
        values=torch.FloatTensor(sparse_coo_matrix.data),
        size=torch.Size(sparse_coo_matrix.shape),
    ).coalesce()

    return sparse_coo_matrix_tensor


# ----------------------------------------------------------------------------------------------------------------------
def build_sparse_coo_array_from_components(components_dict: dict, as_torch_tensor: bool = False):
    sparse_coo_matrix = sp.coo_matrix(
        (components_dict["data"], (components_dict["row"], components_dict["col"])), shape=components_dict["shape"]
    )

    if as_torch_tensor:
        sparse_coo_matrix = sparse_coo_to_tensor(sparse_coo_matrix=sparse_coo_matrix)

    return sparse_coo_matrix


# ----------------------------------------------------------------------------------------------------------------------
def compute_limiter_mask(  # NOSONAR
    R_matrix: np.ndarray,  # noqa
    Z_matrix: np.ndarray,  # noqa
    limiter_shape: np.ndarray,
    min_value: float = 0,  # 5e-2,
    plot_shape_and_mask=False,
):
    # ..................................................................................................................
    def convert_coord_index(R_mat, Z_mat, points):

        indices_arr = []
        for point in points:
            x, y = point
            idx_x, idx_y = 0, 0

            for idx in range(R_mat.shape[0] - 1):
                if R_mat[idx, 0] <= x < R_mat[idx + 1, 0]:
                    idx_x = idx
                    break

            for idx in range(R_mat.shape[1] - 1):
                if Z_mat[0, idx] <= y < Z_mat[0, idx + 1]:
                    idx_y = idx
                    break

            indices_arr.append([idx_x, idx_y])

        return np.array(indices_arr)

    # ..................................................................................................................

    limiter_mask_rz = (
        np.ones_like(
            a=R_matrix,
            dtype=np.float32,
        )
        * min_value
    )
    contour = convert_coord_index(R_mat=R_matrix, Z_mat=Z_matrix, points=limiter_shape)

    # Create an empty image to store the masked array
    rr, cc = polygon(contour[:, 0], contour[:, 1], R_matrix.shape)
    limiter_mask_rz[rr, cc] = 1

    if plot_shape_and_mask:
        fig_ = plt.figure(figsize=(6, 6))
        fig_.suptitle(t="Limiter mask and shape", fontsize=16, y=0.98)
        gs_ = GridSpec(nrows=1, ncols=1)

        ax_ = fig_.add_subplot(gs_[:, 0], projection="3d")
        ax_.plot_surface(X=R_matrix, Y=Z_matrix, Z=limiter_mask_rz, cmap="autumn_r", lw=0.5, rstride=1, cstride=1)
        ax_.plot(xs=limiter_shape[:, 0], ys=limiter_shape[:, 1], zs=-0.1, zdir="z", lw=3, color="0.0")
        plt.show()

    return limiter_mask_rz


# ======================================================================================================================
if __name__ == "__main__":
    shot_path_ = "/mast/tokamark/v1/30421.zarr"  # 30421.zarr, 30471.zarr

    # ------------------------------------------------------------------------------------------------------------------
    # Load relevant MAST data for Grad-Shafranov equation

    GS_mast_data = load_gs_relevant_data_from_mast_shot(shot_path=shot_path_)
    # print(gs_mast_data.keys())

    # ------------------------------------------------------------------------------------------------------------------
    # Time references

    # T_REF = 0.35
    # time_vector = GS_mast_data["time_values_vector"]
    # ref_time_index = int(np.argmin(np.abs(time_vector - T_REF)))

    # ------------------------------------------------------------------------------------------------------------------
    # Build Grad-Shafranov context

    GS_ctx = build_GS_context(
        r_axis_vector=GS_mast_data["r_axis_vector"],
        z_axis_vector=GS_mast_data["z_axis_vector"],
    )
    # print(GS_ctx.GS_op_csr_matrix)
    # print(GS_ctx.R_matrix)

    # ------------------------------------------------------------------------------------------------------------------
    # Build limiter masks

    MAST_LIM_MASK_RZ = compute_limiter_mask(
        R_matrix=GS_ctx.R_matrix,
        Z_matrix=GS_ctx.Z_matrix,
        limiter_shape=MAST_LIM_SHAPE_NUMPY_RZ,
        # plot_shape_and_mask=True,
    )

    BASE_J_TOR_LIM_MASK_RZ = compute_limiter_mask(
        R_matrix=GS_ctx.R_matrix,
        Z_matrix=GS_ctx.Z_matrix,
        limiter_shape=BASE_J_TOR_LIM_SHAPE_NUMPY_RZ,
        plot_shape_and_mask=False,
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Save data

    arrays_to_save = get_sparse_coo_array_components(
        sparse_coo_array=GS_ctx.GS_op_csr_matrix.tocoo(),
        prefix="GS_op",
    )

    # print(GS_ctx.r_axis_vector)
    # raise

    arrays_to_save.update(
        {
            "n_r": GS_ctx.n_r,
            "n_z": GS_ctx.n_z,
            "R_matrix": GS_ctx.R_matrix,
            "Z_matrix": GS_ctx.Z_matrix,
            "r_axis_vector": GS_ctx.r_axis_vector,
            "z_axis_vector": GS_ctx.z_axis_vector,
            "mu0": mu0,
            "MAST_lim_mask_rz": MAST_LIM_MASK_RZ,
            "base_j_tor_lim_mask_rz": BASE_J_TOR_LIM_MASK_RZ,
        }
    )

    save_arrays_via_savez_compressed(
        filename=GRAD_SHAFRANOV_PARAMS_FILE,
        **arrays_to_save,
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Load data

    loaded_data = load_arrays_from_savez_compressed(filename=GRAD_SHAFRANOV_PARAMS_FILE)
    print(list(loaded_data.keys()))

    # ------------------------------------------------------------------------------------------------------------------
    # Recover params from loaded data

    GS_op_sparse_coo_tensor = build_sparse_coo_array_from_components(
        components_dict={
            "data": loaded_data["GS_op_coo_data"],
            "row": loaded_data["GS_op_coo_row"],
            "col": loaded_data["GS_op_coo_col"],
            "shape": loaded_data["GS_op_coo_shape"],
        },
        as_torch_tensor=True,
    )

    n_r_loaded = loaded_data["n_r"]
    n_z_loaded = loaded_data["n_z"]

    R_matrix_loaded = torch.from_numpy(loaded_data["R_matrix"])
    Z_matrix_loaded = torch.from_numpy(loaded_data["Z_matrix"])

    # ------------------------------------------------------------------------------------------------------------------
    # Play with data

    tt = 20
    j_tor_t = torch.Tensor(GS_mast_data["j_tor_matrix_over_time"][:, :, tt])
    psi_t = torch.Tensor(GS_mast_data["psi_matrix_over_time"][:, :, tt])

    numerator = torch.matmul(-GS_op_sparse_coo_tensor, psi_t.ravel()).to_dense().reshape(n_r_loaded, n_z_loaded)

    denominator = mu0 * R_matrix_loaded.T

    j_tor_rec = numerator / denominator
    j_tor_rec = np.clip(j_tor_rec, 0, float(np.nanmax(j_tor_rec)))

    print(j_tor_t.max())
    print(j_tor_rec.max())
