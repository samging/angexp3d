"""Angle-driven manifold compression with power-law and Fourier-Power models.

Shared implementation of the transformers that mainfoldByAngle.ipynb used to
re-declare in every cell (AngleExponent3D/4D/ND and AngleExponentFourier*):

    1. rotate the point cloud with an orthogonal matrix R,
    2. keep the first two rotated coordinates as a 2D plane,
    3. try both plane axes as the stored driver x' and fit
           power model         y~ = a * x~^(p+1)
           Fourier-Power model y~ = a * x~^(p+1) + A * sin(omega * x~ + phi)
       on the positively shifted coordinates (x~ = x' + s_x, y~ = y' + s_y),
    4. keep the axis with the lower in-plane MSE, store only x' and the
       parameters, and regenerate y' on the fly during reconstruction.

The remaining rotated coordinates (index 2 and up) are the depth priors the
decoder receives as side information.

CurveCodec is the self-contained alternative: it stores one latent value t per
point and one Fourier-Power law per ambient dimension, so the decoder needs
nothing else. pca_codec is the linear baseline it is compared against.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

SEED = 42
POWER, FOURIER = "power", "fourier"

# Initial (A, omega, phi) guesses used by the notebook experiments
P0_3D = (0.01, 1.0, 0.0)
P0_ND = (1.0, 2.0, 0.0)
P0_EDGE = (0.1, 1.0, 0.0)


# =============================================================
# MODELS
# =============================================================
def power_model(x, a, p_plus_1):
    """Pure power law y = a * x^(p+1)."""
    return a * (x ** p_plus_1)


def fourier_power_model(x, a, p_plus_1, A, omega, phi):
    """Power-law trend + harmonic sine wave oscillation."""
    return a * (x ** p_plus_1) + A * np.sin(omega * x + phi)


# =============================================================
# ROTATIONS
# =============================================================
def _plane_rotation(dim: int, i: int, j: int, angle: float) -> np.ndarray:
    """Givens rotation by `angle` (radians) in the (i, j) coordinate plane."""
    G = np.eye(dim)
    c, s = np.cos(angle), np.sin(angle)
    G[i, i], G[i, j], G[j, i], G[j, j] = c, -s, s, c
    return G


def rotation_3d(theta_x_deg: float = 30.0, theta_y_deg: float = 45.0) -> np.ndarray:
    """R = Ry(theta_y) @ Rx(theta_x)."""
    Rx = _plane_rotation(3, 1, 2, np.radians(theta_x_deg))
    Ry = _plane_rotation(3, 2, 0, np.radians(theta_y_deg))
    return Ry @ Rx


def rotation_4d(angles_deg: tuple[float, ...] = (15.0, 30.0, 45.0, 10.0, 20.0, 35.0)) -> np.ndarray:
    """Product of the six 4D Givens rotations, angles ordered (XY, XZ, XW, YZ, YW, ZW).

    Applied as R_zw @ R_yw @ R_yz @ R_xw @ R_xz @ R_xy.
    """
    planes = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    R = np.eye(4)
    for (i, j), angle in zip(planes, angles_deg):
        R = _plane_rotation(4, i, j, np.radians(angle)) @ R
    return R


def random_rotation(dim: int, seed: int = SEED) -> np.ndarray:
    """Random orthonormal matrix in O(dim) from the QR factor of a Gaussian matrix.

    Uses a private RandomState, which draws the same numbers as
    np.random.seed(seed) followed by np.random.normal without touching global state.
    """
    G = np.random.RandomState(seed).normal(0, 1, (dim, dim))
    Q, _ = np.linalg.qr(G)
    return Q


# =============================================================
# TRANSFORMER
# =============================================================
@dataclass
class AxisFit:
    """Fit of one candidate driver axis."""
    driver_idx: int
    mse: float
    a_fit: float
    exponent: float
    fourier: tuple[float, float, float]
    x_raw: np.ndarray = field(repr=False)
    y_raw: np.ndarray = field(repr=False)
    shift: np.ndarray = field(repr=False)

    @property
    def params(self) -> tuple[float, ...]:
        return (self.a_fit, self.exponent, *self.fourier)


class AngleExponentTransformer:
    """Compresses an (n, D) point cloud to a 1D driver plus a handful of scalars.

    model="power" reproduces AngleExponent{3D,4D,ND}Transformer,
    model="fourier" reproduces AngleExponentFourier*Transformer.
    """

    def __init__(self, R: np.ndarray, model: str = FOURIER,
                 fourier_p0: tuple[float, float, float] = P0_ND, maxfev: int = 5000):
        if model not in (POWER, FOURIER):
            raise ValueError(f"model must be {POWER!r} or {FOURIER!r}, got {model!r}")
        self.R = np.asarray(R, dtype=float)
        self.R_inv = self.R.T  # R is orthogonal
        self.model = model
        self.fourier_p0 = fourier_p0
        self.maxfev = maxfev
        self.axis_fits: dict[int, AxisFit] = {}
        self.best: AxisFit | None = None

    # ---- constructors -------------------------------------------------
    @classmethod
    def from_angles(cls, theta_x_deg: float = 30.0, theta_y_deg: float = 45.0, **kw):
        return cls(rotation_3d(theta_x_deg, theta_y_deg), **kw)

    @classmethod
    def from_givens_4d(cls, angles_deg=(15.0, 30.0, 45.0, 10.0, 20.0, 35.0), **kw):
        return cls(rotation_4d(angles_deg), **kw)

    @classmethod
    def random(cls, target_dim: int, seed: int = SEED, **kw):
        return cls(random_rotation(target_dim, seed), **kw)

    # ---- notebook-compatible attributes -------------------------------
    @property
    def x_1d(self):
        return None if self.best is None else self.best.x_raw

    @property
    def driver_axis_idx(self) -> int:
        return 0 if self.best is None else self.best.driver_idx

    @property
    def computed_exponent(self):
        return None if self.best is None else self.best.exponent

    @property
    def a_fit(self):
        return None if self.best is None else self.best.a_fit

    @property
    def fourier_params(self):
        return None if self.best is None else self.best.fourier

    @property
    def shift(self):
        return None if self.best is None else self.best.shift

    @property
    def axis_errors(self) -> dict[int, dict]:
        return {k: {"mse": f.mse, "exponent": f.exponent, "a_fit": f.a_fit, "fourier": f.fourier}
                for k, f in self.axis_fits.items()}

    @property
    def param_bytes(self) -> int:
        """Bytes counted for the stored scalars: 2 (power) or 5 (Fourier-Power) float32s."""
        return 8 if self.model == POWER else 5 * 4

    # ---- core ----------------------------------------------------------
    def rotate(self, points: np.ndarray) -> np.ndarray:
        return points @ self.R.T

    def depth_priors(self, points: np.ndarray) -> np.ndarray:
        """Rotated coordinates 2..D-1, the side information used by `reconstruct`."""
        return self.rotate(points)[:, 2:]

    def _fit_axis(self, points_2d: np.ndarray, driver_idx: int) -> AxisFit:
        x = points_2d[:, driver_idx]
        y = points_2d[:, 1 - driver_idx]

        # Shift into the positive domain so log() and real powers are defined
        shift = np.array([np.abs(x.min()) + 0.1, np.abs(y.min()) + 0.1])
        xs, ys = x + shift[0], y + shift[1]

        # 45 degree slope multipliers m = x / y, log-linear fit: log m = -log a - p log x
        m = xs / np.maximum(ys, 1e-6)
        slope, intercept = np.polyfit(np.log(xs), np.log(m), 1)
        a_fit, exponent = np.exp(-intercept), 1.0 - slope
        fourier = (0.0, 0.0, 0.0)

        if self.model == FOURIER:
            try:
                popt, _ = curve_fit(fourier_power_model, xs, ys,
                                    p0=[a_fit, exponent, *self.fourier_p0], maxfev=self.maxfev)
                if np.all(np.isfinite(popt)):
                    a_fit, exponent, *fourier = popt
                    fourier = tuple(fourier)
            except (RuntimeError, ValueError):
                pass  # keep the pure power-law fit

        y_hat = fourier_power_model(xs, a_fit, exponent, *fourier) - shift[1]
        mse = float(np.mean((y - y_hat) ** 2))
        return AxisFit(driver_idx, mse, float(a_fit), float(exponent),
                       tuple(float(v) for v in fourier), x, y, shift)

    def project_and_compress(self, points: np.ndarray) -> np.ndarray:
        """Fits both candidate driver axes, keeps the better one and returns the 1D x' driver."""
        points_2d = self.rotate(points)[:, :2]
        self.axis_fits = {idx: self._fit_axis(points_2d, idx) for idx in (0, 1)}
        self.best = min(self.axis_fits.values(), key=lambda f: f.mse)
        return self.x_1d

    def regenerate_y(self) -> np.ndarray:
        """y'_hat = f(x' + s_x) - s_y, computed from the stored driver and parameters."""
        if self.best is None:
            raise ValueError("Run `project_and_compress` before reconstruction.")
        b = self.best
        return fourier_power_model(b.x_raw + b.shift[0], *b.params) - b.shift[1]

    def reconstruct(self, depth_priors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (reconstructed points, reconstructed 2D plane)."""
        y_hat = self.regenerate_y()
        cols = [self.x_1d, y_hat] if self.driver_axis_idx == 0 else [y_hat, self.x_1d]
        plane = np.column_stack(cols)
        priors = np.asarray(depth_priors).reshape(len(plane), -1)
        return np.column_stack([plane, priors]) @ self.R_inv.T, plane

    # Names used by the original notebook classes
    def reconstruct_3d(self, z_topology_priors):
        return self.reconstruct(z_topology_priors)

    def reconstruct_4d(self, topology_priors_2d):
        return self.reconstruct(topology_priors_2d)

    def reconstruct_nd(self, depth_priors_nd):
        return self.reconstruct(depth_priors_nd)


# =============================================================
# DATASETS
# =============================================================
def egg_3d(num_u: int = 30, num_v: int = 30, a: float = 1.0, b: float = 1.5, k: float = 0.2):
    """Egg surface used by the first power-law experiment."""
    U, V = np.meshgrid(np.linspace(0, np.pi, num_u), np.linspace(0, 2 * np.pi, num_v))
    r = a * np.sin(U) * (1 + k * np.cos(U))
    return np.column_stack([(r * np.cos(V)).ravel(), (r * np.sin(V)).ravel(), (b * np.cos(U)).ravel()])


def square_3d_mesh(side_length: float = 2.0, num_points_per_side: int = 30):
    """Flat square in the z = 0 plane."""
    g = np.linspace(-side_length / 2, side_length / 2, num_points_per_side)
    X, Y = np.meshgrid(g, g)
    return np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])


def cube_3d_point_cloud(side_length: float = 2.0, num_samples: int = 1000, seed: int = SEED):
    """Cube surface: each (u, v) sample is placed on all six faces."""
    c = side_length / 2
    uv = np.random.RandomState(seed).uniform(-c, c, (num_samples // 6, 2))
    u, v = uv[:, 0], uv[:, 1]
    full = np.full_like(u, c)
    faces = [(full, u, v), (-full, u, v), (u, full, v), (u, -full, v), (u, v, full), (u, v, -full)]
    return np.stack([np.column_stack(f) for f in faces], axis=1).reshape(-1, 3)


def sphere_3d_mesh(radius: float = 1.0, num_points: int = 1000):
    """Fibonacci-lattice sphere."""
    idx = np.arange(num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * idx / num_points)
    theta = np.pi * (1 + 5 ** 0.5) * idx
    return radius * np.column_stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])


def hypersphere_4d(radius: float = 1.0, num_samples: int = 1000, seed: int = SEED):
    """Uniform points on the 3-sphere x^2 + y^2 + z^2 + w^2 = r^2."""
    raw = np.random.RandomState(seed).normal(0, 1, (num_samples, 4))
    return radius * raw / np.linalg.norm(raw, axis=1, keepdims=True)


# (phase step, base power, power step, cosine amplitude, cosine frequency) per dimension
_SINUSOIDAL_CONFIG = {
    20: (np.pi / 10.0, 1.2, 0.05, 0.1, 2),
    100: (np.pi / 50.0, 1.1, 0.01, 0.05, 3),
    600: (np.pi / 300.0, 1.05, 0.001, 0.02, 4),
}


def sinusoidal_manifold(dim: int = 20, num_samples: int = 1500):
    """Phase-shifted multi-power waves: v_d(t) = sin(t + d*dphi) |t|^(p0 + d*dp) + eps cos(k t)."""
    phase_step, p0, p_step, amp, freq = _SINUSOIDAL_CONFIG[dim]
    t = np.linspace(-3.0, 3.0, num_samples)[:, None]
    d = np.arange(dim)[None, :]
    return np.sin(t + d * phase_step) * np.abs(t) ** (p0 + d * p_step) + amp * np.cos(freq * t)


def plane_mesh(side_length: float = 2.0, num_points_per_side: int = 30):
    """Tilted plane z = 0.5x - 0.8y + 1.2."""
    g = np.linspace(-side_length / 2, side_length / 2, num_points_per_side)
    X, Y = np.meshgrid(g, g)
    return np.column_stack([X.ravel(), Y.ravel(), (0.5 * X - 0.8 * Y + 1.2).ravel()])


def cad_cube(points_per_face: int = 250, seed: int = SEED):
    """Hollow cube with independent uniform samples on each of the six faces."""
    rng = np.random.RandomState(seed)
    faces = []
    for axis in (2, 0, 1):  # front/back, left/right, top/bottom
        for value in (-1.0, 1.0):
            a, b = rng.uniform(-1, 1, points_per_face), rng.uniform(-1, 1, points_per_face)
            face = np.insert(np.column_stack([a, b]), axis, value, axis=1)
            faces.append(face)
    return np.vstack(faces)


def gaussian_noise(dim: int = 100, num_samples: int = 1500, seed: int = SEED):
    """Uncorrelated N(0, 1) noise."""
    return np.random.RandomState(seed).normal(0, 1, (num_samples, dim))


def cad_paraboloid(num_radius: int = 30, num_angles: int = 30):
    """Paraboloid dish z = 0.5x^2 + 0.3y^2 on a polar grid."""
    Rg, Tg = np.meshgrid(np.linspace(0.1, 2.0, num_radius), np.linspace(0, 2 * np.pi, num_angles))
    X, Y = Rg * np.cos(Tg), Rg * np.sin(Tg)
    return np.column_stack([X.ravel(), Y.ravel(), (0.5 * X ** 2 + 0.3 * Y ** 2).ravel()])


def multifrequency_curve(dim: int = 100, num_samples: int = 1500):
    """1D curve whose coordinates oscillate at different rates, so it spans many linear dimensions.

    v_d(t) = 0.5 t^(1 + 0.5 d/D) + sin(w_d t + 2 pi d / D),  w_d from 0.5 to 4,  t in [0, 3].
    """
    t = np.linspace(0.0, 3.0, num_samples)[:, None]
    d = np.arange(dim)[None, :]
    omega = 0.5 + 3.5 * d / max(dim - 1, 1)
    return 0.5 * t ** (1.0 + 0.5 * d / dim) + np.sin(omega * t + 2 * np.pi * d / dim)


def hugelschaffer_egg(a: float = 1.0, b: float = 0.7, w: float = 0.2, num_u: int = 30, num_v: int = 30):
    """Parametric Hügelschäffer egg."""
    U, V = np.meshgrid(np.linspace(-np.pi / 2, np.pi / 2, num_u), np.linspace(0, 2 * np.pi, num_v))
    s = b * (1.0 - w * np.sin(U)) * np.sin(U)
    return np.column_stack([(a * np.cos(U)).ravel(), (s * np.cos(V)).ravel(), (s * np.sin(V)).ravel()])


# =============================================================
# CURVE CODEC (no side information)
# =============================================================
def fourier_power_series(x, a, p_plus_1, omega, *harmonics):
    """a * x^(p+1) + sum_j A_j sin(j * omega * x + phi_j), harmonics = (A_1, phi_1, A_2, phi_2, ...)."""
    y = a * (x ** p_plus_1)
    for j in range(len(harmonics) // 2):
        A, phi = harmonics[2 * j], harmonics[2 * j + 1]
        y = y + A * np.sin((j + 1) * omega * x + phi)
    return y


def arc_length_latent(points: np.ndarray) -> np.ndarray:
    """Cumulative arc length along the sample order (assumes points are ordered along the curve)."""
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def pca_latent(points: np.ndarray) -> np.ndarray:
    """Score on the first principal component (order-free, but folds curves that turn back)."""
    centred = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return centred @ vt[0]


LATENTS = {"arc": arc_length_latent, "pca": pca_latent}


@dataclass
class CurveCode:
    """Everything the decoder needs: the latent per point and one law per dimension."""
    t: np.ndarray            # (n,) shifted latent, t > 0
    params: np.ndarray       # (D, 3 + 2K) rows of (a, p+1, omega, A_1, phi_1, ...)
    shifts: np.ndarray       # (D,) positive-domain shift of each dimension

    def nbytes(self, value_bytes: int = 8) -> int:
        return value_bytes * (self.t.size + self.params.size + self.shifts.size)


class CurveCodec:
    """Self-contained 1D-manifold codec.

    Encodes each point by a single latent t and each ambient dimension d by a
    Fourier-Power law v_d(t) = a_d t^(p_d+1) + sum_j A_dj sin(j w_d t + phi_dj).
    Storage is n + (4 + 2K) D numbers (latent, law parameters and shift).
    """

    def __init__(self, harmonics: int = 1, latent: str = "arc", maxfev: int = 5000,
                 cycles: np.ndarray = np.arange(0.25, 8.01, 0.25)):
        if latent not in LATENTS:
            raise ValueError(f"latent must be one of {sorted(LATENTS)}, got {latent!r}")
        self.harmonics = harmonics
        self.latent = latent
        self.maxfev = maxfev
        self.cycles = cycles  # candidate numbers of oscillations over the latent range

    def _fit_dimension(self, t: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, float]:
        shift = abs(v.min()) + 0.1
        vs = v + shift

        # 45 degree slope-multiplier initialiser for the power trend
        slope, intercept = np.polyfit(np.log(t), np.log(t / np.maximum(vs, 1e-6)), 1)
        a0, e0 = np.exp(-intercept), 1.0 - slope

        # Frequency initialiser: best single sinusoid (linear least squares) on the residual
        residual = vs - a0 * t ** e0
        best = None
        for omega in 2 * np.pi * self.cycles / (t.max() - t.min()):
            basis = np.column_stack([np.sin(omega * t), np.cos(omega * t)])
            coef, *_ = np.linalg.lstsq(basis, residual, rcond=None)
            err = np.sum((residual - basis @ coef) ** 2)
            if best is None or err < best[0]:
                best = (err, omega, coef)
        _, omega0, (c_sin, c_cos) = best
        params = np.array([a0, e0, omega0, np.hypot(c_sin, c_cos), np.arctan2(c_cos, c_sin)])

        # Add harmonics one at a time, warm-started from the previous fit; never accept a worse fit
        best_sse = np.inf
        for k in range(1, self.harmonics + 1):
            start = np.concatenate([params, np.zeros(3 + 2 * k - len(params))])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", OptimizeWarning)  # covariance is not used
                    fitted, _ = curve_fit(fourier_power_series, t, vs, p0=start, maxfev=self.maxfev)
                if not np.all(np.isfinite(fitted)):
                    fitted = start
            except (RuntimeError, ValueError):
                fitted = start
            sse = np.sum((vs - fourier_power_series(t, *fitted)) ** 2)
            if sse < best_sse:
                best_sse, params = sse, fitted
            else:
                params = start
        return np.asarray(params, dtype=float), shift

    def encode(self, points: np.ndarray) -> CurveCode:
        t = LATENTS[self.latent](points)
        t = t - t.min() + 0.1
        fits = [self._fit_dimension(t, points[:, d]) for d in range(points.shape[1])]
        return CurveCode(t, np.array([p for p, _ in fits]), np.array([s for _, s in fits]))

    @staticmethod
    def decode(code: CurveCode) -> np.ndarray:
        return np.column_stack([fourier_power_series(code.t, *p) - s
                                for p, s in zip(code.params, code.shifts)])


def pca_codec(points: np.ndarray, k: int) -> tuple[np.ndarray, int]:
    """Rank-k PCA reconstruction and its storage (n*k scores, D*k loadings, D means) in bytes."""
    mean = points.mean(axis=0)
    u, s, vt = np.linalg.svd(points - mean, full_matrices=False)
    n, D = points.shape
    return (u[:, :k] * s[:k]) @ vt[:k] + mean, 8 * (n * k + D * k + D)


def r2_score(points: np.ndarray, reconstructed: np.ndarray) -> float:
    """1 - SSE / total variance over all coordinates."""
    return 1.0 - np.sum((points - reconstructed) ** 2) / np.sum((points - points.mean(axis=0)) ** 2)


# =============================================================
# EXPERIMENTS
# =============================================================
@dataclass
class Experiment:
    title: str
    data: Callable[[], np.ndarray]
    transformer: Callable[[], AngleExponentTransformer]


@dataclass
class Result:
    title: str
    data: np.ndarray
    transformer: AngleExponentTransformer
    reconstructed: np.ndarray
    plane_2d: np.ndarray

    @property
    def raw_bytes(self) -> int:
        return self.data.nbytes

    @property
    def compressed_bytes(self) -> int:
        return self.transformer.x_1d.nbytes + self.transformer.param_bytes

    @property
    def compression_ratio(self) -> float:
        return self.raw_bytes / self.compressed_bytes

    @property
    def space_saved_pct(self) -> float:
        return (1.0 - self.compressed_bytes / self.raw_bytes) * 100.0

    @property
    def mse(self) -> float:
        """Reconstruction MSE over all D coordinates (equals mse_y / D)."""
        return float(np.mean((self.data - self.reconstructed) ** 2))

    @property
    def mse_y(self) -> float:
        """MSE of the regenerated y' coordinate on the projected plane."""
        return self.transformer.best.mse

    @property
    def r2_y(self) -> float:
        """Dimension-free fit score: 1 - MSE_y' / Var(y')."""
        return 1.0 - self.mse_y / float(np.var(self.transformer.best.y_raw))


def _power(factory, *args, **kw):
    return lambda: factory(*args, model=POWER, **kw)


def _fourier(factory, *args, p0=P0_ND, **kw):
    return lambda: factory(*args, model=FOURIER, fourier_p0=p0, **kw)


T = AngleExponentTransformer
EXPERIMENTS: dict[str, Experiment] = {
    # Power-law (log-linear 45 degree) transformers
    "egg_power": Experiment("3D Egg (Power)", egg_3d, _power(T.from_angles, 30, 45)),
    "square_power": Experiment("3D Square Mesh (Power)", square_3d_mesh, _power(T.from_angles, 30, 45)),
    "cube_power": Experiment("3D Cube Point Cloud (Power)", cube_3d_point_cloud, _power(T.from_angles, 30, 45)),
    "sphere_power": Experiment("3D Sphere (Power)", sphere_3d_mesh, _power(T.from_angles, 30, 45)),
    "hypersphere_power": Experiment("4D Hypersphere (Power)", hypersphere_4d, _power(T.from_givens_4d)),
    "sin20_power": Experiment("20D Sinusoidal Manifold (Power)", lambda: sinusoidal_manifold(20), _power(T.random, 20)),
    # Fourier-Power hybrid transformers
    "sin20_fourier": Experiment("20D Sinusoidal Manifold (Fourier-Power)", lambda: sinusoidal_manifold(20), _fourier(T.random, 20)),
    "sin100_fourier": Experiment("100D Sinusoidal Manifold (Fourier-Power)", lambda: sinusoidal_manifold(100), _fourier(T.random, 100)),
    "plane_fourier": Experiment("3D Plane (Fourier-Power)", plane_mesh, _fourier(T.from_angles, 30, 45, p0=P0_3D)),
    "sin600_fourier": Experiment("600D Sinusoidal Manifold (Fourier-Power)", lambda: sinusoidal_manifold(600), _fourier(T.random, 600)),
    "noise_fourier": Experiment("100D Gaussian Noise (Fourier-Power)", gaussian_noise, _fourier(T.random, 100, p0=P0_EDGE)),
    "cad_cube_fourier": Experiment("3D CAD Cube Mesh (Fourier-Power)", cad_cube, _fourier(T.random, 3, p0=P0_EDGE)),
    "paraboloid_fourier": Experiment("3D CAD Paraboloid (Fourier-Power)", cad_paraboloid, _fourier(T.from_angles, 25, 35, p0=P0_3D)),
    "hugelschaffer_fourier": Experiment("3D Hügelschäffer Egg (Fourier-Power)", hugelschaffer_egg, _fourier(T.from_angles, 30, 45, p0=P0_3D)),
    # Power-law baseline on the tilted plane (benchmark table)
    "plane_power": Experiment("3D Plane (Power Baseline)", plane_mesh, _power(T.from_angles, 30, 45)),
}


def run(data: np.ndarray, transformer: AngleExponentTransformer, title: str = "") -> Result:
    """Compress and reconstruct `data` with `transformer`."""
    priors = transformer.depth_priors(data)
    transformer.project_and_compress(data)
    reconstructed, plane = transformer.reconstruct(priors)
    return Result(title, data, transformer, reconstructed, plane)


def run_experiment(name: str) -> Result:
    exp = EXPERIMENTS[name]
    return run(exp.data(), exp.transformer(), exp.title)


# =============================================================
# REPORTING & PLOTS
# =============================================================
def print_report(res: Result) -> None:
    t = res.transformer
    print(f"=== CANDIDATE AXES FITTING ERRORS ({res.title}) ===")
    for idx, fit in t.axis_fits.items():
        flag = " [SELECTED AS OPTIMAL DRIVER]" if idx == t.driver_axis_idx else ""
        print(f"Axis {idx} as x' Driver -> 2D Fit MSE: {fit.mse:.8e} | Exponent (p+1): {fit.exponent:.4f}{flag}")

    print(f"\n=== SUMMARY ({res.title}) ===")
    print(f"Original Matrix Shape           : {res.data.shape} ({res.raw_bytes / 1024:.2f} KB)")
    print(f"Selected Optimal Driver Axis    : Axis {t.driver_axis_idx}")
    print(f"Stored 1D x' Array Shape        : {t.x_1d.shape} ({res.compressed_bytes / 1024:.2f} KB)")
    print("y' Coordinate Storage Status    : DROPPED (Generated on the fly)")
    print(f"Extracted Shape Exponent (p+1)  : {t.computed_exponent:.4f}")
    if t.model == FOURIER:
        A, omega, phi = t.fourier_params
        print(f"Fourier Wave (A, omega, phi)    : ({A:.6f}, {omega:.6f}, {phi:.6f})")
    print(f"Compression Factor              : {res.compression_ratio:.2f}x Reduction "
          f"({res.space_saved_pct:.2f}% space saved)")
    print(f"Final Reconstruction MSE        : {res.mse:.8e}")
    print(f"y' Fit Score R^2                : {res.r2_y:.4f}")


def _add_3d_axes(ax, length: float = 1.2):
    """Red (X), green (Y) and blue (Z) orientation arrows at the origin."""
    for vec, color in zip(np.eye(3) * length, ("red", "green", "blue")):
        ax.quiver(0, 0, 0, *vec, color=color, linewidth=2.5, arrow_length_ratio=0.1)


def plot_pipeline(res: Result, cmap: str = "viridis", plane_color: str = "purple", show_axes: bool = False):
    """Original cloud (first 3 coords), regenerated 2D plane, reconstructed cloud.

    For D > 3 the colour encodes the 4th coordinate.
    """
    orig, rec, t = res.data, res.reconstructed, res.transformer
    color_col = 3 if orig.shape[1] > 3 else 2
    fig = plt.figure(figsize=(18, 5))

    for pos, pts, label in ((131, orig, "1. Original"), (133, rec, "3. Reconstructed")):
        ax = fig.add_subplot(pos, projection="3d")
        ax.set_title(f"{label} {res.title}")
        ax.scatter3D(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, color_col], cmap=cmap, s=12, alpha=0.6)
        if show_axes:
            _add_3d_axes(ax)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    ax2 = fig.add_subplot(132)
    ax2.set_title(f"2. Reconstructed 2D Plane\nOptimal Driver $x'$ = Axis {t.driver_axis_idx} "
                  f"($p+1 = {t.computed_exponent:.2f}$)")
    ax2.scatter(res.plane_2d[:, 0], res.plane_2d[:, 1], c=plane_color, s=10, alpha=0.6, label="Generated Mesh")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("2D Projected X'"); ax2.set_ylabel("2D Projected Y'")
    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_trajectories(res: Result, dims: tuple[int, ...], num_samples: int = 100, plane_color: str = "teal"):
    """Regenerated 2D plane next to true vs reconstructed trajectories of chosen dimensions."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    t = res.transformer
    axes[0].set_title(f"{res.title}\nProjected 2D Plane, Optimal Driver Axis {t.driver_axis_idx}")
    axes[0].scatter(res.plane_2d[:, 0], res.plane_2d[:, 1], c=plane_color, s=10, alpha=0.6)
    axes[0].set_xlabel("Stored 1D x' Driver"); axes[0].set_ylabel("Generated y' Coordinate")
    axes[0].grid(True)

    axes[1].set_title(f"True vs Reconstructed Trajectory\n(Dimensions {', '.join(map(str, dims))})")
    for d, color in zip(dims, ("r", "g", "b", "m", "c", "y")):
        axes[1].plot(res.data[:num_samples, d], f"{color}-", label=f"True Dim {d}", linewidth=2, alpha=0.7)
        axes[1].plot(res.reconstructed[:num_samples, d], f"{color}--", label=f"Reconstructed Dim {d}")
    axes[1].set_xlabel("Sample Index"); axes[1].set_ylabel("Coordinate Value")
    axes[1].legend(ncol=2); axes[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    for key in EXPERIMENTS:
        print_report(run_experiment(key))
        print()
