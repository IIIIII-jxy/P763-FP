"""
        VARIABLE-STRENGTH BACK-ACTION TOMOGRAPHY
This sequence implements two-quadrature, phase-preserving, variable-strength readout on a single transmon to:
(i) map (Im, Qm) to the post-measurement Bloch vector
(ii) extract the chain efficiency η
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM, Transmon
from quam_libs.macros import qua_declaration, active_reset
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
import json
import pickle
from pathlib import Path
from datetime import datetime


# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = ["q4"]
    num_shots: int = 50000
    photon_numbers: List[float] = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    weak_readout_duration_ns: int = 300
    strong_readout_duration_ns: int = 300
    ringdown_time_ns: int = 400
    repetition_delay_ns: int = 8000
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
    conditional_map_bins: int = 51
    T1_us: float = 50.0
    T2_us: float = 30.0
    tau_us: float = 0.62
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    min_counts_per_bin: int = 20
    smoothing_sigma: float = 1.0


node = QualibrationNode(name="Variable_Strength_Back_Action_Tomography", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
u = unit(coerce_to_integer=True)
machine = QuAM.load()
config = machine.generate_config()

if node.parameters.load_data_id is None:
    qmm = machine.connect()

qubits = [machine.qubits["q4"]]
num_qubits = 1


# %% {QUA_program}
n_shots = node.parameters.num_shots
photon_numbers = np.array(node.parameters.photon_numbers)
n_photon_points = len(photon_numbers)
flux_point = node.parameters.flux_point_joint_or_independent
reset_type = node.parameters.reset_type_thermal_or_active
ringdown = node.parameters.ringdown_time_ns
rep_delay = node.parameters.repetition_delay_ns

tomo_axes = [0, 1, 2]
n_axes = len(tomo_axes)

with program() as back_action_tomography:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    
    I_weak = [declare(fixed) for _ in range(num_qubits)]
    Q_weak = [declare(fixed) for _ in range(num_qubits)]
    I_weak_st = [declare_stream() for _ in range(num_qubits)]
    Q_weak_st = [declare_stream() for _ in range(num_qubits)]
    
    I_strong = [declare(fixed) for _ in range(num_qubits)]
    Q_strong = [declare(fixed) for _ in range(num_qubits)]
    I_strong_st = [declare_stream() for _ in range(num_qubits)]
    Q_strong_st = [declare_stream() for _ in range(num_qubits)]
    
    shot = declare(int)
    photon_idx = declare(int)
    tomo_axis = declare(int)
    amp_scale = declare(fixed)
    
    amp_scales = declare(fixed, value=[float(np.sqrt(p / photon_numbers[-1])) for p in photon_numbers])

    for i, qubit in enumerate(qubits):
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)

        with for_(photon_idx, 0, photon_idx < n_photon_points, photon_idx + 1):
            assign(amp_scale, amp_scales[photon_idx])
            
            with for_(tomo_axis, 0, tomo_axis < n_axes, tomo_axis + 1):
                
                with for_(shot, 0, shot < n_shots, shot + 1):
                    save(shot, n_st)
                    
                    if reset_type == "active":
                        active_reset(qubit, "readout")
                    else:
                        qubit.resonator.wait(qubit.thermalization_time * u.ns)
                    
                    qubit.align()
                    qubit.xy.play("y90")
                    qubit.align()
                    
                    qubit.resonator.measure("readout", qua_vars=(I_weak[i], Q_weak[i]), amplitude_scale=amp_scale)
                    save(I_weak[i], I_weak_st[i])
                    save(Q_weak[i], Q_weak_st[i])
                    
                    qubit.resonator.wait(u.to_clock_cycles(ringdown))
                    qubit.align()
                    
                    with switch_(tomo_axis, unsafe=False):
                        with case_(0):
                            pass
                        with case_(1):
                            qubit.xy.play("-y90")
                        with case_(2):
                            qubit.xy.play("-x90")
                    
                    qubit.align()
                    
                    qubit.resonator.measure("readout", qua_vars=(I_strong[i], Q_strong[i]))
                    save(I_strong[i], I_strong_st[i])
                    save(Q_strong[i], Q_strong_st[i])
                    
                    qubit.resonator.wait(u.to_clock_cycles(rep_delay))

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_weak_st[i].buffer(n_shots).buffer(n_axes).buffer(n_photon_points).save(f"I_weak{i + 1}")
            Q_weak_st[i].buffer(n_shots).buffer(n_axes).buffer(n_photon_points).save(f"Q_weak{i + 1}")
            I_strong_st[i].buffer(n_shots).buffer(n_axes).buffer(n_photon_points).save(f"I_strong{i + 1}")
            Q_strong_st[i].buffer(n_shots).buffer(n_axes).buffer(n_photon_points).save(f"Q_strong{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)
    job = qmm.simulate(config, back_action_tomography, simulation_config)
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()), 1, i + 1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(back_action_tomography)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n_val = results.fetch_all()[0]
            progress_counter(n_val, n_shots, start_time=results.start_time)


# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        ds = fetch_results_as_xarray(
            job.result_handles,
            qubits,
            {
                "shot": np.arange(n_shots),
                "tomo_axis": ["Z", "X", "Y"],
                "photon_number": photon_numbers,
            },
        )
    else:
        ds, machine, json_data, qubits, node.parameters = load_dataset(
            node.parameters.load_data_id, parameters=node.parameters
        )
    
    node.results = {"ds": ds}


    # %% {Analysis_functions}
    
    def normalize_strong_readout(I_strong_Z):
        from scipy.signal import find_peaks
        hist, bin_edges = np.histogram(I_strong_Z, bins=100)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        hist_smooth = gaussian_filter(hist.astype(float), sigma=3)
        peaks, _ = find_peaks(hist_smooth, height=np.max(hist_smooth) * 0.1, distance=10)
        
        if len(peaks) >= 2:
            peak_heights = hist_smooth[peaks]
            top2_idx = np.argsort(peak_heights)[-2:]
            peak_positions = bin_centers[peaks[top2_idx]]
            I_g, I_e = np.min(peak_positions), np.max(peak_positions)
        else:
            I_g = np.percentile(I_strong_Z, 10)
            I_e = np.percentile(I_strong_Z, 90)
        
        I_mid = (I_g + I_e) / 2
        I_range = (I_e - I_g) / 2
        return I_mid, I_range, I_g, I_e
    
    def compute_apparent_strength_from_data(I_weak, Q_weak):
        I_clean = I_weak[np.abs(I_weak - np.median(I_weak)) < 5 * np.std(I_weak)]
        Q_clean = Q_weak[np.abs(Q_weak - np.median(Q_weak)) < 5 * np.std(Q_weak)]
        I_mean, Q_mean = np.mean(I_clean), np.mean(Q_clean)
        I_std, Q_std = np.std(I_clean), np.std(Q_clean)
        sigma = np.sqrt((I_std**2 + Q_std**2) / 2)
        signal = np.sqrt(I_mean**2 + Q_mean**2)
        s = signal / sigma if sigma > 0 else 0
        return s, sigma, I_std, Q_std
    
    def build_conditional_map_optimized(I_weak, Q_weak, outcome, n_bins=51, 
                                        sigma=None, min_counts=20, smooth_sigma=1.0):
        if sigma is None:
            sigma = np.sqrt((np.std(I_weak)**2 + np.std(Q_weak)**2) / 2)
        
        I_norm = I_weak / sigma
        Q_norm = Q_weak / sigma
        
        I_range = min(5, np.percentile(np.abs(I_norm), 99))
        Q_range = min(5, np.percentile(np.abs(Q_norm), 99))
        bin_range = max(I_range, Q_range)
        
        bin_edges = np.linspace(-bin_range, bin_range, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        sum_map, _, _ = np.histogram2d(I_norm, Q_norm, bins=[bin_edges, bin_edges], weights=outcome)
        count_map, _, _ = np.histogram2d(I_norm, Q_norm, bins=[bin_edges, bin_edges])
        
        cond_map = np.full((n_bins, n_bins), np.nan)
        valid = count_map.T >= min_counts
        cond_map[valid] = (sum_map.T[valid] / count_map.T[valid])
        
        if smooth_sigma > 0:
            cond_map_filled = np.nan_to_num(cond_map, nan=0)
            weight_map = (~np.isnan(cond_map)).astype(float)
            smoothed_values = gaussian_filter(cond_map_filled * weight_map, sigma=smooth_sigma)
            smoothed_weights = gaussian_filter(weight_map, sigma=smooth_sigma)
            with np.errstate(divide='ignore', invalid='ignore'):
                cond_map_smooth = np.where(smoothed_weights > 0.1, 
                                           smoothed_values / smoothed_weights, np.nan)
            cond_map = cond_map_smooth
        
        return cond_map, bin_centers, count_map.T
    
    def extract_slopes_robust(cond_map_Z, cond_map_X, bin_centers, counts_Z, counts_X, min_counts=20):
        center_idx = len(bin_centers) // 2
        window = min(10, center_idx - 1)
        
        def weighted_linfit(x, y, weights):
            valid = np.isfinite(y) & np.isfinite(weights) & (weights > 0)
            if np.sum(valid) < 3:
                return np.nan, np.nan
            x_v, y_v, w_v = x[valid], y[valid], weights[valid]
            W = np.diag(w_v)
            X = np.vstack([x_v, np.ones_like(x_v)]).T
            try:
                XtWX_inv = np.linalg.inv(X.T @ W @ X)
                beta = XtWX_inv @ X.T @ W @ y_v
                return beta[0], beta[1]
            except:
                return np.nan, np.nan
        
        I_slice = bin_centers[center_idx - window:center_idx + window + 1]
        Z_slice = cond_map_Z[center_idx, center_idx - window:center_idx + window + 1]
        counts_slice_Z = counts_Z[center_idx, center_idx - window:center_idx + window + 1]
        slope_Z_I, _ = weighted_linfit(I_slice, Z_slice, counts_slice_Z)
        
        Q_slice = bin_centers[center_idx - window:center_idx + window + 1]
        X_slice = cond_map_X[center_idx - window:center_idx + window + 1, center_idx]
        counts_slice_X = counts_X[center_idx - window:center_idx + window + 1, center_idx]
        slope_X_Q, _ = weighted_linfit(Q_slice, X_slice, counts_slice_X)
        
        return slope_Z_I, slope_X_Q
    
    def compute_bloch_length(cond_map_X, cond_map_Y, cond_map_Z):
        with np.errstate(invalid='ignore'):
            bloch_length = np.sqrt(
                np.nan_to_num(cond_map_X, nan=0)**2 + 
                np.nan_to_num(cond_map_Y, nan=0)**2 + 
                np.nan_to_num(cond_map_Z, nan=0)**2
            )
        all_nan = np.isnan(cond_map_X) & np.isnan(cond_map_Y) & np.isnan(cond_map_Z)
        bloch_length[all_nan] = np.nan
        return bloch_length
    
    def fit_eta_from_Y_decay_robust(s_values, Y_uncond, tau_us, T2_us):
        T2_factor = np.exp(-tau_us / T2_us)
        Y_abs = np.abs(Y_uncond)
        
        valid = (Y_abs > 1e-10) & np.isfinite(Y_abs) & (s_values > 0)
        if np.sum(valid) < 3:
            return np.nan, np.nan, None
        
        s_valid = s_values[valid]
        Y_valid = Y_abs[valid]
        
        def model(s, A, eta):
            return A * np.exp(-eta * s**2 / 2) * T2_factor
        
        try:
            A0 = Y_valid[0] / T2_factor if T2_factor > 0 else 1.0
            popt, pcov = curve_fit(
                model, s_valid, Y_valid,
                p0=[A0, 0.5],
                bounds=([0, 0.01], [2.0, 1.0]),
                maxfev=5000, method='trf'
            )
            A_fit, eta = popt
            eta_err = np.sqrt(pcov[1, 1]) if pcov[1, 1] > 0 else np.nan
            return eta, eta_err, (A_fit, T2_factor)
        except:
            log_Y = np.log(Y_valid / T2_factor)
            s_sq = s_valid**2
            try:
                coeffs = np.polyfit(s_sq, log_Y, 1)
                eta = -2 * coeffs[0]
                eta = np.clip(eta, 0.01, 1.0)
                return eta, np.nan, None
            except:
                return 0.5, np.nan, None
    
    def get_origin_bloch_length(cond_map_X, cond_map_Y, cond_map_Z, bin_centers, window=3):
        center_idx = len(bin_centers) // 2
        X_center = cond_map_X[center_idx-window:center_idx+window+1, center_idx-window:center_idx+window+1]
        Y_center = cond_map_Y[center_idx-window:center_idx+window+1, center_idx-window:center_idx+window+1]
        Z_center = cond_map_Z[center_idx-window:center_idx+window+1, center_idx-window:center_idx+window+1]
        X_avg, Y_avg, Z_avg = np.nanmean(X_center), np.nanmean(Y_center), np.nanmean(Z_center)
        return np.sqrt(X_avg**2 + Y_avg**2 + Z_avg**2)


    # %% {Main_analysis}
    qubit = qubits[0]
    qubit_name = qubit.name
    n_bins = node.parameters.conditional_map_bins
    tau_us = node.parameters.tau_us
    T1_us = node.parameters.T1_us
    T2_us = node.parameters.T2_us
    min_counts = node.parameters.min_counts_per_bin
    smooth_sigma = node.parameters.smoothing_sigma
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"back_action_tomography_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Calibration
    I_strong_Z_cal = ds["I_strong"].sel(qubit=qubit_name, photon_number=photon_numbers[-1], tomo_axis="Z").values
    I_mid, I_range, I_g, I_e = normalize_strong_readout(I_strong_Z_cal)
    
    print(f"Strong readout calibration: I_g = {I_g:.4f}, I_e = {I_e:.4f}")
    
    results_per_photon = {}
    s_values, Y_unconditioned = [], []
    slopes_Z_I, slopes_X_Q = [], []
    bloch_lengths_at_origin = []
    
    for n_bar in photon_numbers:
        I_w_Z = ds["I_weak"].sel(qubit=qubit_name, photon_number=n_bar, tomo_axis="Z").values
        Q_w_Z = ds["Q_weak"].sel(qubit=qubit_name, photon_number=n_bar, tomo_axis="Z").values
        I_s_Z = ds["I_strong"].sel(qubit=qubit_name, photon_number=n_bar, tomo_axis="Z").values
        
        I_w_X = ds["I_weak"].sel(qubit=qubit_name, photon_number=n_bar, tomo_axis="X").values
        Q_w_X = ds["Q_weak"].sel(qubit=qubit_name, photon_number=n_bar, tomo_axis="X").values
        I_s_X = ds["I_strong"].sel(qubit=qubit_name, photon_number=n_bar, tomo_axis="X").values
        
        I_w_Y = ds["I_weak"].sel(qubit=qubit_name, photon_number=n_bar, tomo_axis="Y").values
        Q_w_Y = ds["Q_weak"].sel(qubit=qubit_name, photon_number=n_bar, tomo_axis="Y").values
        I_s_Y = ds["I_strong"].sel(qubit=qubit_name, photon_number=n_bar, tomo_axis="Y").values
        
        outcome_Z = np.clip((I_s_Z - I_mid) / I_range, -1.5, 1.5)
        outcome_X = np.clip((I_s_X - I_mid) / I_range, -1.5, 1.5)
        outcome_Y = np.clip((I_s_Y - I_mid) / I_range, -1.5, 1.5)
        
        s, sigma, I_std, Q_std = compute_apparent_strength_from_data(I_w_Z, Q_w_Z)
        s_values.append(s)
        Y_unconditioned.append(np.mean(outcome_Y))
        
        cond_Z, bins, counts_Z = build_conditional_map_optimized(I_w_Z, Q_w_Z, outcome_Z, n_bins, sigma, min_counts, smooth_sigma)
        cond_X, _, counts_X = build_conditional_map_optimized(I_w_X, Q_w_X, outcome_X, n_bins, sigma, min_counts, smooth_sigma)
        cond_Y, _, counts_Y = build_conditional_map_optimized(I_w_Y, Q_w_Y, outcome_Y, n_bins, sigma, min_counts, smooth_sigma)
        
        slope_Z_I, slope_X_Q = extract_slopes_robust(cond_Z, cond_X, bins, counts_Z, counts_X, min_counts)
        slopes_Z_I.append(slope_Z_I)
        slopes_X_Q.append(slope_X_Q)
        
        bloch_at_origin = get_origin_bloch_length(cond_X, cond_Y, cond_Z, bins, window=2)
        bloch_lengths_at_origin.append(bloch_at_origin)
        
        bloch_map = compute_bloch_length(cond_X, cond_Y, cond_Z)
        
        results_per_photon[n_bar] = {
            "s": s, "sigma": sigma,
            "cond_map_Z": cond_Z, "cond_map_X": cond_X, "cond_map_Y": cond_Y,
            "bin_centers": bins,
            "counts_Z": counts_Z, "counts_X": counts_X, "counts_Y": counts_Y,
            "slope_Z_I": slope_Z_I, "slope_X_Q": slope_X_Q,
            "bloch_length_map": bloch_map, "bloch_at_origin": bloch_at_origin,
        }
    
    s_values = np.array(s_values)
    Y_unconditioned = np.array(Y_unconditioned)
    slopes_Z_I = np.array(slopes_Z_I)
    slopes_X_Q = np.array(slopes_X_Q)
    bloch_lengths_at_origin = np.array(bloch_lengths_at_origin)
    
    eta, eta_err, fit_params = fit_eta_from_Y_decay_robust(s_values, Y_unconditioned, tau_us, T2_us)
    
    T1_decay = np.exp(-tau_us / T1_us)
    T2_decay = np.exp(-tau_us / T2_us)
    expected_slope_Z = s_values * T1_decay
    expected_slope_X = s_values * T2_decay
    
    # Success metrics
    valid_slopes_Z = ~np.isnan(slopes_Z_I) & (s_values > 0.1)
    if np.any(valid_slopes_Z):
        slope_ratio_Z = slopes_Z_I[valid_slopes_Z] / expected_slope_Z[valid_slopes_Z]
        slopes_within_90pct = np.mean(np.abs(slope_ratio_Z - 1) < 0.1) > 0.5
    else:
        slopes_within_90pct = False
    
    small_s_idx = s_values < np.median(s_values)
    bloch_ge_0p9_small_s = np.nanmean(bloch_lengths_at_origin[small_s_idx]) >= 0.8 if np.any(small_s_idx) else False
    
    print(f"\n{'='*60}")
    print(f"Back-Action Tomography Results for {qubit_name}")
    print(f"{'='*60}")
    print(f"Extracted η = {eta:.4f}")
    print(f"T1 decay = {T1_decay:.4f}, T2 decay = {T2_decay:.4f}")


    # %% {Individual_plots}
    all_figures = {}
    
    # 1. η Extraction Plot
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    Y_abs = np.abs(Y_unconditioned)
    valid_plot = Y_abs > 1e-10
    ax1.semilogy(s_values[valid_plot]**2, Y_abs[valid_plot], 'o', markersize=12, 
                 color='#1f77b4', label="Data", markeredgecolor='white', markeredgewidth=1.5)
    if fit_params is not None:
        A_fit, T2_factor = fit_params
        s_fit = np.linspace(0, np.max(s_values) * 1.1, 100)
        Y_fit = A_fit * np.exp(-eta * s_fit**2 / 2) * T2_factor
        ax1.semilogy(s_fit**2, Y_fit, '--', linewidth=3, color='#d62728', label=f"Fit: η = {eta:.3f}")
    ax1.set_xlabel(r"$s^2$ (apparent strength squared)", fontsize=14)
    ax1.set_ylabel(r"$|\langle Y \rangle|$ (unconditioned)", fontsize=14)
    ax1.set_title(f"η Extraction: η = {eta:.3f}", fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=12)
    plt.tight_layout()
    fig1.savefig(figures_dir / "01_eta_extraction.png", dpi=200, bbox_inches='tight', facecolor='white')
    fig1.savefig(figures_dir / "01_eta_extraction.pdf", bbox_inches='tight')
    all_figures["eta_extraction"] = fig1
    print(f"Saved: 01_eta_extraction.png")
    
    # 2. Back-Action Slopes Plot
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(s_values, slopes_Z_I, 'o-', color='#1f77b4', markersize=10, linewidth=2.5, 
             label=r"$\partial\langle Z\rangle_c/\partial I_m \cdot \sigma$ (data)")
    ax2.plot(s_values, expected_slope_Z, '--', color='#1f77b4', linewidth=2, alpha=0.6, 
             label=r"$s \cdot e^{-\tau/T_1}$ (theory)")
    ax2.plot(s_values, slopes_X_Q, 's-', color='#ff7f0e', markersize=10, linewidth=2.5, 
             label=r"$\partial\langle X\rangle_c/\partial Q_m \cdot \sigma$ (data)")
    ax2.plot(s_values, expected_slope_X, '--', color='#ff7f0e', linewidth=2, alpha=0.6, 
             label=r"$s \cdot e^{-\tau/T_2}$ (theory)")
    ax2.set_xlabel("Apparent strength s", fontsize=14)
    ax2.set_ylabel("Slope", fontsize=14)
    ax2.set_title("Back-Action Slopes", fontsize=16, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper left')
    ax2.set_xlim(left=0)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=12)
    plt.tight_layout()
    fig2.savefig(figures_dir / "02_back_action_slopes.png", dpi=200, bbox_inches='tight', facecolor='white')
    fig2.savefig(figures_dir / "02_back_action_slopes.pdf", bbox_inches='tight')
    all_figures["back_action_slopes"] = fig2
    print(f"Saved: 02_back_action_slopes.png")
    
    # 3. Conditional Purity Plot
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.plot(s_values, bloch_lengths_at_origin, 'o-', color='#2ca02c', markersize=12, 
             linewidth=2.5, markeredgecolor='white', markeredgewidth=1.5)
    ax3.axhline(y=0.9, color='#d62728', linestyle='--', linewidth=2.5, label="Target ≥ 0.9")
    ax3.fill_between([0, np.max(s_values)*1.1], 0.9, 1.0, alpha=0.2, color='#2ca02c')
    ax3.set_xlabel("Apparent strength s", fontsize=14)
    ax3.set_ylabel("Bloch vector length |r|", fontsize=14)
    ax3.set_title("Conditional Purity at Origin", fontsize=16, fontweight='bold')
    ax3.set_ylim([0, 1.1])
    ax3.set_xlim(left=0)
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=12)
    plt.tight_layout()
    fig3.savefig(figures_dir / "03_conditional_purity.png", dpi=200, bbox_inches='tight', facecolor='white')
    fig3.savefig(figures_dir / "03_conditional_purity.pdf", bbox_inches='tight')
    all_figures["conditional_purity"] = fig3
    print(f"Saved: 03_conditional_purity.png")
    
    # 4-6. Conditional Maps for lowest photon number
    lowest_n = photon_numbers[0]
    res = results_per_photon[lowest_n]
    bins = res["bin_centers"]
    extent = [bins[0], bins[-1], bins[0], bins[-1]]
    
    map_info = [
        ("cond_map_Z", r"$\langle Z \rangle_c$", "04_conditional_map_Z"),
        ("cond_map_X", r"$\langle X \rangle_c$", "05_conditional_map_X"),
        ("cond_map_Y", r"$\langle Y \rangle_c$", "06_conditional_map_Y"),
    ]
    
    for map_key, title_label, filename in map_info:
        fig, ax = plt.subplots(figsize=(8, 7))
        data = res[map_key]
        vmax = min(max(np.nanmax(np.abs(data)), 0.1), 1.0)
        im = ax.imshow(data, origin='lower', extent=extent, aspect='equal', 
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
        ax.set_xlabel(r"$I_m/\sigma$", fontsize=14)
        ax.set_ylabel(r"$Q_m/\sigma$", fontsize=14)
        ax.set_title(f"{title_label} (n̄ = {lowest_n})", fontsize=16, fontweight='bold')
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='k', linewidth=0.5, alpha=0.5)
        ax.tick_params(labelsize=12)
        cbar = plt.colorbar(im, ax=ax, shrink=0.85)
        cbar.ax.tick_params(labelsize=11)
        plt.tight_layout()
        fig.savefig(figures_dir / f"{filename}.png", dpi=200, bbox_inches='tight', facecolor='white')
        fig.savefig(figures_dir / f"{filename}.pdf", bbox_inches='tight')
        all_figures[map_key] = fig
        print(f"Saved: {filename}.png")
    
    # 7. Conditional maps for all photon numbers (Z only)
    n_plots = len(photon_numbers)
    for i, n_bar in enumerate(photon_numbers):
        fig, ax = plt.subplots(figsize=(7, 6))
        res = results_per_photon[n_bar]
        bins = res["bin_centers"]
        extent = [bins[0], bins[-1], bins[0], bins[-1]]
        data = res["cond_map_Z"]
        vmax = min(max(np.nanmax(np.abs(data)), 0.1), 1.0)
        im = ax.imshow(data, origin='lower', extent=extent, aspect='equal', 
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_xlabel(r"$I_m/\sigma$", fontsize=13)
        ax.set_ylabel(r"$Q_m/\sigma$", fontsize=13)
        ax.set_title(f"⟨Z⟩_c, n̄={n_bar:.3f}, s={res['s']:.2f}", fontsize=14, fontweight='bold')
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='k', linewidth=0.5, alpha=0.5)
        cbar = plt.colorbar(im, ax=ax, shrink=0.85)
        plt.tight_layout()
        fig.savefig(figures_dir / f"07_Z_map_nbar_{i:02d}_{n_bar:.3f}.png", dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    print(f"Saved: 07_Z_map_nbar_*.png (all photon numbers)")
    
    # 8. Readout Calibration Histogram
    fig8, ax8 = plt.subplots(figsize=(8, 6))
    ax8.hist(I_strong_Z_cal, bins=100, density=True, alpha=0.7, color='#1f77b4', edgecolor='white')
    ax8.axvline(I_g, color='#d62728', linestyle='--', linewidth=2.5, label=f'I_g = {I_g:.4f}')
    ax8.axvline(I_e, color='#2ca02c', linestyle='--', linewidth=2.5, label=f'I_e = {I_e:.4f}')
    ax8.set_xlabel("Strong readout I", fontsize=14)
    ax8.set_ylabel("Density", fontsize=14)
    ax8.set_title("Readout Calibration", fontsize=16, fontweight='bold')
    ax8.legend(fontsize=12)
    ax8.tick_params(labelsize=12)
    plt.tight_layout()
    fig8.savefig(figures_dir / "08_readout_calibration.png", dpi=200, bbox_inches='tight', facecolor='white')
    fig8.savefig(figures_dir / "08_readout_calibration.pdf", bbox_inches='tight')
    all_figures["readout_calibration"] = fig8
    print(f"Saved: 08_readout_calibration.png")
    
    # 9. Slope Ratios
    fig9, ax9 = plt.subplots(figsize=(8, 6))
    valid_Z = ~np.isnan(slopes_Z_I) & (expected_slope_Z > 0)
    valid_X = ~np.isnan(slopes_X_Q) & (expected_slope_X > 0)
    if np.any(valid_Z):
        ax9.plot(s_values[valid_Z], slopes_Z_I[valid_Z]/expected_slope_Z[valid_Z], 
                 'o-', label='Z slope ratio', color='#1f77b4', markersize=10, linewidth=2)
    if np.any(valid_X):
        ax9.plot(s_values[valid_X], slopes_X_Q[valid_X]/expected_slope_X[valid_X], 
                 's-', label='X slope ratio', color='#ff7f0e', markersize=10, linewidth=2)
    ax9.axhline(1.0, color='k', linestyle='--', linewidth=2, alpha=0.5)
    ax9.fill_between([0, max(s_values)*1.1], 0.9, 1.1, alpha=0.2, color='green')
    ax9.set_xlabel("Apparent strength s", fontsize=14)
    ax9.set_ylabel("Measured / Expected Slope", fontsize=14)
    ax9.set_title("Slope Ratio (target: 1.0)", fontsize=16, fontweight='bold')
    ax9.legend(fontsize=12)
    ax9.grid(True, alpha=0.3)
    ax9.tick_params(labelsize=12)
    plt.tight_layout()
    fig9.savefig(figures_dir / "09_slope_ratios.png", dpi=200, bbox_inches='tight', facecolor='white')
    fig9.savefig(figures_dir / "09_slope_ratios.pdf", bbox_inches='tight')
    all_figures["slope_ratios"] = fig9
    print(f"Saved: 09_slope_ratios.png")
    
    # 10. Summary Figure
    fig10, ax10 = plt.subplots(figsize=(10, 8))
    ax10.axis('off')
    summary_text = f"""
    ══════════════════════════════════════════════════════════════
                    BACK-ACTION TOMOGRAPHY SUMMARY
    ══════════════════════════════════════════════════════════════
    
    Qubit: {qubit_name}
    Date: {timestamp}
    
    ──────────────────────────────────────────────────────────────
    EXTRACTED PARAMETERS
    ──────────────────────────────────────────────────────────────
    
    η (chain efficiency) = {eta:.4f} ± {f"{eta_err:.4f}" if not np.isnan(eta_err) else "N/A"}

    Decay parameters:
        T₁ = {T1_us:.1f} μs
        T₂ = {T2_us:.1f} μs
        τ  = {tau_us:.2f} μs
        exp(-τ/T₁) = {T1_decay:.4f}
        exp(-τ/T₂) = {T2_decay:.4f}
    
    Readout calibration:
        I_g = {I_g:.4f}
        I_e = {I_e:.4f}
    
    ──────────────────────────────────────────────────────────────
    SUCCESS METRICS
    ──────────────────────────────────────────────────────────────
    
    η in valid range [0, 1]:          {'✓ PASS' if 0 < eta <= 1 else '✗ FAIL'}
    Slopes within 90% of expected:    {'✓ PASS' if slopes_within_90pct else '✗ FAIL'}
    Bloch length ≥ 0.8 at small s:    {'✓ PASS' if bloch_ge_0p9_small_s else '✗ FAIL'}
    
    ──────────────────────────────────────────────────────────────
    OVERALL OUTCOME: {'SUCCESSFUL' if (0 < eta <= 1) else 'NEEDS REVIEW'}
    ══════════════════════════════════════════════════════════════
    """
    ax10.text(0.05, 0.95, summary_text, transform=ax10.transAxes, fontsize=12, 
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.tight_layout()
    fig10.savefig(figures_dir / "10_summary.png", dpi=200, bbox_inches='tight', facecolor='white')
    fig10.savefig(figures_dir / "10_summary.pdf", bbox_inches='tight')
    all_figures["summary"] = fig10
    print(f"Saved: 10_summary.png")
    
    plt.show()


    # %% {Save_results}
    
    # Save dataset
    ds.to_netcdf(output_dir / "dataset.nc")
    print(f"Dataset saved to {output_dir / 'dataset.nc'}")
    
    # JSON-serializable helper
    def to_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        return obj
    
    # Save fit results
    json_results = {
        qubit_name: {
            "eta": to_json(eta),
            "eta_err": to_json(eta_err),
            "s_values": [to_json(x) for x in s_values],
            "Y_unconditioned": [to_json(x) for x in Y_unconditioned],
            "slopes_Z_I": [to_json(x) for x in slopes_Z_I],
            "slopes_X_Q": [to_json(x) for x in slopes_X_Q],
            "expected_slope_Z": [to_json(x) for x in expected_slope_Z],
            "expected_slope_X": [to_json(x) for x in expected_slope_X],
            "bloch_lengths_at_origin": [to_json(x) for x in bloch_lengths_at_origin],
            "photon_numbers": [to_json(x) for x in photon_numbers],
            "T1_us": T1_us, "T2_us": T2_us, "tau_us": tau_us,
            "calibration": {"I_g": to_json(I_g), "I_e": to_json(I_e)},
            "success_metrics": {
                "eta_valid": bool(0 < eta <= 1),
                "slopes_within_90pct": bool(slopes_within_90pct),
                "bloch_ge_0p9_small_s": bool(bloch_ge_0p9_small_s),
            }
        }
    }
    with open(output_dir / "fit_results.json", "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Fit results saved to {output_dir / 'fit_results.json'}")
    
    # Save full results (pickle)
    with open(output_dir / "full_results.pkl", "wb") as f:
        pickle.dump({"results_per_photon": results_per_photon, "s_values": s_values, 
                     "eta": eta, "eta_err": eta_err}, f)
    print(f"Full results saved to {output_dir / 'full_results.pkl'}")
    
    # Save parameters
    with open(output_dir / "parameters.json", "w") as f:
        json.dump(node.parameters.model_dump(), f, indent=2)
    print(f"Parameters saved to {output_dir / 'parameters.json'}")
    
    # Final output
    node.results["figures"] = all_figures
    node.results["fit_results"] = json_results
    node.outcomes = {qubit.name: "successful" if 0 < eta <= 1 else "needs_review"}
    node.results["output_dir"] = str(output_dir)
    node.machine = machine
    node.save()
    
    print(f"\n{'='*60}")
    print(f"All results saved to: {output_dir}")
    print(f"Figures saved to: {figures_dir}")
    print(f"Outcome: {node.outcomes[qubit.name]}")
    print(f"{'='*60}")