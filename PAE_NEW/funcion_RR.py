import sys
sys.path.append('./py-ecg-detectors')  # The ecg_detectors folder is in the same directory
from ecgdetectors import Detectors
import numpy as np
from data_loader import try_extract_signal
from config import DATA_CONFIG
import vitaldb
from utils_zarr import list_available_tracks,leer_senyal

def funcion_rr(vital_file,ecg_signal_candidates=None, fs=None):
    """
    Extracts R-R intervals from a .vital file using the Pan-Tompkins detector.

    Parameters:
    vital_file: Path to the .vital file
    ecg_signal_candidates: List of possible ECG channel names (optional; will use those in DATA_CONFIG if not provided)
    fs: Sampling frequency (optional; uses 'sampling_rate' from DATA_CONFIG if not specified)

    Returns:
    rr_intervals: Array of R-R intervals in seconds
    """

    signals= vitaldb.read_vital(vital_file)
    
    # Set ECG candidates and sampling frequency
    if ecg_signal_candidates is None:
        ecg_signal_candidates = DATA_CONFIG.get('ecg_signal_candidates', [DATA_CONFIG.get('ECG', 'Demo/ECG')])
    if fs is None:
        fs = DATA_CONFIG['sampling_rate']

    # Extract the ECG signal
    ecg_signal, ecg_raw = try_extract_signal(signals, ecg_signal_candidates, fs)
    ecg_signal = np.array(ecg_signal, dtype=np.float64)
    ecg_signal = ecg_signal[~np.isnan(ecg_signal)]

    # Generate time vector
    times = np.arange(len(ecg_signal)) / fs

    # Detector Pan-Tompkins
    detectors = Detectors(fs)
    r_peaks_ind = detectors.pan_tompkins_detector(ecg_signal)

    # Calculate R-R intervals (in seconds)
    r_peaks_times = times[r_peaks_ind]
    rr_intervals = np.diff(r_peaks_times)
    return rr_intervals

"""""
Incompleted function

def zarr_RR(zarr_path):
    
    tracks,_ = list_available_tracks(zarr_path)
    candidatos_ecg=DATA_CONFIG['ecg_signal_candidates']
    nombre_ecg= next((cand for cand in candidatos_ecg if cand in tracks), None)

    if nombre_ecg is None:
        print("No se encontraron señales")
        return None
    
    resultado = leer_senyal(zarr_path, nombre_ecg)
    ecg_signal = resultado['values']
    tiempos_ecg = resultado['t_abs_ms']
    fs = DATA_CONFIG['sampling_rate']

    # Detector Pan-Tompkins
    detectors = Detectors(fs)
    r_peaks_ind = detectors.pan_tompkins_detector(ecg_signal)

    # Calcula los intervalos R-R (segundos)
    r_peaks_times = tiempos_ecg[r_peaks_ind]
    rr_intervals = np.diff(r_peaks_times)
    return rr_intervals
"""
if __name__ == "__main__":
    vital_file = "C:/Users/junle/Desktop/PAE/PAE2/VitalDB_data/VitalDB_data/230807/QUI12_230807_163951.vital"
    rr_intervals = funcion_rr(vital_file)
    print("R-R Intervals (segundos):", rr_intervals)