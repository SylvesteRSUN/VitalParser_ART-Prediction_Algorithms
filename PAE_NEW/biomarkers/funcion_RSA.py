import numpy as np
from scipy.signal import find_peaks
from biomarkers.funcion_RR import funcion_rr, zarr_RR
from config.config import DATA_CONFIG
import vitaldb
from core.data_loader import try_extract_signal
# from utils_zarr import list_available_tracks,leer_senyal

def calcular_rsa(vital_file_path):
    """
    Calculates Respiratory Sinus Arrhythmia (RSA) using respiratory cycles and RR intervals
    from a .vital file.

    Parameters:
    vital_file_path: Path to the .vital file

    Returns:
    rsa_values_clean: List of RSA values per respiratory cycle (float, in seconds difference)
    """

    # Read the vital file
    signals = vitaldb.read_vital(vital_file_path)

    # Extract the respiration signal
    resp_signal_candidates = DATA_CONFIG.get('resp_signal_candidates', [DATA_CONFIG.get('RESP', 'Demo/RESP')])
    fs = DATA_CONFIG['sampling_rate']
    resp_signal, resp_raw = try_extract_signal(signals, resp_signal_candidates, fs)
    resp_signal = np.array(resp_signal)
    resp_signal = resp_signal[~np.isnan(resp_signal)]

    # Extract RR intervals (intervals between R-peaks in the ECG)
    rr_intervals = funcion_rr(vital_file_path)

    # Detect peaks in the respiration signal: each peak marks a new respiratory cycle
    peaks, _ = find_peaks(resp_signal, distance=50)

    # Calculate RSA for each respiratory cycle using the peak-to-trough method
    rsa_values = []
    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1]
        rr_cycle = rr_intervals[start:end]
        if len(rr_cycle) > 0:
            rsa = np.max(rr_cycle) - np.min(rr_cycle)
            rsa_values.append(rsa)

     # Convert values to float type 
    rsa_values_clean = [float(x) for x in rsa_values]

    return rsa_values_clean

"""
Incompleted function

def zarr_RSA(zarr_path):
    tracks,_ = list_available_tracks(zarr_path)
    resp_candidatos = DATA_CONFIG['resp_signal_candidates']
    nombre_resp= next((cand for cand in resp_candidatos if cand in tracks),None)
    if nombre_resp is None:
        print ("No se encontro la señal")
        return None
    
    resultado = leer_senyal(zarr_path, nombre_resp)
    resp_signal = resultado['values']
    tiempos_resp = resultado['t_abs_ms']
    fs = DATA_CONFIG['sampling_rate']

    rr= zarr_RR(zarr_path)

    peaks, _ = find_peaks(resp_signal, distance=50)

    # Calcular RSA para cada ciclo respiratorio
    rsa_values = []
    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1]
        rr_cycle = rr[start:end]
        if len(rr_cycle) > 0:
            rsa = np.max(rr_cycle) - np.min(rr_cycle)
            rsa_values.append(rsa)

    # Convertir a lista de floats
    rsa_values_clean = [float(x) for x in rsa_values]

    return rsa_values_clean
"""

if __name__== "__main__":
    vital_file = "C:/Users/junle/Desktop/PAE/VitalParser_ART-Prediction_Algorithms/PAE_NEW/VitalDB_data/230718/QUI12_230718_000947.vital"
    rsa_result = calcular_rsa(vital_file)
    print("RSA values (seconds):", rsa_result)
