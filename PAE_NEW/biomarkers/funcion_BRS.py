import numpy as np
from scipy.stats import linregress
from biomarkers.funcion_RR import funcion_rr, zarr_RR
from config.config import DATA_CONFIG
import vitaldb
from core.data_loader import try_extract_signal
from scipy.signal import find_peaks
# from utils_zarr import leer_senyal,list_available_tracks

def calcular_brs(vital_file_path):
    # Read the vital file (.vital)
    signals= vitaldb.read_vital(vital_file_path)

    #Get RR intervals (seconds), then convert to milliseconds
    rr_intervals = funcion_rr(vital_file_path) 
    rr= rr_intervals*1000
    print("R-R Intervals (ms):", rr)

    fs=DATA_CONFIG['sampling_rate']
    
    #Extract the ART using candidates names
    art_signal_candidates = DATA_CONFIG.get('art_signal_candidates', [DATA_CONFIG.get('ART', 'Demo/ART')])
    fs = DATA_CONFIG['sampling_rate']
    art_signal, art_raw = try_extract_signal(signals, art_signal_candidates, fs)
    art_signal = np.array(art_signal)
    art_signal = art_signal[~np.isnan(art_signal)]
    
    #Find systolic peaks in the ART signal
    peaks,_=find_peaks(art_signal,distance=100)
    sbp=art_signal[peaks]

    #Sequence method for BRS calculation
    brs = []
    n = min(len(sbp), len(rr)) - 2
    for i in range(n):
        # Ascending sequence
        if sbp[i] < sbp[i+1] < sbp[i+2] and rr[i] < rr[i+1] < rr[i+2]:
            slope,intercept,r_value,p,std_err = linregress(sbp[i:i+3], rr[i:i+3])
            if r_value > 0.6:
                brs.append(slope)
        # Descending sequence
        elif sbp[i] > sbp[i+1] > sbp[i+2] and rr[i] > rr[i+1] > rr[i+2]:
            slope,intercept,r_value,p,std_err = linregress(sbp[i:i+3], rr[i:i+3])
            if r_value > 0.6:
                brs.append(slope)
        
    if len(brs) > 0:
        print("BRS promedio:", np.mean(brs))
        print("BRS:", [float(x) for x in brs])
    else:
        print("No se encontraron secuencias válidas para BRS.")

    return np.array(brs)

"""""
Incomplete function

def zarr_BRS(zarr_path):

    tracks,preds=list_available_tracks(zarr_path)
    art_candidatos= DATA_CONFIG['art_signal_candidates']
    nombre_art= next ((cand for cand in art_candidatos if cand in tracks),None)
    rr= zarr_RR(zarr_path)
    resultado = leer_senyal(zarr_path, nombre_art)
    art_signal = resultado['values']
    tiempos_art = resultado['t_abs_ms']
    fs = DATA_CONFIG['sampling_rate']
    peaks,_=find_peaks(art_signal,distance=100)
    sbp=art_signal[peaks]
    brs = []
    n = min(len(sbp), len(rr)) - 2
    for i in range(n):
        # Secuencia ascendente
        if sbp[i] < sbp[i+1] < sbp[i+2] and rr[i] < rr[i+1] < rr[i+2]:
            slope,intercept,r_value,p,std_err = linregress(sbp[i:i+3], rr[i:i+3])
            if r_value > 0.6:
                brs.append(slope)
        # Secuencia descendente
        elif sbp[i] > sbp[i+1] > sbp[i+2] and rr[i] > rr[i+1] > rr[i+2]:
            slope,intercept,r_value,p,std_err = linregress(sbp[i:i+3], rr[i:i+3])
            if r_value > 0.6:
                brs.append(slope)
        
    if len(brs) > 0:
        print("BRS promedio:", np.mean(brs))
        print("BRS:", [float(x) for x in brs])
    else:
        print("No se encontraron secuencias válidas para BRS.")

    return np.array(brs)
"""

if __name__ == "__main__":
    #Example 
    vital_file = "C:/Users/junle/Desktop/PAE/VitalParser_ART-Prediction_Algorithms/PAE_NEW/VitalDB_data/230718/QUI12_230718_000947.vital"
    calcular_brs(vital_file)