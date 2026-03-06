from __future__ import annotations
from pathlib import Path
import numpy as np

def read_pcd_xyzi_ascii(pcd_path: str | Path) -> np.ndarray:
    """
    Legge un PCD con header standard e DATA ascii.
    Ritorna un array float32 shape (N,4): x,y,z,intensity
    """
    p = Path(pcd_path)
    if not p.exists():
        raise FileNotFoundError(p)

    header = []
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Header PCD incompleto (manca DATA).")
            s = line.strip()
            header.append(s)
            if s.startswith("DATA"):
                data_line = s
                break

    if data_line != "DATA ascii":
        raise ValueError(f"Questo reader gestisce solo DATA ascii. Trovato: {data_line}")

    # Trova ordine campi
    fields_line = next((h for h in header if h.startswith("FIELDS")), None)
    if fields_line is None:
        raise ValueError("Manca riga FIELDS nel PCD.")
    fields = fields_line.split()[1:]  # es: ['x','y','z','intensity']

    needed = ["x", "y", "z", "intensity"]
    for k in needed:
        if k not in fields:
            raise ValueError(f"Campo richiesto '{k}' non presente. FIELDS={fields}")

    # Indici colonne
    idx = [fields.index(k) for k in needed]

    # Carica tutti i punti (ascii)
    data = np.loadtxt(p, comments="#", dtype=np.float32, skiprows=len(header))
    # Se c'è un solo punto, loadtxt produce (C,) → lo forziamo a (1,C)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    xyzi = data[:, idx].astype(np.float32, copy=False)
    return xyzi

def read_pcd_ascii(pcd_path: str | Path) -> np.ndarray:
    """
    Legge un PCD con header standard e DATA ascii.
    Supporta FIELDS: x y z [intensity opzionale]
    
    Ritorna un array float32 shape (N,4): x,y,z,intensity
    Se intensity non è presente → viene impostata a 0.
    """
    p = Path(pcd_path)
    if not p.exists():
        raise FileNotFoundError(p)

    header = []
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Header PCD incompleto (manca DATA).")
            s = line.strip()
            header.append(s)
            if s.startswith("DATA"):
                data_line = s
                break

    if data_line != "DATA ascii":
        raise ValueError(f"Questo reader gestisce solo DATA ascii. Trovato: {data_line}")

    # Trova ordine campi
    fields_line = next((h for h in header if h.startswith("FIELDS")), None)
    if fields_line is None:
        raise ValueError("Manca riga FIELDS nel PCD.")
        
    fields = fields_line.split()[1:]

    needed_xyz = ["x", "y", "z"]
    for k in needed_xyz:
        if k not in fields:
            raise ValueError(f"Campo richiesto '{k}' non presente. FIELDS={fields}")

    # Indici XYZ
    idx_xyz = [fields.index(k) for k in needed_xyz]

    intensity_present = "intensity" in fields
    if intensity_present:
        idx_int = fields.index("intensity")

    # Carica dati
    data = np.loadtxt(p, comments="#", dtype=np.float32, skiprows=len(header))

    if data.ndim == 1:
        data = data.reshape(1, -1)

    xyz = data[:, idx_xyz].astype(np.float32, copy=False)

    if intensity_present:
        intensity = data[:, idx_int].reshape(-1, 1).astype(np.float32, copy=False)
    else:
        intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)

    xyzi = np.hstack((xyz, intensity))

    return xyzi

if __name__ == "__main__":
    test_path = r"pcd_tests/Test01/test02 (Frame 0020).pcd"
    pts = read_pcd_xyzi_ascii(test_path)
    print("Shape:", pts.shape, "dtype:", pts.dtype)
    print("First row:", pts[0])
    print("Intensity min/max:", float(pts[:,3].min()), float(pts[:,3].max()))