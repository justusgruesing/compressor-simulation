# debug_solve_w_KM_range.py
from vclibpy.media import RefProp
from vclibpy.media.lubricant_fitting_shared_refprop import LubricantFitting

med = RefProp(fluid_name="PROPANE")
lub = LubricantFitting(fluid_name="propane", lub_name="LPG 68", shared_refprop=med)

# Drei typische Drücke: niedrig (Tc30), mittel (Tc50), hoch (Tc60)
pressures = [10.7e5, 17.2e5, 21.2e5]

for p in pressures:
    print(f"\n=== p = {p/1e5:.1f} bar ===")
    first_valid = None
    for T_C in range(-10, 130, 2):
        T_K = T_C + 273.15
        w = lub.solve_w_KM(T_K, p)
        if w is not None and first_valid is None:
            first_valid = T_C
        status = f"{w:.4f}" if w is not None else "None"
        if T_C < 40 or w is None:  # Nur den interessanten Bereich drucken
            print(f"  T={T_C:4d} °C  →  w = {status}")
    if first_valid is not None:
        print(f"  ... (weiter gültig ab T={first_valid} °C)")
    print(f"  Erste gültige Temperatur: {first_valid} °C")