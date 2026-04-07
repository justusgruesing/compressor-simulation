# investigate_point41.py
from vclibpy.media import RefProp
from vclibpy.datamodels import FlowsheetState
from vclibpy.components.compressors.rolling_piston_Molinaroli_oil_path import (
    Molinaroli_2017_Compressor_Oil_Path,
)

# Deine gefitteten Parameter hier eintragen
params = {
    "Ua_suc_ref": 34.14, "Ua_dis_ref": 12.70, "Ua_amb": 0.625,
    "A_tot": 2.954e-7, "A_dis": 4.794e-4, "V_IC": 2.798e-5,
    "alpha_loss": 0.2766, "W_dot_loss_ref": 62.85,
    "alpha_fric_tot": 879.7, "m_dot_oil_ref": 1.712e-4,
    "Ua_suc_oil_ref": 11.26,
    "f_ref": 50.0, "m_dot_ref": 0.01589, "mu_fallback": 5.0,
}

med = RefProp(fluid_name="PROPANE")
comp = Molinaroli_2017_Compressor_Oil_Path(
    N_max=120.0, V_h=30.7e-6,
    fluid_name="propane", lub_name="LPG 68", parameters=params)
comp.med_prop = med
comp.debug_enabled = True

# Punkt 41: N70_SH10_Te20_Tc40
p_suc = 8.17964e5   # Pa
T_suc = 30.2259 + 273.15  # K
p_out = 13.6985e5   # Pa
T_amb = 27.254 + 273.15   # K
n_rel = (70.163 / 120.0)

class Ctrl:
    n = n_rel
class Inp:
    control = Ctrl()
    T_amb = T_amb
    lsq_max_nfev = 20000
    lsq_ftol = 1e-8
    lsq_xtol = 1e-8

fs = FlowsheetState()
comp.state_inlet = med.calc_state("PT", p_suc, T_suc)

try:
    x = comp.simulate_operating_point(Inp(), p_out, fs)
    print(f"\nm_flow = {comp.m_flow*1e3:.2f} g/s (meas: 34.01 g/s)")
    print(f"P_el   = {comp.P_el:.1f} W (meas: 1351.5 W)")
    print(f"T_dis  = {comp.state_outlet.T - 273.15:.1f} °C (meas: 57.2 °C)")
    print(f"m_dot_gas_exit = {comp.m_dot_KM_gas*1e3:.4f} g/s (should ≈ m_flow)")
    print(f"\n{comp.get_debug_report()}")
    print(f"\nSolver solution: {x}")
except Exception as e:
    print(f"FAILED: {e}")