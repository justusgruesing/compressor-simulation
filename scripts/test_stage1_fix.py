# scripts/test_stage1_fix.py
#
# Tests the Stage 1 brentq fix on 4 representative points:
# - N60_SH10_Te0_Tc30  (vorher Fallback, niedriges Tc)
# - N60_SH10_Te0_Tc50  (vorher Fallback, mittleres Tc)
# - N70_SH10_Te0_Tc60  (vorher Fallback, hohes Tc)
# - N60_SH30_Te0_Tc50  (vorher erfolgreich, Kontrolle)
#
# Prüft: stage1_fallback_count, w_KM_mix vs w_KM_after, Q_dissolve_3

from vclibpy.media import RefProp
from vclibpy.datamodels import FlowsheetState
from vclibpy.components.compressors.rolling_piston_Molinaroli_oil_path import (
    Molinaroli_2017_Compressor_Oil_Path,
)

params = {
    "Ua_suc_ref": 34.14, "Ua_dis_ref": 12.70, "Ua_amb": 0.625,
    "A_tot": 2.954e-7, "A_dis": 4.794e-4, "V_IC": 2.798e-5,
    "alpha_loss": 0.2766, "W_dot_loss_ref": 62.85,
    "alpha_fric_tot": 879.7, "m_dot_oil_ref": 1.712e-4,
    "Ua_suc_oil_ref": 11.26,
    "f_ref": 50.0, "m_dot_ref": 0.01589, "mu_fallback": 5.0,
}

# Testpunkte: (name, p_suc_bar, T_suc_C, p_out_bar, T_amb_C, N_rpm, m_meas_gs)
test_points = [
    ("N60_SH10_Te0_Tc30",  4.771, 9.67,  10.694, 21.60, 3609, 17.41),
    ("N60_SH10_Te0_Tc50",  4.753, 9.90,  17.194, 26.24, 3609, 16.46),
    ("N70_SH10_Te0_Tc60",  4.787, 9.87,  21.210, 29.47, 4210, 19.06),
    ("N60_SH30_Te0_Tc50",  4.767, 29.79, 17.216, 29.72, 3609, 15.06),
]

med = RefProp(fluid_name="PROPANE")
N_max = 120.0

class Ctrl:
    n = 0.0
class Inp:
    control = Ctrl()
    T_amb = 298.15
    lsq_max_nfev = 20000
    lsq_ftol = 1e-8
    lsq_xtol = 1e-8

print(f"{'Point':30s}  {'s1_fb':>6s}  {'w_after':>7s}  {'w_mix':>7s}  {'w_dis':>7s}  "
      f"{'Q_dis3':>8s}  {'Q_sump':>8s}  {'e_m%':>6s}  {'e_T[K]':>7s}  {'pc_gap':>7s}")
print("-" * 120)

for name, p_suc_bar, T_suc_C, p_out_bar, T_amb_C, N_rpm, m_meas in test_points:
    comp = Molinaroli_2017_Compressor_Oil_Path(
        N_max=N_max, V_h=30.7e-6,
        fluid_name="propane", lub_name="LPG 68", parameters=dict(params))
    comp.med_prop = med
    comp.debug_enabled = True

    p_suc = p_suc_bar * 1e5
    p_out = p_out_bar * 1e5
    T_suc = T_suc_C + 273.15
    n_rel = (N_rpm / 60.0) / N_max

    Inp.control.n = max(1e-9, min(1.0, n_rel))
    Inp.T_amb = T_amb_C + 273.15

    fs = FlowsheetState()
    comp.state_inlet = med.calc_state("PT", p_suc, T_suc)

    try:
        comp.simulate_operating_point(Inp, p_out, fs)

        e_m = (comp.m_flow * 1e3 / m_meas - 1.0) * 100
        T_dis_C = comp.state_outlet.T - 273.15

        w_mix_str = f"{comp.w_KM_mix:.4f}" if comp.w_KM_mix is not None else "  None"
        w_dis_str = f"{comp.w_KM_dis:.4f}" if comp.w_KM_dis is not None else "  None"

        print(f"{name:30s}  {comp._stage1_fallback_count:6d}  "
              f"{comp.w_KM_after:7.4f}  {w_mix_str:>7s}  {w_dis_str:>7s}  "
              f"{comp.Q_dissolve_3:8.3f}  {comp.Q_oil_sump:8.3f}  "
              f"{e_m:+6.2f}  {comp.pc_convergence_gap:7.4f}  "
              f"{'OK' if comp._stage1_fallback_count == 0 else 'FALLBACK'}")
    except Exception as e:
        print(f"{name:30s}  FAILED: {e}")

print("\nExpected: stage1_fb = 0 for all points (especially the first three)")