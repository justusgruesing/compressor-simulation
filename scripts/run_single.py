import argparse
from dataclasses import dataclass

from vclibpy.media.cool_prop import CoolProp
from vclibpy.datamodels import FlowsheetState
from vclibpy.components.compressors import (
    Molinaroli_2017_Compressor,
    Molinaroli_2017_Compressor_Modified,
)


@dataclass
class Control:
    n: float


@dataclass
class SimpleInputs:
    control: Control
    T_amb: float  # K


# Keep defaults here so we can safely override only some parameters (e.g., m_dot_ref)
DEFAULT_PARAMS = {
    "Ua_suc_ref": 16.05,
    "Ua_dis_ref": 13.96,
    "Ua_amb": 0.36,
    "A_tot": 9.47e-9,
    "A_dis": 86.1e-9,
    "V_IC": 16.11e-6,
    "alpha_loss": 0.16,
    "W_dot_loss_ref": 83,
    "m_dot_ref": 0.0083,  # will be overwritten by computed value
    "f_ref": 50.0,        # can be overridden by CLI
}


def pick_model(model_name: str, N_max: float, V_h: float, parameters: dict):
    m = model_name.lower().strip()
    if m in ("orig", "original"):
        return Molinaroli_2017_Compressor(N_max=N_max, V_h=V_h, parameters=parameters)
    if m in ("mod", "modified"):
        return Molinaroli_2017_Compressor_Modified(N_max=N_max, V_h=V_h, parameters=parameters)
    raise ValueError("Unknown --model. Use: original | modified")


def calc_rho_sat_vapor(med: CoolProp, T_ref: float) -> float:
    """
    Calculate saturated vapor density at temperature T_ref (quality Q=1).
    Tries 'TQ' first, falls back to 'QT' in case the wrapper expects swapped order.
    """
    try:
        state_ref = med.calc_state("TQ", T_ref, 1.0)  # T [K], Q=1
        return state_ref.d
    except Exception:
        # Some wrappers use "QT" with (Q, T) ordering
        state_ref = med.calc_state("QT", 1.0, T_ref)  # Q=1, T [K]
        return state_ref.d


def main():
    ap = argparse.ArgumentParser(description="Run Molinaroli compressor (original/modified)")

    ap.add_argument("--model", default="original", help="original | modified")
    ap.add_argument("--refrigerant", default="R290")

    # Operating point inputs
    ap.add_argument("--p_suc", type=float, required=True, help="Suction pressure [Pa]")
    ap.add_argument("--T_suc", type=float, required=True, help="Suction temperature [K]")
    ap.add_argument("--p_outlet", type=float, required=True, help="Outlet pressure [Pa]")
    ap.add_argument("--n", type=float, required=True, help="Relative speed (0..1). Absolute = N_max*n.")
    ap.add_argument("--T_amb", type=float, required=True, help="Ambient temperature [K]")

    # Compressor constants
    ap.add_argument("--N_max", type=float, required=True, help="Max compressor frequency [1/s] (rounds/sec)")
    ap.add_argument("--V_h", type=float, required=True, help="Displacement volume [m^3]")

    # Reference settings for m_dot_ref calculation
    ap.add_argument("--f_ref", type=float, default=50.0, help="Reference frequency f_ref [1/s]")
    ap.add_argument("--T_ref", type=float, default=273.15, help="Reference temperature for rho_sat_vapor [K]")

    args = ap.parse_args()

    # Media backend (needed to compute rho_sat_vapor)
    med = CoolProp(fluid_name=args.refrigerant)

    # Compute m_dot_ref = rho_sat_vapor(T_ref) * V_h * f_ref
    rho_ref = calc_rho_sat_vapor(med, args.T_ref)
    m_dot_ref = rho_ref * args.V_h * args.f_ref

    # Build parameters dict: start from defaults and override f_ref and m_dot_ref
    params = DEFAULT_PARAMS.copy()
    params["f_ref"] = args.f_ref
    params["m_dot_ref"] = m_dot_ref

    # Instantiate compressor (original or modified) with overridden params
    comp = pick_model(args.model, args.N_max, args.V_h, parameters=params)

    # Attach media backend
    comp.med_prop = med

    # Inlet state from (p_suc, T_suc)
    comp.state_inlet = comp.med_prop.calc_state("PT", args.p_suc, args.T_suc)

    # Inputs wrapper (model expects inputs.control.n and inputs.T_amb)
    inputs = SimpleInputs(control=Control(n=args.n), T_amb=args.T_amb)
    fs_state = FlowsheetState()

    # Run simulation
    comp.calc_state_outlet(p_outlet=args.p_outlet, inputs=inputs, fs_state=fs_state)

    # Print key outputs + helpful reference info
    print(f"model    = {args.model}")
    print(f"fluid    = {args.refrigerant}")
    print(f"f_ref    = {args.f_ref:.6g} 1/s")
    print(f"T_ref    = {args.T_ref:.3f} K")
    print(f"rho_ref  = {rho_ref:.6g} kg/m^3")
    print(f"m_dot_ref= {m_dot_ref:.6g} kg/s")
    print()
    print(f"m_flow   = {comp.m_flow:.6e} kg/s")
    print(f"P_el     = {comp.P_el:.3f} W")
    print(f"T_dis    = {comp.state_outlet.T:.3f} K")


if __name__ == "__main__":
    main()

