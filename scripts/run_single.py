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


def pick_model(model_name: str, N_max: float, V_h: float):
    m = model_name.lower()
    if m in ("orig", "original"):
        return Molinaroli_2017_Compressor(N_max=N_max, V_h=V_h)
    if m in ("mod", "modified"):
        return Molinaroli_2017_Compressor_Modified(N_max=N_max, V_h=V_h)
    raise ValueError("Unknown --model. Use: original | modified")


def main():
    ap = argparse.ArgumentParser(description="Run Molinaroli compressor (original/modified)")
    ap.add_argument("--model", default="original", help="original | modified")
    ap.add_argument("--refrigerant", default="R290")

    ap.add_argument("--p_suc", type=float, required=True, help="Suction pressure [Pa]")
    ap.add_argument("--T_suc", type=float, required=True, help="Suction temperature [K]")
    ap.add_argument("--p_outlet", type=float, required=True, help="Outlet pressure [Pa]")
    ap.add_argument("--n", type=float, required=True, help="Speed input (often relative 0..1)")
    ap.add_argument("--T_amb", type=float, required=True, help="Ambient temperature [K]")

    ap.add_argument("--N_max", type=float, required=True, help="Max speed parameter for baseclass")
    ap.add_argument("--V_h", type=float, required=True, help="Displacement volume [m^3]")

    args = ap.parse_args()

    comp = pick_model(args.model, args.N_max, args.V_h)

    # Media backend
    comp.med_prop = CoolProp(fluid_name=args.refrigerant)

    # Inlet state
    comp.state_inlet = comp.med_prop.calc_state("PT", args.p_suc, args.T_suc)

    # Inputs wrapper (because model expects inputs.control.n and inputs.T_amb)
    inputs = SimpleInputs(control=Control(n=args.n), T_amb=args.T_amb)
    fs_state = FlowsheetState()

    comp.calc_state_outlet(p_outlet=args.p_outlet, inputs=inputs, fs_state=fs_state)

    print(f"model  = {args.model}")
    print(f"m_flow = {comp.m_flow:.6e} kg/s")
    print(f"P_el   = {comp.P_el:.3f} W")
    print(f"T_dis  = {comp.state_outlet.T:.3f} K")


if __name__ == "__main__":
    main()
