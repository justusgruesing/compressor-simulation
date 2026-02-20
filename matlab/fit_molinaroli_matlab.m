function fit_molinaroli_matlab(csv_path, oil, model, refrigerant, x0_csv, out_dir, maxFunEvals, pythonExe)
% fit_molinaroli_matlab
% MATLAB Optimization Toolbox wrapper that minimizes Molinaroli objective g
% (exactly as in Molinaroli et al. 2017, eq. 31–33)
% by calling the Python model (cached) via scripts/molinaroli_matlab_bridge.py
%
% Notes (important):
% - V_IC is allowed only within ±5% around the *start value* used for V_IC.
%   The start value is taken from x0_csv if provided, otherwise from defaults.
% - Finite-difference step sizes are set explicitly (vector) to avoid "numerical flatness".
% - Problem scaling is enabled via TypicalX + ScaleProblem.
%
% Example:
% fit_molinaroli_matlab("data/Datensatz_Fitting_1.csv","LPG68","original","PROPANE", ...
%                       "data/start_params.csv","results/matlab_fit", 5000, "")
%
% matlab -batch "addpath('matlab'); fit_molinaroli_matlab('data/Datensatz_Fitting_1.csv','LPG68','original','PROPANE','data/start_params.csv','results/matlab_fit',5000,'');"


arguments
    csv_path (1,1) string
    oil (1,1) string = "all"
    model (1,1) string = "original"
    refrigerant (1,1) string = "PROPANE"
    x0_csv (1,1) string = ""
    out_dir (1,1) string = "results/matlab_fit"
    maxFunEvals (1,1) double = 5000
    pythonExe (1,1) string = ""
end

% --- 1) Configure MATLAB->Python
if strlength(pythonExe) > 0
    pyenv("Version", pythonExe);
end

% Add repo root to py.sys.path (so `scripts.*` is importable)
thisFile = mfilename("fullpath");
repoRoot = fileparts(fileparts(thisFile)); % .../matlab/fit_molinaroli_matlab.m -> repo root
if ~any(strcmp(string(py.sys.path), repoRoot))
    insert(py.sys.path, int32(0), repoRoot);
end

% Import & reload bridge
bridge = py.importlib.import_module("scripts.molinaroli_matlab_bridge");
py.importlib.reload(bridge);

% init cache once
bridge.init(char(csv_path), char(oil), char(model), char(refrigerant));

% --- 2) Parameter names + x0 (start values)
paramNames = ["Ua_suc_ref","Ua_dis_ref","Ua_amb","A_tot","A_dis","V_IC","alpha_loss","W_dot_loss_ref"];

% Default start values (paper-like). If x0_csv is given, it overrides these.
V_IC_default = 30.7e-6; % 30.7 cm³ = 30.7e-6 m³
x0_default = [16.05, 13.96, 0.36, 9.47e-9, 86.1e-6, V_IC_default, 0.16, 83.0];

x0 = x0_default;

% Override x0 using x0_csv if provided
if strlength(x0_csv) > 0
    T = readtable(x0_csv);
    if height(T) ~= 1
        error("x0_csv must contain exactly one row.");
    end
    for i = 1:numel(paramNames)
        if ismember(paramNames(i), string(T.Properties.VariableNames))
            v = T{1, char(paramNames(i))};
            if ~isnan(v)
                x0(i) = v;
            end
        end
    end
end

% --- 3) Bounds
% Global (physical/numerical) bounds for all parameters
lb = [0.01, 0.01, 0.0,   1e-12, 1e-8,  1e-9,  0.0,  0.0];
ub = [500.0, 500.0, 50.0, 1e-6,  1e-3,  1e-3,  1.0,  5000.0];

% Constrain V_IC within ±5% around the actual start value (from x0_csv if present)
V_IC0 = x0(6);
if ~(isfinite(V_IC0) && V_IC0 > 0)
    error("Invalid start value for V_IC (x0(6)). Must be finite and > 0.");
end
lb(6) = 0.95 * V_IC0;
ub(6) = 1.05 * V_IC0;

% Clamp x0 to bounds (must be feasible for fmincon start)
x0 = max(lb, min(ub, x0));

% Optional: print active V_IC bounds for traceability
fprintf("V_IC start = %.8e m^3, bounds = [%.8e, %.8e]\n", x0(6), lb(6), ub(6));

% --- 4) Objective handle (calls Python cost)
fun = @(x) double( bridge.cost(py.list(num2cell(x))) );

% --- 5) Finite-difference step sizes (vector) + scaling
% Step sizes chosen to be "felt" by the Python model.
% V_IC step is relative to the start value (1%), with a small lower bound.
fdStep = [ ...
    0.1, ...                       % Ua_suc_ref
    0.1, ...                       % Ua_dis_ref
    0.05, ...                      % Ua_amb
    5e-10, ...                     % A_tot
    1e-6, ...                      % A_dis
    max(1e-8, 0.01 * V_IC0), ...    % V_IC (1% of start)
    0.01, ...                      % alpha_loss
    1.0  ...                       % W_dot_loss_ref
];

% TypicalX for scaling: use |x0| as typical magnitude (robust for your scale spread)
typX = max(1e-12, abs(x0));

% --- 6) Run optimizer (MATLAB Optimization Toolbox)
opts = optimoptions("fmincon", ...
    "Algorithm","interior-point", ...
    "Display","iter", ...
    "MaxFunctionEvaluations", maxFunEvals, ...
    "FiniteDifferenceType","central", ...
    "FiniteDifferenceStepSize", fdStep, ...
    "TypicalX", typX, ...
    "ScaleProblem","obj-and-constr", ...
    "StepTolerance", 1e-12);

[xbest, fbest, exitflag, output] = fmincon(fun, x0, [],[],[],[], lb, ub, [], opts);

% --- pick best feasible solution if available (important for interior-point)
xsave = xbest;
fsave = fbest;

if isfield(output,"bestfeasible") && ~isempty(output.bestfeasible)
    if isfield(output.bestfeasible,"x") && ~isempty(output.bestfeasible.x)
        xsave = output.bestfeasible.x;
    end
    if isfield(output.bestfeasible,"fval") && ~isempty(output.bestfeasible.fval)
        fsave = output.bestfeasible.fval;
    end
end

fprintf("\n=== MATLAB FIT DONE ===\n");
fprintf("g (objective) = %.6e\n", fsave);
fprintf("exitflag = %d\n", exitflag);
disp(output);


% --- 7) Save results via Python (same CSV format as your pipeline)
if ~isfolder(out_dir)
    mkdir(out_dir);
end
run_id = datestr(now, "yyyy-mm-dd_HHMMSS");
tag = "matlab_fmincon";

out_params = fullfile(out_dir, sprintf("fitted_params_%s_%s_%s_%s.csv", lower(oil), lower(model), tag, run_id));
out_pred   = fullfile(out_dir, sprintf("fit_predictions_%s_%s_%s_%s.csv", lower(oil), lower(model), tag, run_id));

% Pass x0 too so fitted_params CSV contains x0_* columns (requires updated bridge)
bridge.save_results(py.list(num2cell(xsave)), out_params, out_pred, tag, py.list(num2cell(x0)));

fprintf("Saved params: %s\n", out_params);
fprintf("Saved pred  : %s\n", out_pred);
end
