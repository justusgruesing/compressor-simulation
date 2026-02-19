function fit_molinaroli_matlab(csv_path, oil, model, refrigerant, x0_csv, out_dir, maxFunEvals, pythonExe)
% fit_molinaroli_matlab
% MATLAB Optimization Toolbox wrapper that minimizes Molinaroli objective g
% (exactly as in Molinaroli et al. 2017, eq. 31–33)
% by calling the Python model (cached) via scripts/molinaroli_matlab_bridge.py
%
% Example:
% fit_molinaroli_matlab("data/Datensatz_Fitting_1.csv","LPG68","original","R290", ...
%                       "data/start_params.csv","results/matlab_fit", 5000, ...
%                       "C:\...\envs\molinaroli\python.exe")

arguments
    csv_path (1,1) string
    oil (1,1) string = "all"
    model (1,1) string = "original"
    refrigerant (1,1) string = "R290"
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

% init cache once (NOTE: bridge.init signature no longer has T_dis options)
bridge.init(char(csv_path), char(oil), char(model), char(refrigerant));

% --- 2) x0 (start values)
paramNames = ["Ua_suc_ref","Ua_dis_ref","Ua_amb","A_tot","A_dis","V_IC","alpha_loss","W_dot_loss_ref"];
x0_default = [16.05, 13.96, 0.36, 9.47e-9, 86.1e-6, 16.11e-6, 0.16, 83.0];

x0 = x0_default;
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

% --- 3) Bounds (like your SciPy version)
lb = [0.01, 0.01, 0.0,   1e-12, 1e-8,  1e-9,  0.0,  0.0];
ub = [500.0, 500.0, 50.0, 1e-6,  1e-3,  1e-3,  1.0,  5000.0];

x0 = max(lb, min(ub, x0));

% --- 4) Objective handle (calls Python cost)
fun = @(x) double( bridge.cost(py.list(num2cell(x))) );

% --- 5) Run optimizer (MATLAB Optimization Toolbox)
opts = optimoptions("fmincon", ...
    "Algorithm","interior-point", ...
    "Display","iter", ...
    "MaxFunctionEvaluations", maxFunEvals);

[xbest, fbest, exitflag, output] = fmincon(fun, x0, [],[],[],[], lb, ub, [], opts);

fprintf("\n=== MATLAB FIT DONE ===\n");
fprintf("g (objective) = %.6e\n", fbest);
fprintf("exitflag = %d\n", exitflag);
disp(output);

% --- 6) Save results via Python (same CSV format as your pipeline)
if ~isfolder(out_dir)
    mkdir(out_dir);
end
run_id = datestr(now, "yyyy-mm-dd_HHMMSS");
tag = "matlab_fmincon";

out_params = fullfile(out_dir, sprintf("fitted_params_%s_%s_%s_%s.csv", lower(oil), lower(model), tag, run_id));
out_pred   = fullfile(out_dir, sprintf("fit_predictions_%s_%s_%s_%s.csv", lower(oil), lower(model), tag, run_id));

bridge.save_results(py.list(num2cell(xbest)), out_params, out_pred, tag);

fprintf("Saved params: %s\n", out_params);
fprintf("Saved pred  : %s\n", out_pred);
end
