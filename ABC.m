function [opt, hive, time] = ABC(dim, f, lb, ub, ...
                    n_emp, n_onl, maxIter, hive_i, ...
                    cycle)
% ABC - Artificial Bee Colony (ABC) classic optimization algorithm
% Optimization algorithm solving
%   argmin f(x)     subject to:     lb <= x <= ub (input bounds)
%      x
%
% Problem settings:
%   dim - problem dimension         [-]
%   f   - cost function             [@fun(), mex()]
%   lb  - solution lower bound      [-inf (default) | double(1, dim)]
%   ub  - solution upper bouond     [inf (default) | double(1, dim)]
%
% Colony settings:
%   n_emp   - # employed bees                   [100 (default)]
%   n_onl   - # onlooker bees                   [100 (default)]
%   maxIter - max # of iter. before rejection   [25 (default)]
%   hive_i  - hive initialization               [double(n_i, dim)]
%
% Options:
%   cycle   - # of optimization cycle           [100 (default)]
%
% Output:
%   opt     - optimal solution      [double(1, dim)]
%   hive    - set of solutions      [double(n_emp + n_onl, dim)]
%   time    - computation time      [double]

%% Default parameters
phi = @() 2*rand - 1;   % random function in [-1, 1]

%% Initialization
if nargin < 2 || isempty(dim) || isempty(f)
    error('Missing problem dimension and/or cost function declaration')
end

if nargin < 3 || isempty(lb) || (size(lb, 2) ~= dim), lb = -inf(1, dim); end
if nargin < 4 || isempty(ub) || (size(ub, 2) ~= dim), ub = inf(1, dim); end
if nargin < 5 || isempty(n_emp), n_emp = 100; else, n_emp = n_emp(1); end
if nargin < 6 || isempty(n_onl), n_onl = 100; else, n_onl = n_onl(1); end
if nargin < 7 || isempty(maxIter), maxIter = 25; else, maxIter = maxIter(1); end
if nargin < 9 || isempty(cycle), cycle = 100; else, cycle = cycle(1); end

% Hive gen function
n = n_emp + n_onl;      % # of bees
ub_sat = min(100, ub); lb_sat = max(-100, lb);
gen = @(n) (ub_sat - lb_sat).*rand(n, dim) + lb_sat;

% HIve intialization
if nargin < 8 || isempty(hive_i) || (size(hive_i, 2) ~= dim)
    hive = gen(n + 1);
else
    hive = [hive_i(min(size(hive_i, 1), n + 1)); gen(max(n + 1 - size(hive_i, 1), 0))];
end
n_nup = zeros(n, 1);

% Cost function init
f = @(x) fun_eval(f, x);
cost = f(hive);

[feas, feas_i] = hiveFeas(hive, lb, ub);

% Optimal solution
index = find(cost == min(cost(feas))); if isempty(index), [~, index] = min(cost); end
hive(end, :) = hive(index, :); cost(end) = cost(index);
feas(end) = feas(index); feas_i(end) = feas_i(index);

%% Algorithm
lenDisp = 0; init = tic;
for iter = 1:cycle
    fprintf([repmat('\b', 1, lenDisp)]);
    lenDisp = fprintf('Iter: %d of %d, dt: %.2f\n', iter, cycle, toc); tic
    
    % RESAMPLE
    fit = zeros(n_emp, 1);
    for k = 1:n_emp
        if cost(k) > 0, fit(k) = 1/(1 + cost(k));
        else, fit(k) = 1 + abs(cost(k)); end
    end
    % Probability and density function
    if sum(fit) < 1e-5, prob = ones(n_emp, 1)/n_emp;
    else, prob = fit/sum(fit); end
    dens_prob = cumsum(prob);

    for k = 1:n_onl
        index = find(dens_prob >= rand, 1);
        hive(n_emp + k, :) = hive(index, :); cost(n_emp + k) = cost(index);
        feas(n_emp + k) = feas(index); feas_i(n_emp + k) = feas_i(index);
    end
    
    % SEARCH
    for k = randperm(n)
        i = k; while i == k, i = randi(n, 1); j = randi(dim, 1); end
        
        v = hive(k, :);
        v(1, j) = v(1, j) + phi()*(v(1, j) - hive(i, j));
        
        [feas_v, feas_iv] = hiveFeas(v, lb, ub);
        [cost(k), hive(k, :), feas(k), feas_i(k, :), check] = bestSol(hive(k, :), ...
            cost(k), feas(k), feas_i(k, :), v, f(v), feas_v, feas_iv);

        if check, n_nup(k) = n_nup(k) + 1; else, n_nup(k) = 0; end
    end

    % Optimal solution
    index = find(cost == min(cost(feas)), 1); if isempty(index), [~, index] = min(cost); end
    hive(end, :) = hive(index, :); cost(end) = cost(index);
    feas(end) = feas(index); feas_i(end) = feas_i(index);


    figure(1)
    plot3(hive(:, 1), hive(:, 2), cost, '.b', 'MarkerSize', 10)
    view([0, 90]), drawnow

    % REPLACING
    index_nup = n_nup >= maxIter;
    hive(index_nup, :) = gen(sum(index_nup));
    cost(index_nup, :) = f(hive(index_nup, :)); [feas(index_nup), feas_i(index_nup, :)] = hiveFeas(hive(index_nup, :), lb, ub);
    n_nup(index_nup) = zeros(sum(index_nup));
end

%% Solutions
opt = hive(end, :); time = toc(init);

end

%% Support functions
function f_x = fun_eval(f, x)
    % Evaluate function data
    % Input:
    %   f   - function    [@fun(), mex()]
    %   x   - data        [double(n, dim)]
    % Output:
    %   f_x - f(x)      [double(n, 1)]

    f_x = zeros(size(x, 1), 1);
    for k = 1:length(f_x)
        f_x(k) = f(x(k, :));
    end
end

function [feas, feas_i] = hiveFeas(hive, lb, ub)
    % Return feasible solutions
    % Input:
    %   hive    - solutions                     [double(n, dim)]
    %   lb      - solution lower bound          [double(1, dim)]
    %   ub      - solution upper bound          [double(1, dim)]
    % Output:
    %   feas    - true if sol. is feasible      [logical(n, 1)]
    %   feas_i  - feasibility index (0 = feas)  [double(n, 1)]

    feas = false(size(hive, 1), 1);
    feas_i = zeros(size(hive, 1), 1);

    feas = all((hive >= lb) & (hive <= ub), 2);
    
    hive_ub = hive - ub; hive_lb = lb - hive;
    hive_ub(hive <= ub) = 0; hive_lb(hive >= lb) = 0;
    feas_i = sum((hive_ub.^2 + hive_lb.^2), 2);
end

function [cost, sol, feas, feas_i, check] = bestSol(sol1, cost1, feas1, feas_i1, sol2, cost2, feas2, feas_i2)
    % Find best solution between provided
    % Input:
    %   sol...      - solution                  [double(1, dim)]
    %   cost...     - solution cost             [double]
    %   feas...     - feasible                  [logic]
    %   feas_i...   - feasibility indeces       [double(1, ...)]
    % Output:
    %   sol         - best solution                     [double(1, dim)]
    %   cost        - best solution cost                [double]
    %   feas        - best sol. feasible                [logic]
    %   feas_i      - best sol. feasibility indeces     [double(1, ...)]
    
    check = NaN;
    switch 2*feas1 + feas2
        case 0 % both unfeasible -> best with less constrained violation
            for k = 1:length(feas_i1)
                if feas_i1(k) > feas_i2(k), check = false; break
                elseif feas_i1(k) < feas_i2(k), check = true; break, end
            end
            if isnan(check), check = false; end
        case 1 % sol2 feasible
            check = true;
        case 2 % sol1 feasible
            check = false;
        case 3 % both feasibles -> best with less cost
            if cost1 > cost2, check = true; else, check = false; end 
    end
    
    if check
        cost = cost2; sol = sol2; feas = feas2; feas_i = feas_i2;
    else
        cost = cost1; sol = sol1; feas = feas1; feas_i = feas_i1;
    end

    if check
            1;
        end
end