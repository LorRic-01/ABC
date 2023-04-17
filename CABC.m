function [opt, hive, time] = CABC(dim, f, lb, ub, g, h, ...
                    n_emp, n_onl, maxIter, hive_i, ...
                    cycle, opts)
% CABC - Constrained Artificial Bee Colony (CABC) optimization algorithm
% Optimization algorithm solving
%   argmin f(x)     subject to:     lb <= x <= ub (input bounds)
%      x                                g(x) = 0  (equality constraint)
%
% Policy used:
%   - input bounds > equality constraints > cost
%
% Problem settings:
%   dim - problem dimension         [-]
%   f   - cost function             [@fun(), mex()]
%   lb  - solution lower bound      [-inf (default) | double(1, dim)]
%   ub  - solution upper bouond     [inf (default) | double(1, dim)]
%   g   - equality constraints      [[] (default) | @fun(), mex()]
%   h   - inequality constraints    [[] (default) | @fun(), mex()]
%
% Colony settings:
%   n_emp   - # employed bees                   [100 (default)]
%   n_onl   - # onlooker bees                   [100 (default)]
%   maxIter - max # of iter. before rejection   [25 (default)]
%   hive_i  - hive initialization               [double(n_i, dim)]
%
% Options:
%   cycle   - # of optimization cycle           [100 (default)]
%   opts    - struct
%               'nFig'      - figure #                          [1 (default)]
%               'showFig'   - show figure ([init, loop, end])   [[false, false, true] (default)]
%               'v'         - verbose                           [false (default)]
%
% Output:
%   opt     - optimal solution      [double(1, dim)]
%   hive    - set of solutions      [double(n_emp, dim)]
%   time    - computation time      [double]

%% Default parameters
phi = @(n) 2*rand(1, n) - 1;   % random function in [-1, 1]

%% Initialization
if nargin < 2 || isempty(dim) || isempty(f)
    error('Missing problem dimension and/or cost function declaration')
end

if nargin < 3 || isempty(lb) || (size(lb, 2) ~= dim), lb = -inf(1, dim); end
if nargin < 4 || isempty(ub) || (size(ub, 2) ~= dim), ub = inf(1, dim); end
if nargin < 5 || isempty(g), g = @(x) 0; end
if nargin < 6 || isempty(h), h = @(x) 0; end
if nargin < 7 || isempty(n_emp), n_emp = 100; else, n_emp = n_emp(1); end
if nargin < 8 || isempty(n_onl), n_onl = 100; else, n_onl = n_onl(1); end
if nargin < 10 || isempty(maxIter), maxIter = 25; else, maxIter = maxIter(1); end
if nargin < 11 || isempty(cycle), cycle = 100; else, cycle = cycle(1); end

if nargin < 12 || isempty(opts)
    opts = struct('nFig', 1, 'showFig', [false, false, true], 'v', false);
else
    fields = {'nFig'; 'showFig'; 'v'};
    missing = ~strcmp(fieldnames(opts), fields);
    if missing(1), opts.nFig = 1; end
    if missing(2) || (numel(opts.showFig) ~= 3), opts.showFig = [false, false, true]; end
    if missing(3), opts.v = false; end
end

% Hive gen function
N = n_emp + n_onl;
ub_sat = ub; ub_sat(isinf(ub)) = 100;
lb_sat = lb; lb_sat(isinf(lb)) = -100;
gen = @(N) (ub_sat - lb_sat).*rand(N, dim) + lb_sat;

% Hive intialization
if nargin < 10 || isempty(hive_i) || (size(hive_i, 2) ~= dim)
    hive = gen(n_emp + 1);
else
    hive = [hive_i(1:min(size(hive_i, 1), n_emp + 1), :); gen(max(n_emp + 1 - size(hive_i, 1), 0))];
end
n_nup = zeros(n_emp, 1); indeces = zeros(n_onl, 1); checks = false(n_emp, 1);

% Cost function
f = @(x) fun_eval(f, x); cost = f(hive);
g = @(x) fun_eval(g, x); h = @(x) fun_eval(h, x);

% Solution feasibility
[feas, feas_i] = hiveFeas(hive, lb, ub, g, h);

% Optimal solution
[~, index] = sortrows([feas_i, cost]);
hive(end, :) = hive(index(1), :); cost(end) = cost(index(1));
feas(end) = feas(index(1)); feas_i(end, :) = feas_i(index(1), :);

%% Algorithm
if opts.showFig(1)
    plotHive(opts.nFig, hive, cost, f, h, lb_sat, ub_sat)
end

lenDisp = 0; init = tic;
for iter = 1:cycle
    if opts.v
        fprintf([repmat('\b', 1, lenDisp)]);
        lenDisp = fprintf('Iter: %d of %d, dt: %.2f, T: %.0fm%.0fs\n',...
            iter, cycle, toc, floor(toc(init)/60), mod(toc(init), 60)); tic
    end

    % RESAMPLE
    fit = fitness(cost, feas);
    if sum(fit) < 1e-5, prob = ones(n_emp, 1)/n_emp; else, prob = fit/sum(fit); end
    dens_prob = cumsum(prob);
    for k = 1:n_onl, indeces(k) = find(dens_prob >= rand, 1); end

    % SEARCH
    tmp_hive = hive;
    for tmp = randperm(N)
        if tmp > n_emp, k = indeces(tmp - n_emp); else, k = tmp; end
        n = k; while n == k, n = randi(n_emp, 1); j = randperm(dim, randi(dim, 1)); end

        v = tmp_hive(k, :);
        v(1, j) = v(1, j) + phi(length(j)).*(v(1, j) - tmp_hive(n, j));

        [feas_v, feas_iv] = hiveFeas(v, lb, ub, g, h);
        [hive(k, :), cost(k), feas(k), feas_i(k, :), check] = ...
            bestSol(hive(k, :), cost(k), feas(k), feas_i(k, :), v, f(v), feas_v, feas_iv);

        checks(k) = checks(k) | check;

    end
    n_nup(checks) = 0; n_nup(~checks) = n_nup(~checks) + 1;
    checks = false(n_emp, 1);

    % REPLACING
    index_nup = n_nup >= maxIter;
    hive(index_nup, :) = gen(sum(index_nup));
    cost(index_nup, :) = f(hive(index_nup, :));
    [feas(index_nup), feas_i(index_nup, :)] = hiveFeas(hive(index_nup, :), lb, ub, g, h);
    n_nup(index_nup) = 0;

    % Optimal solution
    [~, index] = sortrows([feas_i, cost]);
    hive(end, :) = hive(index(1), :); cost(end) = cost(index(1));
    feas(end) = feas(index(1)); feas_i(end, :) = feas_i(index(1), :);
    
    % Plots
    if opts.showFig(2)
        plotHive(opts.nFig, hive, cost, f, h, lb_sat, ub_sat)
    end
end

%% Solutions
if opts.showFig(3)
    plotHive(opts.nFig, hive, cost, f, h, lb_sat, ub_sat)
end
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
    
    if isempty(x), f_x = []; return, end
    f_x = zeros(size(x, 1), length(f(x(1, :))));
    for k = 1:size(f_x, 1)
        f_x(k, :) = f(x(k, :));
    end
end

function fit = fitness(fit, feas)
    % Compute the fitness function given of a given vector
    % Input:
    %   fit     - input vector      [double(n, 1)]
    % Output:
    %   fit     - fitness values    [double(n, 1)]
    
    % Check
    if length(fit) ~= numel(fit)
        error('Passed matrix to fitness evaluation. Needed: vector')
    end
    if length(fit) ~= length(feas)
        error('Check feasibility vector dimension')
    end
    
    if any(feas), fit(~feas) = abs(fit(~feas)) + max(fit(feas));
    else, fit(~feas) = abs(fit(~feas)) + fit(end); end
    fit(fit >= 0) = 1./(1 + fit(fit >= 0));
    fit(fit < 0) = 1 + abs(fit(fit < 0));
    fit = fit(1:end-1);
end

function [feas, feas_i] = hiveFeas(hive, lb, ub, g, h)
    % Compute the feasibility of solutions
    % Input:
    %   hive    - solutions                 [double(n, dim)]
    %   lb      - solution lower bound      [-inf (default) | double(1, dim)]
    %   ub      - solution upper bouond     [inf (default) | double(1, dim)]
    %   g       - equality constr. fun.     [@fun(), mex()]
    %   h       - inequality constr. fun.   [@fun(), mex()]
    % Output:
    %   feas    - sol feasibility           [logic(n, 1)]
    %   feas_i  - feasibility indeces       [double(n, m)]
    
    if isempty(hive), feas = []; feas_i = []; return, end

    g_hive = g(hive); h_hive = h(hive);
    feas_i = zeros(size(hive, 1), 2);
    
    feas = all((hive >= lb) & (hive <= ub), 2) & all(~g_hive, 2) & all(h_hive <= 0, 2);

    % Input constraints
    hive_ub = hive - ub; hive_lb = lb - hive;
    hive_ub(hive <= ub) = 0; hive_lb(hive >= lb) = 0;
    feas_i(:, 1) = sum(hive_ub.^2 + hive_lb.^2, 2);

    % Inequality constraints
    feas_i(:, 2) = sum(abs(max(h_hive, 0)), 2);
    % Equality constraints
    feas_i(:, 3) = sum(abs(g_hive), 2);
end

function plotHive(nFig, hive, cost, f, h, lb_sat, ub_sat)
    % Plot hive
    % Input:
    %   hive    - possible solutions        [double(n_emp, dim)]
    %   cost    - cost function             [double(n_emp + 1, 1)]
    %   f       - cost function             [@fun(), mex()]
    %   g       - equality constr. fun.     [@fun(), mex()]
    %   h       - inequality constr. fun.   [@fun(), mex()]
    %   lb_sat  - saturated lower bound     [double(1, dim)]
    %   ub_sat  - saturated upper bound     [double(1, dim)]
    
    r = 100; % resolution

    if size(hive, 2) < 2, second = cost; third = zeros(size(hive, 1), 1);
    else, second = hive(:, 2); third = cost; end

    figure(nFig), hold off
    plot3(hive(1:end-1, 1), second(1:end-1), third(1:end-1), '.m', 'MarkerSize', 10), hold on
    plot3(hive(end, 1), second(end), third(end), '.r', 'MarkerSize', 20)

    if ~isempty(f)
        if size(hive, 2) < 2
            x = linspace(lb_sat(1), ub_sat(1), r).'; h_x = h(x);
            y = f(x); plot(x(all(h_x <= 0, 2)), y(all(h_x <= 0, 2)), '-k');
            xlim([lb_sat(1), ub_sat(1)]), ylim([min(y, [], 'all'), max(y, [], 'all')])
        else
            [x, y] = meshgrid(linspace(lb_sat(1), ub_sat(1), r), linspace(lb_sat(2), ub_sat(2), r));
            z = zeros(size(x));
            for k = 1:size(x, 1)
                for n = 1:size(x, 2)
                    if any(h([x(k, n), y(k, n), hive(end, 3:end)]) > 0), z(k, n) = NaN;
                    else, z(k, n) = f([x(k, n), y(k, n), hive(end, 3:end)]); end
                end
            end
            surf(x, y, z)
            xlim([lb_sat(1), ub_sat(1)]), ylim([lb_sat(2), ub_sat(2)])
            zlim([min(z, [], 'all'), max(z, [], 'all')])
        end
    end

    xlabel('x'), ylabel('y')
    if size(hive, 2) > 2, zlabel('z'), end

    view([0, 90]); grid on, grid on; drawnow
end

function [sol, cost, feas, feas_i, check] = bestSol(sol1, cost1, feas1, feas_i1, sol2, cost2, feas2, feas_i2)
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
    %   check       - true if 2nd sol is better         [logical]
    
    [tmp, index] = sortrows([-feas1, feas_i1, cost1; -feas2, feas_i2, cost2]);
    feas = tmp(1, 1); cost = tmp(1, end);
    
    switch index(1)
        case 1, sol = sol1; feas_i = feas_i1; check = false;
        case 2, sol = sol2; feas_i = feas_i2; check = true;
    end

    % if norm(sol - sol2) < 1e-2, check = false; end
end