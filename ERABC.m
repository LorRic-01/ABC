function [opt, hive, time] = ERABC(dim, f, lb, ub, ...
                    n_emp, n_onl, maxIter, hive_i, ...
                    cycle, opts)
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

if nargin < 10 || isempty(opts)
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
gen = @(N) (ub_sat - lb_sat).*logisticEq(4, [N, dim]) + lb_sat;

% Hive intialization
if nargin < 8 || isempty(hive_i) || (size(hive_i, 2) ~= dim)
    hive = gen(n_emp + 1);
else
    hive = [hive_i(1:min(size(hive_i, 1), n_emp + 1), :); gen(max(n_emp + 1 - size(hive_i, 1), 0))];
end
n_nup = zeros(n_emp, 1); indeces = zeros(n_onl, 1); checks = false(n_emp, 1);

% Cost function init
f = @(x) fun_eval(f, x);
cost = f(hive);

[feas, feas_i] = hiveFeas(hive, lb, ub);

% Optimal solution
index = find(cost == min(cost(feas)), 1); if isempty(index), [~, index] = min(cost); end
hive(end, :) = hive(index, :); cost(end) = cost(index);
feas(end) = feas(index); feas_i(end) = feas_i(index);

%% Algorithm
if opts.showFig(1)
    plotHive(opts.nFig, hive, cost, f, lb_sat, ub_sat)
end

lenDisp = 0; init = tic;
for iter = 1:cycle
    if opts.v
        fprintf([repmat('\b', 1, lenDisp)]);
        lenDisp = fprintf('Iter: %d of %d, dt: %.2f, T: %.0fm%.0fs\n',...
            iter, cycle, toc, floor(toc(init)/60), mod(toc(init), 60)); tic
    end

    % RESAMPLE
    fit = zeros(n_emp, 1);
    for k = 1:n_emp
        if cost(k) > 0, fit(k) = 1/(1 + cost(k));
        else, fit(k) = 1 + abs(cost(k)); end
    end
    if sum(fit) < 1e-5, prob = ones(n_emp, 1)/n_emp; else, prob = (1./fit)/sum(1./fit); end
    dens_prob = cumsum(prob);
    for k = 1:n_onl, indeces(k) = find(dens_prob >= rand, 1); end

    % SEARCH
    tmp_hive = hive;
    for k = randperm(N)
        if k <= n_emp, index = k; else, index = indeces(k - n_emp); end
        j = randi(dim, 1);
        
        v = tmp_hive(index, :);
        v(1, j) = v(1, j) + phi()*(v(1, j) - hive(end, j))*cost(end);
        
        [feas_v, feas_iv] = hiveFeas(v, lb, ub);
        [hive(index, :), cost(index), feas(index), feas_i(index, :), check1] = bestSol(hive(index, :), ...
            cost(index), feas(index), feas_i(index, :), v, f(v), feas_v, feas_iv);
        
        v = tmp_hive(index, :);
        v(1, j) = v(1, j) + phi()*(v(1, j) - hive(end, j))*exp(iter/cycle);
        
        [feas_v, feas_iv] = hiveFeas(v, lb, ub);
        [hive(index, :), cost(index), feas(index), feas_i(index, :), check2] = bestSol(hive(index, :), ...
            cost(index), feas(index), feas_i(index, :), v, f(v), feas_v, feas_iv);

        checks(index) = checks(index) | check1 | check2;
    end
    n_nup(checks) = 0; n_nup(~checks) = n_nup(~checks) + 1;
    checks = false(n_emp, 1);
    
    % Optimal solution
    index = find(cost == min(cost(feas)), 1); if isempty(index), [~, index] = min(cost); end
    hive(end, :) = hive(index, :); cost(end) = cost(index);
    feas(end) = feas(index); feas_i(end) = feas_i(index);

    % Plots
    if opts.showFig(2)
        plotHive(opts.nFig, hive, cost, f, lb_sat, ub_sat)
    end

    % REPLACING
    index_nup = n_nup >= maxIter;
    hive(index_nup, :) = gen(sum(index_nup));
    cost(index_nup, :) = f(hive(index_nup, :)); [feas(index_nup), feas_i(index_nup, :)] = hiveFeas(hive(index_nup, :), lb, ub);
    n_nup(index_nup) = 0;
end

%% Solutions
if opts.showFig(3)
    plotHive(opts.nFig, hive, cost, f, lb_sat, ub_sat)
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

    % feas = false(size(hive, 1), 1);
    % feas_i = zeros(size(hive, 1), 1);

    feas = all((hive >= lb) & (hive <= ub), 2);
    
    hive_ub = hive - ub; hive_lb = lb - hive;
    hive_ub(hive <= ub) = 0; hive_lb(hive >= lb) = 0;
    feas_i = sum((hive_ub.^2 + hive_lb.^2), 2);
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
end

function plotHive(nFig, hive, cost, f, lb_sat, ub_sat)
    % Plot hive
    % Input:
    %   hive    - possible solutions        [double(n_emp, dim)]
    %   cost    - cost function             [double(n_emp + 1, 1)]
    %   f       - cost function             [@fun(), mex()]
    %   lb_sat  - saturated lower bound     [double(1, dim)]
    %   ub_sat  - saturated upper bound     [double(1, dim)]

    if size(hive, 2) < 2, second = cost; third = zeros(size(hive, 1), 1);
    else, second = hive(:, 2); third = cost; end

    figure(nFig), hold off
    plot3(hive(1:end-1, 1), second(1:end-1), third(1:end-1), '.m', 'MarkerSize', 10), hold on
    plot3(hive(end, 1), second(end), third(end), '.r', 'MarkerSize', 20)

    if size(hive, 2) < 2
        x = linspace(lb_sat(1), ub_sat(1), 100).';
        y = f(x); plot(x, y, '-k'); 
        xlim([min(x, [], 'all'), max(x, [], 'all')]); ylim([min(y, [], 'all'), max(y, [], 'all')])
        xlabel('x'), ylabel('y'), view([0, 90])
    else
        [x, y] = meshgrid(linspace(lb_sat(1), ub_sat(1), 100), linspace(lb_sat(2), ub_sat(2), 100));
        z = zeros(size(x));
        for k = 1:size(x, 1)
            for n = 1:size(x, 2)
                z(k, n) = f([x(k, n), y(k, n), hive(end, 3:end)]);
            end
        end
        surf(x, y, z)
        xlim([min(lb_sat(1)), max(ub_sat(1))]), ylim([min(lb_sat(2)), max(ub_sat(2))])
        zlim([min(z, [], 'all'), max(z, [], 'all')])
        xlabel('x'), ylabel('y'), zlabel('z')
    end

    grid on, grid on; drawnow
end

function x = logisticEq(mu, x_size, x0)
    % Generate random vector using logistic equation
    x = zeros(x_size);
    if x_size(1)*x_size(2) == 0, return, end
    if nargin < 3 || isempty(x0), x0 = rand; end
    while any(x0 == [0, 0.25, 0.5, 0.75, 1]), x0 = rand; end

    x(1) = x0;
    for k = 2:numel(x)
        x(k) = mu*x(k-1)*(1 - x(k-1));
        while any(x(k) == [0, 0.25, 0.5, 0.75, 1]), x(k) = rand; end
    end
end
    
    