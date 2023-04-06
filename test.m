% Benchmark functions
% @(x) -20*exp(-0.2*sqrt(0.5*(x(1).^2 + x(2).^2))) - exp(0.5*(cos(2*pi*x(1)) + cos(2*pi*x(2)))) + exp(1) + 20;
% @(x) sum(x.^4 - 16*x.^2 + 5*x, 2)/2;

%% Initialization
dim = 1;
f = @(x) sum(x.^4 - 16*x.^2 + 5*x, 2)/2;
lb = -10*ones(1, dim);
ub = 10*ones(1, dim);
n_emp = 100;
n_onl = 100;
maxIter = 50;
hive = [];
cycle = 100;
opts = struct('nFig', 1, 'showFig', [true, true, true], 'v', true);


%% Run optimization
[opt, hive, time] = ABC(dim, f, lb, ub, ...
                    n_emp, n_onl, maxIter, hive, ...
                    cycle, opts);