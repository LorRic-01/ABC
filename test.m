%% Initialization
dim = 2;
f = @(x) sum(x.^2);
lb = -100*ones(1, dim);
ub = 100*ones(1, dim);
n_emp = 100;
n_onl = 100;
maxIter = 50;
hive = [];
cycle = 100;


%% Run optimization
[opt, hive, time] = ABC(dim, f, lb, ub, ...
                    n_emp, n_onl, maxIter, hive, ...
                    cycle);