clc, clearvars

% Benchmark functions
% f = @(x) -20*exp(-0.2*sqrt(0.5*(x(1).^2 + x(2).^2))) - exp(0.5*(cos(2*pi*x(1)) + cos(2*pi*x(2)))) + exp(1) + 20;
% f = @(x) sum(x.^4 - 16*x.^2 + 5*x, 2)/2;
% f = @(x) 100*sum(sqrt(abs(exp((x-1).^2) - 1)));
% f = @(x) sum(abs(exp((x-1).^2./(exp(x)+1))-1));
% f = @(x) sum(0.01*rand(size(x)).*(x-1).^2);
% g = []; h = [];

f = @(x) (1 - x(1))^2 + 100*(x(2) - x(1)^2)^2;
h = @(x) [(x(1) - 1)^3 - x(2) + 1, x(1) + x(2) - 2];

% f = @(x) 4*x(1)^2 - 2.1*x(1)^4 + 1/3*x(1)^6 + x(1)*x(2) - 4*x(2)^2 + 4*x(2)^4;
% h = @(x) -sin(4*pi*x(1)) + 2*sin(2*pi*x(2))^2 - 1.5;

% f = @(x) 0.1*x(1)*x(2);
% h = @(x) x(1)^2 + x(2)^2 - (1 + 0.2*cos(8*atan2(x(2), x(1))))^2;
g = [];

%% Initialization
dim = 2;
% f = [];
lb = -10*ones(1, dim);
ub = 10*ones(1, dim);
% g = [];
% h = [];
n_emp = 100;
n_onl = 100;
maxIter = inf;
hive = [];
cycle = 100;
opts = struct('nFig', 1, 'showFig', [true, false, true], 'v', true);


%% Run optimization
[opt, hive, time] = CABC(dim, f, lb, ub, g, h,...
                    n_emp, n_onl, maxIter, hive, ...
                    cycle, opts);

fprintf('Solution: [%f, ', opt(1)); if dim > 1, fprintf('%f, ', opt(2: end)); end, fprintf('\b\b]\n');
fprintf('Sol. cost: %f\n', f(opt));
