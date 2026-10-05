% I'm trying to learn MATLAB instead of just using Python, 
% so I asked ChatGPT to generate the code for this assignment.

%% Problem 1
clear; close all; clc;

%% Part 1: compare numerical and exact derivative

N = 100;
% the ' is transpose
x = linspace(0, 2*pi, N)';
h = x(2) - x(1);

f = exp(sin(x));

% Exact derivative
df_exact = exp(sin(x)).*cos(x);

% Numerical derivative
df_num = zeros(N,1);

% Centered differences at interior points
df_num(2:N-1) = (f(3:N) - f(1:N-2))/(2*h);

% First-order one-sided differences at endpoints
df_num(1) = (f(2) - f(1))/h;
df_num(N) = (f(N) - f(N-1))/h;

% Plot settings -- change these to adjust the font
fontName = 'Helvetica';
fontSize = 14;

% Get two colors from the parula colormap
colors = parula(256);
color1 = colors(1, :);
color2 = colors(200, :);

% create the graph of the exact analytic solution and the numerical solution
figure('Color', 'white');
plot(x, df_exact, 'Color', color1, 'LineWidth', 2);
hold on;
plot(x, df_num, '--', 'Color', color2, 'LineWidth', 2);
xlabel('x', 'FontName', fontName, 'FontSize', fontSize, 'Color', 'black');
ylabel('f ''(x)', 'FontName', fontName, 'FontSize', fontSize, 'Color', 'black');
legend('Exact', 'Finite difference', ...
    'Location', 'best', ...
    'FontName', fontName, ...
    'FontSize', fontSize, ...
    'TextColor', 'black', ...
    'Color', 'white');
grid on;
set(gca, 'FontName', fontName, 'FontSize', fontSize, 'Color', 'white', 'XColor', 'black', 'YColor', 'black');


%% Problem 1 convergence

% increase the mesh granularity
Ns = [20 40 80 100 120 140 160];

% make zero vectors to store the errors
err_inf = zeros(size(Ns));
err_2 = zeros(size(Ns));

% for each mesh size
for k = 1:length(Ns)
    N = Ns(k);
    x = linspace(0, 2*pi, N)';
    h = x(2) - x(1);

    % exact solution
    df_exact = exp(sin(x)).*cos(x);

    % approximation
    f = exp(sin(x));
    df_num = zeros(N,1);

    % Interior
    df_num(2:N-1) = (f(3:N) - f(1:N-2))/(2*h);

    % Endpoints
    df_num(1) = (f(2) - f(1))/h;
    df_num(N) = (f(N) - f(N-1))/h;

    % calculate the errors
    e = df_exact - df_num;
    err_inf(k) = norm(e,inf)/norm(df_exact,inf);
    err_2(k) = norm(e,2)/norm(df_exact,2);
end

% Estimate convergence slopes
p_inf = polyfit(log(Ns), log(err_inf), 1);
p_2 = polyfit(log(Ns), log(err_2), 1);

fprintf('Infinity norm slope = %.3f\n', p_inf(1));
fprintf('2-norm slope = %.3f\n', p_2(1));

figure('Color', 'white');
loglog(Ns, err_inf, 'o-', 'Color', color1, 'LineWidth', 2);
hold on;
loglog(Ns, err_2, 's-', 'Color', color2, 'LineWidth', 2);

xlabel('Grid size N', 'FontName', fontName, 'FontSize', fontSize, 'Color', 'black');
ylabel('Relative error', 'FontName', fontName, 'FontSize', fontSize, 'Color', 'black');
legend('Infinity norm', '2-norm', 'Location', 'best', 'FontName', fontName, 'FontSize', fontSize, 'TextColor', 'black', 'Color', 'white');
grid on;
set(gca, 'FontName', fontName, 'FontSize', fontSize, 'Color', 'white', 'XColor', 'black', 'YColor', 'black');
