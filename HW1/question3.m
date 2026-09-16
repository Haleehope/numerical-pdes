%% Question 3
% u'' + sin(x)u' + u = f(x)
% u(0) = 0, u(2*pi) = 0
% Exact solution: u(x) = sin(x)

clear; clc; close all;

Ns = [20 40 80 100 120 140 160];

err_inf = zeros(size(Ns));
err_2   = zeros(size(Ns));

for m = 1:length(Ns)

    N = Ns(m);

    % Make grid
    a = 0;
    b = 5;

    x = linspace(a,b,N)';
    h = (b-a)/(N-1);

    % Exact solution
    u_exact = sin(x);

    % Right-hand side
    f = sin(x).*cos(x);

    % Interior grid points
    xi = x(2:N-1);
    n = length(xi);

    % Boundary conditions
    uL = 0;
    uR = sin(5);

    % ---------------------------------------------
    % Build finite difference matrices
    % ---------------------------------------------

    e = ones(n,1);

    % Second derivative:
    % u''_i = (u_{i-1} - 2u_i + u_{i+1}) / h^2
    Dxx = spdiags([e -2*e e], [-1 0 1], n, n) / h^2;

    % First derivative:
    % u'_i = (u_{i+1} - u_{i-1}) / (2h)
    Dx = spdiags([-e zeros(n,1) e], [-1 0 1], n, n) / (2*h);

    % sin(x_i) coefficient
    S = spdiags(sin(xi),0,n,n);

    % Identity matrix for the +u term
    I = speye(n);

    % u'' + sin(x)u' + u
    A = Dxx + S*Dx + I;

    % Interior right-hand side
    rhs = f(2:N-1);

    % ---------------------------------------------
    % Implement boundary conditions
    % ---------------------------------------------

    % Left boundary contribution
    rhs(1) = rhs(1) ...
        - (1/h^2 - sin(xi(1))/(2*h))*uL;

    % Right boundary contribution
    rhs(end) = rhs(end) ...
        - (1/h^2 + sin(xi(end))/(2*h))*uR;

    % Solve linear system
    ui = A \ rhs;

    % Add boundary values back to solution
    u_num = [uL; ui; uR];

    % ---------------------------------------------
    % Errors
    % ---------------------------------------------
    err = u_num - u_exact;

    err_inf(m) = norm(err,inf) / norm(u_exact,inf);
    err_2(m)   = norm(err,2) / norm(u_exact,2);

end


%% Plot solution on finest grid

scriptFolder = fileparts(mfilename('fullpath'));

fontName = 'Helvetica';
fontSize = 14;

% Get two colors from the parula colormap
colors = parula(256);
color1 = colors(1, :);
color2 = colors(200, :);


%% Plot solution on finest grid

figure('Color','white');

plot(x, u_exact, 'LineWidth', 2, 'Color', color1);

hold on;

plot(x, u_num, '--', 'LineWidth', 2, 'Color', color2);

xlabel('x', 'FontName', fontName, 'FontSize', fontSize, 'Color', 'black');

ylabel('u(x)', 'FontName', fontName, 'FontSize', fontSize, 'Color', 'black');

% Give padding to the y-axis
ymin = min([u_exact; u_num]);
ymax = max([u_exact; u_num]);

padding = 0.05 * (ymax - ymin);
ylim([ymin - padding, ymax + padding]);

legend('Exact', 'Finite Difference', 'Location', 'best', 'FontName', fontName, ...
    'FontSize', fontSize, 'TextColor', 'black', 'Color', 'white');

grid on;

set(gca, 'FontName', fontName, 'FontSize', fontSize, 'Color', 'white', ...
    'XColor', 'black', 'YColor', 'black');

exportgraphics(gcf, fullfile(scriptFolder, 'Q3a.pdf'), ...
    'ContentType', 'vector', ...
    'BackgroundColor', 'white');

%% Convergence plot

figure('Color','white');

loglog(Ns, err_inf, 'o-', 'LineWidth', 2, 'Color', color1);

hold on;

loglog(Ns, err_2, 's-', 'LineWidth', 2, 'Color', color2);

xlabel('Grid size N', 'FontName', fontName, 'FontSize', fontSize, 'Color', 'black');

ylabel('Error', 'FontName', fontName, 'FontSize', fontSize, 'Color', 'black');

legend('Infinity Norm', 'L2 Norm', 'Location', 'southwest', 'FontName', fontName, ...
    'FontSize', fontSize, 'TextColor', 'black', 'Color', 'white');

grid on;

set(gca, 'FontName', fontName, 'FontSize', fontSize, 'Color', 'white', ...
    'XColor', 'black', 'YColor', 'black');

exportgraphics(gcf, fullfile(scriptFolder, 'Q3b.pdf'), ...
    'ContentType', 'vector', 'BackgroundColor', 'white');


%% Calculate observed convergence rates

p_inf = polyfit(log(Ns), log(err_inf), 1);
p_2   = polyfit(log(Ns), log(err_2), 1);

fprintf('Question 3 infinity-norm convergence rate: %.4f\n', -p_inf(1));
fprintf('Question 3 L2 convergence rate: %.4f\n', -p_2(1));

