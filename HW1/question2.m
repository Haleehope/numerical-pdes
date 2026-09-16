%% Problem 2
clear; close all; clc;

%% Part 1: compare numerical and exact derivative
N = 100;

h = 2*pi/N;
x = (0:N-1)'*h;

f = exp(sin(x));

% Exact second derivative
d2f_exact = exp(sin(x)).*(cos(x).^2 - sin(x));

% Periodic centered second difference
% because of periodicity, we can treat f[N] as if it's f[-1]
% circshift(f,1) dequeues and pushes the last element; f = [5, 1, 2, 3, 4]
% circshift(f,-1) pops and queues the first element; f = [2, 3, 4, 5, 1]
d2f_num = (circshift(f,1) - 2*f + circshift(f,-1))/h^2;


% Plot settings -- change these to adjust the font
fontName = 'Helvetica';
fontSize = 14;

% Get two colors from the parula colormap
colors = parula(256);
color1 = colors(1, :);
color2 = colors(200, :);

% make the exact vs. numerical solution plot
figure('Color', 'white');
plot(x, d2f_exact, 'LineWidth', 2, 'Color', color1);
hold on;
plot(x, d2f_num, '--', 'LineWidth', 2, 'Color', color2);

xlabel('x', 'FontName', fontName, 'FontSize', fontSize, 'Color', 'black');
ylabel('f ''''(x)', 'FontName', fontName, 'FontSize', fontSize, 'Color', 'black');

% give padding to the y-axis
ymin = min([d2f_exact; d2f_num]);
ymax = max([d2f_exact; d2f_num]);
padding = 0.05 * (ymax - ymin);
ylim([ymin - padding, ymax + padding]);

legend('Exact', 'Finite difference', ...
    'Location', 'best', ...
    'FontName', fontName, ...
    'FontSize', fontSize, ...
    'TextColor', 'black', ...
    'Color', 'white');
grid on;
set(gca, 'FontName', fontName, 'FontSize', fontSize, 'Color', 'white', 'XColor', 'black', 'YColor', 'black');


%% Problem 2 convergence

Ns = [20 40 80 100 120 140 160];

err_inf = zeros(size(Ns));
err_2 = zeros(size(Ns));

for k = 1:length(Ns)

    N = Ns(k);

    h = 2*pi/N;
    x = (0:N-1)'*h;

    f = exp(sin(x));

    d2f_exact = exp(sin(x)).*(cos(x).^2 - sin(x));

    d2f_num = (circshift(f,1) - 2*f + circshift(f,-1))/h^2;

    e = d2f_exact - d2f_num;

    err_inf(k) = norm(e,inf)/norm(d2f_exact,inf);
    err_2(k) = norm(e,2)/norm(d2f_exact,2);
end

p_inf = polyfit(log(Ns),log(err_inf),1);
p_2 = polyfit(log(Ns),log(err_2),1);

fprintf('Infinity slope = %.3f\n',p_inf(1));
fprintf('2-norm slope = %.3f\n',p_2(1));

figure('Color', 'white');
loglog(Ns,err_inf,'o-', 'Color', color1, 'LineWidth', 2);
hold on;
loglog(Ns,err_2,'s-', 'Color', color2, 'LineWidth', 2);

xlabel('Grid size N', 'FontName', fontName, 'FontSize', fontSize, 'Color', 'black');
ylabel('Relative error', 'FontName', fontName, 'FontSize', fontSize, 'Color', 'black');

legend('Infinity norm', '2-norm', ...
    'Location', 'best', ...
    'FontName', fontName, ...
    'FontSize', fontSize, ...
    'TextColor', 'black', ...
    'Color', 'white');
grid on;
set(gca, 'FontName', fontName, 'FontSize', fontSize, 'Color', 'white', 'XColor', 'black', 'YColor', 'black');
