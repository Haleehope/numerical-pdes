function [x,y,U,Uexact] = problem4_dirichlet(N)

    x = linspace(0,5,N);
    y = linspace(0,5,N);

    h = x(2)-x(1);

    [X,Y] = meshgrid(x,y);

    % Exact solution
    Uexact = sin(X).*cos(Y);

    % Forcing function
    F = -2*sin(X).*cos(Y);

    % Start numerical solution with exact boundary conditions
    U = zeros(N,N);

    U(:,1) = Uexact(:,1);
    U(:,N) = Uexact(:,N);
    U(1,:) = Uexact(1,:);
    U(N,:) = Uexact(N,:);

    % Number of interior points in each dimension
    n = N-2;

    % Total number of unknowns
    M = n*n;

    A = sparse(M,M);
    b = zeros(M,1);

    % Map interior point (i,j) to vector location k
    index = @(i,j) (j-1)*n + i;

    for j = 1:n
        for i = 1:n

            % Full-grid indices
            ii = i+1;
            jj = j+1;

            k = index(i,j);

            % PDE forcing
            b(k) = F(jj,ii);

            % Center
            A(k,k) = -4/h^2;

            %% Left neighbor
            if i > 1

                A(k,index(i-1,j)) = 1/h^2;

            else

                b(k) = b(k) - U(jj,1)/h^2;

            end

            %% Right neighbor
            if i < n

                A(k,index(i+1,j)) = 1/h^2;

            else

                b(k) = b(k) - U(jj,N)/h^2;

            end

            %% Bottom neighbor
            if j > 1

                A(k,index(i,j-1)) = 1/h^2;

            else

                b(k) = b(k) - U(1,ii)/h^2;

            end

            %% Top neighbor
            if j < n

                A(k,index(i,j+1)) = 1/h^2;

            else

                b(k) = b(k) - U(N,ii)/h^2;

            end
        end
    end

    % Solve
    uvec = A\b;

    % Put values back into the 2-D array
    for j = 1:n
        for i = 1:n

            k = index(i,j);

            U(j+1,i+1) = uvec(k);

        end
    end

end


%% Problem 4 solution

clear; close all; clc;

scriptFolder = fileparts(mfilename('fullpath'));

N = 51;

[x,y,U,Uexact] = problem4_dirichlet(N);

[X,Y] = meshgrid(x,y);

% Plot settings
fontName = 'Helvetica';
fontSize = 14;

% Parula colors
colors = parula(256);
color1 = colors(1, :);
color2 = colors(200, :);

% Numerical solution
figure('Color', 'white');
surf(X,Y,U);
shading interp;
colormap(parula);

xlabel('x', 'FontName', fontName, 'FontSize', fontSize, 'Color', 'black');
ylabel('y', 'FontName', fontName, 'FontSize', fontSize, 'Color', 'black');
zlabel('u(x,y)', 'FontName', fontName, 'FontSize', fontSize, 'Color', 'black');
set(gca, 'FontName', fontName, 'FontSize', fontSize, 'Color', 'white', ...
    'XColor', 'black', 'YColor', 'black', 'ZColor', 'black');
grid on;

% Add colorbar and make its text black
cb = colorbar;
cb.Color = 'black';
cb.FontName = fontName;
cb.FontSize = fontSize;

exportgraphics(gcf, fullfile(scriptFolder, 'Q4a.pdf'), 'ContentType', 'image', 'Resolution', 300, 'BackgroundColor', 'white');

%% Problem 4 convergence

Ns = [20 40 80 100 120 140 160];

err_inf = zeros(size(Ns));
err_2 = zeros(size(Ns));

for k = 1:length(Ns)

    [~,~,U_temp,Uexact_temp] = problem4_dirichlet(Ns(k));

    e = U_temp-Uexact_temp;

    err_inf(k) = norm(e(:),inf)/norm(Uexact_temp(:),inf);
    err_2(k) = norm(e(:),2)/norm(Uexact_temp(:),2);

end

% convergence plot
figure('Color', 'white');

loglog(Ns, err_inf, 'o-', 'LineWidth', 2, 'Color', color1);

hold on;

loglog(Ns, err_2, 's-', 'LineWidth', 2, 'Color', color2);

xlabel('Grid size N', 'FontName', fontName, 'FontSize', fontSize, 'Color', 'black');

ylabel('Relative error', 'FontName', fontName, 'FontSize', fontSize, 'Color', 'black');

legend('Infinity norm', '2-norm', 'Location', 'best', 'FontName', fontName, ...
       'FontSize', fontSize, 'TextColor', 'black', 'Color', 'white');

grid on;

set(gca, 'FontName', fontName, 'FontSize', fontSize, 'Color', 'white', 'XColor', 'black', 'YColor', 'black');

exportgraphics(gcf, fullfile(scriptFolder, 'Q4b.pdf'), 'ContentType', 'vector', 'BackgroundColor', 'white');

p_inf = polyfit(log(Ns), log(err_inf), 1);
p_2   = polyfit(log(Ns), log(err_2), 1);

fprintf('Question 4 infinity-norm convergence rate: %.4f\n', -p_inf(1));
fprintf('Question 4 L2 convergence rate: %.4f\n', -p_2(1));