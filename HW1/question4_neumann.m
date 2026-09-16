function [x,y,U,Uexact,A] = problem4_neumann(N)

    x = linspace(0,5,N);
    y = linspace(0,5,N);

    h = x(2)-x(1);

    [X,Y] = meshgrid(x,y);

    % Exact solution
    Uexact = sin(X).*cos(Y);

    % Forcing function
    F = -2*sin(X).*cos(Y);

    % With Neumann conditions, boundary values are also unknown.
    % Therefore, there are N*N total unknowns.
    M = N*N;

    A = sparse(M,M);
    b = zeros(M,1);

    % Map grid point (i,j) to vector location k
    index = @(i,j) (j-1)*N + i;


    %% Interior points

    for j = 2:N-1
        for i = 2:N-1

            k = index(i,j);

            % PDE forcing
            b(k) = F(j,i);

            % Five-point stencil
            A(k,k) = -4/h^2;

            A(k,index(i-1,j)) = 1/h^2;
            A(k,index(i+1,j)) = 1/h^2;
            A(k,index(i,j-1)) = 1/h^2;
            A(k,index(i,j+1)) = 1/h^2;

        end
    end


    %% Left boundary: x = 0
    % du/dn = -u_x = -cos(y)
    %
    % -(u(2,j)-u(1,j))/h = -cos(y)

    for j = 2:N-1

        k = index(1,j);

        A(k,index(1,j)) = 1/h;
        A(k,index(2,j)) = -1/h;

        b(k) = -cos(y(j));

    end


    %% Right boundary: x = 5
    % du/dn = u_x = cos(5)cos(y)

    for j = 2:N-1

        k = index(N,j);

        A(k,index(N-1,j)) = -1/h;
        A(k,index(N,j)) = 1/h;

        b(k) = cos(5)*cos(y(j));

    end


    %% Bottom boundary: y = 0
    % du/dn = -u_y = 0

    for i = 2:N-1

        k = index(i,1);

        A(k,index(i,1)) = 1/h;
        A(k,index(i,2)) = -1/h;

        b(k) = 0;

    end


    %% Top boundary: y = 5
    % du/dn = u_y = -sin(x)sin(5)

    for i = 2:N-1

        k = index(i,N);

        A(k,index(i,N-1)) = -1/h;
        A(k,index(i,N)) = 1/h;

        b(k) = -sin(x(i))*sin(5);

    end


    %% Corners
    % Each corner belongs to two boundaries.
    % Use the x-direction Neumann condition at the corners.

    % Bottom-left corner
    k = index(1,1);

    A(k,index(1,1)) = 1/h;
    A(k,index(2,1)) = -1/h;

    b(k) = -cos(y(1));


    % Top-left corner
    k = index(1,N);

    A(k,index(1,N)) = 1/h;
    A(k,index(2,N)) = -1/h;

    b(k) = -cos(y(N));


    % Bottom-right corner
    k = index(N,1);

    A(k,index(N-1,1)) = -1/h;
    A(k,index(N,1)) = 1/h;

    b(k) = cos(5)*cos(y(1));


    % Top-right corner
    k = index(N,N);

    A(k,index(N-1,N)) = -1/h;
    A(k,index(N,N)) = 1/h;

    b(k) = cos(5)*cos(y(N));


    %% Try to solve the system

    uvec = A\b;

    % Put solution vector back into 2-D array
    U = reshape(uvec,N,N);

end

%% Problem 4 - Neumann boundary conditions

N = 50;

[x,y,U_neumann,Uexact_neumann,A_neumann] = ...
    problem4_neumann(N);

% Check whether a constant vector is in the null space
% A matrix is singular if it has a nonzero vector in its null space
% So, there is a constant null vector and the solution using Neumann conditions is not unique
constant_vector = ones(N*N,3);

null_error = norm(A_neumann*constant_vector);

fprintf('||A*3|| = %.4e\n', null_error);