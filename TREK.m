function [X,nerr,k,time_var] = TREK(A,B,Xbar,maxit,tol,Xbarnorm)
%TREBK Tensor randomized extended Kaczmarz
% Tensor randomized extended Kaczmarz methods for large inconsistent tensor linear equations with t-product
% Guang-Xin Huang1; Shuang-You Zhong Numerical algorithms
%Peng Zhang 20260826
tic;
[n1, n2, ~] = size(A);
kdim = size(B, 2);
n3 = size(A, 3);
rows = make_contiguous_blocks(n1, 1);
cols = make_contiguous_blocks(n2, 1);
[prow, ~] = block_probabilities(A, rows, 'row');
[pcol, ~] = block_probabilities(A, cols, 'column');
%Xbarnorm = norm(Xbar(:))^2;
%row_pinv = cell(n1, 1);
%for i = 1:n1
    %Aitran = tran(A(i, :, :));
    %row_projector = tpinv(tprod(A(i, :, :),Aitran));
    %row_pinv{i} = tprod(Aitran,row_projector);
%end
%col_pinv = cell(n2, 1);
%for j = 1:n2
    %Aj = A(:, j, :);
    %col_projector = tpinv(tran(Aj),Aj);
    %col_pinv{j} = tprod(Aj,col_projector);
%end

X = zeros(n2, kdim, n3);
Z = B;
%Xbarnorm = norm(Xbar(:));
   for k = 1:maxit
       j = sample_discrete(pcol);
       C = A(:, j, :);
       Ctran = tran(C);
       col_projector = tpinv(tprod(Ctran,C));
       col_pinv = tprod(C,col_projector);
       Z = Z - tprod(col_pinv, tprod(Ctran, Z));

       i = sample_discrete(prow);
       I = A(i, :, :);
       R = tprod(I, X) - B(i, :, :) + Z(i, :, :);
       Aitran = tran(I);
       row_projector = tpinv(tprod(I,Aitran));
       row_pinv = tprod(Aitran,row_projector);
       X = X - tprod(row_pinv, R);

       nerr(k) = norm(X(:)-Xbar(:))^2/Xbarnorm;
       if nerr(k)<tol
             disp('Stopping criteria reached -- TREK');
             break;
       end

   end
time_var=toc;
disp(k);
disp(time_var);
end

