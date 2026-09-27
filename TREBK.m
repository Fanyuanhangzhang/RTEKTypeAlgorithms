function [X,nerr,k,time_var] = TREBK(A,B,Xbar,maxit,tol,row_block_size,col_block_size,Xbarnorm)
%TREBK Tensor randomized extended block Kaczmarz
% Tensor randomized extended Kaczmarz methods for large inconsistent tensor linear equations with t-product
% Guang-Xin Huang1; Shuang-You Zhong Numerical algorithms
%Peng Zhang 20260826
tic;
[n1, n2, ~] = size(A);
kdim = size(B, 2);
n3 = size(A, 3);
row_blocks = make_contiguous_blocks(n1, row_block_size);
col_blocks = make_contiguous_blocks(n2, col_block_size);
[prow, ~] = block_probabilities(A, row_blocks, 'row');
[pcol, ~] = block_probabilities(A, col_blocks, 'column');
%row_pinv = cell(numel(row_blocks), 1);
%col_projector = cell(numel(col_blocks), 1);

  %for i = 1:numel(row_blocks)
        %row_pinv{i} = tpinv(A(row_blocks{i}, :, :));
  %end

  %for j = 1:numel(col_blocks)
        %C = A(:, col_blocks{j}, :);
        %col_projector{j} = tprod(C, tpinv(C));
  %end
X = zeros(n2, kdim, n3);
Z = B;
%Xbarnorm = norm(Xbar(:))^2;
   for k = 1:maxit
       j = sample_discrete(pcol);
       J = col_blocks{j};
       C = A(:, J, :);
       Z = Z - tprod(C, tprod(tpinv(C), Z));

       i = sample_discrete(prow);
       I = row_blocks{i};
       R = tprod(A(I, :, :), X) - B(I, :, :) + Z(I, :, :);
       X = X - tprod(tpinv(A(I, :, :)), R);

       nerr(k) = norm(X(:)-Xbar(:))^2/Xbarnorm;
       if nerr(k)<tol
             disp('Stopping criteria reached -- TREBK');
             break;
       end

   end
time_var=toc;
disp(k);
disp(time_var);
end

