function [X,nerr,k,time_var] = TREABK(A,B,Xbar,maxit,tol,row_block_size,col_block_size,alpha,Xbarnorm)
%TREABK Randomized extended average block Kaczmarz method for inconsistent tensor equations under t-product
%Liyuan An; Kun Liang; Han Jiao; Qilong Liu
%Numerical algorithm
%Peng Zhang 20260826
tic;
[n1, n2, ~] = size(A);
kdim = size(B, 2);
n3 = size(A, 3);
row_blocks = make_contiguous_blocks(n1, row_block_size);
col_blocks = make_contiguous_blocks(n2, col_block_size);
[prow, row_norms2] = block_probabilities(A, row_blocks, 'row');
[pcol, col_norms2] = block_probabilities(A, col_blocks, 'column');
X = zeros(n2, kdim, n3);
Z = B;
%Xbarnorm = norm(Xbar(:))^2;
%t0 = tic;
   for k = 1:maxit
         j = sample_discrete(pcol);
         J = col_blocks{j};
         C = A(:, J, :);
         Z = Z - ((alpha/n3) / col_norms2(j)) * tprod(C, tprod(tran(C), Z));

         i = sample_discrete(prow);
         I = row_blocks{i};
         R = tprod(A(I, :, :), X) - B(I, :, :) + Z(I, :, :);
         X = X - ((alpha/n3) / row_norms2(i)) * tprod(tran(A(I, :, :)), R);


         nerr(k) = norm(X(:)-Xbar(:))^2/Xbarnorm;
         if nerr(k)<tol
             disp('Stopping criteria reached -- TREABK');
             break;
         end

   end
time_var=toc;
disp(k);
disp(time_var);
end

