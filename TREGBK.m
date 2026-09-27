function [X,nerr,k,time_var] = TREGBK(A,B,Xbar,maxit,tol,row_block_size,delta,Xbarnorm)
%TREGBK Tensor randomized extended greedy block Kaczmarz algorithm
% Tensor randomized extended Kaczmarz methods for large inconsistent tensor linear equations with t-product
% Guang-Xin Huang1; Shuang-You Zhong Numerical algorithms
%Peng Zhang 20260826
tic;
[n1, n2, n3] = size(A);
kdim = size(B,2);
row_blocks = make_contiguous_blocks(n1, row_block_size);
[prow, ~] = block_probabilities(A, row_blocks, 'row');
%Xbarnorm = norm(Xbar(:))^2;
X=zeros(n2,kdim,n3); 
Z=B;
    for k =1:maxit

       
       %score=zeros(n2,1);
       %for j=1:n2
          %Cj=A(:,j,:);
          %q=tprod(tran(Cj),Z);
          %score(j)=sum(q(:).^2);
       %end
       
       C = tprod(tran(A),Z);
       score = reshape(sum(sum(C.^2, 2), 3), n2, 1);
       threshold= delta*max(score);
       U=score>=threshold;

       C2 = A(:,U,:);
       Z = Z - tprod(C2, tprod(tpinv(C2), Z));


       i = sample_discrete(prow);
       I = row_blocks{i};
       R = tprod(A(I, :, :), X) - B(I, :, :) + Z(I, :, :);
       X = X - tprod(tpinv(A(I, :, :)), R);


       nerr(k) = norm(X(:)-Xbar(:))^2/Xbarnorm;
       if nerr(k)<tol
             disp('Stopping criteria reached -- TREGBK');
             break;
       end
    end
time_var=toc;
disp(k);
disp(time_var);
end

