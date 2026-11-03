function [C,W,time,savedState,F] = AGEB( X, Y, alpha, beta, gamma,lambda)
%% function discription
% ----------------------------------------------------------------------
% Auto view and Embbeded gragh
% ----------------------------------------------------------------------
% rng(6)
% Calucate some statitics about the data
rng('shuffle')
savedState = rng;
[num_train, num_feature] = size(X) ;
num_label = size(Y, 2);

c=ceil(sqrt(num_feature));%自动视图数
% lambda=1;%对rank=c的约束参数

%Initialize C,W
C = rand(num_feature, num_feature);
W = rand(num_feature, num_label);

%% 初始化F,用于求解C的c聚类问题
A0 = (C+C')/2;
D0 = diag(sum(A0));
L0 = D0 - A0;%(LZR)compute Laplacian matrix

start=tic;

%调用eig1函数，计算F
% F是L0的前c个最小特征值对应的特征向量组成的矩阵 num * c
[F, ~, ~]=eig1(L0, c, 0);%(LZR)[eigvec, eigval, eigval_full] = eig1(A, c, isMax, isSym)

N = 0.5*pdist2(F, F).^2; %||F_i - F_j||_2^2
XC = X*C;
K = pdist2(XC', Y').^2;%||XC_i - Y_j||_2^2
E = XC*W-Y;
E12=sqrt(sum(E.*E,2)+eps);
W12=sqrt(sum(W.*W,2)+eps);

val=[];
iter = 1; obji = 1;
while 1

%Update W--------------------------------------------------------------
    d = 0.5./W12;
    Dw = diag(d);

    d1 = 0.5./E12;
    D1 = diag(d1);

    XCTD1=XC'*D1;
    W1 = 2*XCTD1*Y;
    W2 = 2*XCTD1*XC*W + alpha*K + 2*beta*Dw*W;
    W=W.*(W1./(W2+eps));

%Update C--------------------------------------------------------------
    d1 = 0.5./E12;
    D1 = diag(d1);

    D = diag(sum(W,2));% W的行和对角阵
    XTD1=X'*D1;
    XTXC=X'*XC;
    C1 = 2*(XTD1*Y*W' + alpha*X'*Y*W' + gamma*X'*X);
    C2 = 2*(XTD1*XC*W*W' + alpha*XTXC*D + gamma*XTXC) + lambda*N;
    C=C.*(C1./(C2+eps));

%Update F--------------------------------------------------------------
    % (LZR)update F, which is form by the c eigenvectors of L corresponding
    % to the c smallest eigenvalues
    % 计算拉普拉斯矩阵L
    A1 = (C+C')/2;
    D1 = diag(sum(A1));
    L1 = D1-A1;
    
    % F是L的前c个最小特征值对应的特征向量组成的矩阵 num * c
    [F, ~, ~]=eig1(L1, c, 0);%(LZR)[eigvec, eigval, eigval_full] = eig1(A, c, isMax, isSym)


    N = 0.5*pdist2(F, F).^2; %||F_i - F_j||_2^2
    XC=X*C;
    K = pdist2(XC', Y').^2;%||XC_i - Y_j||_2^2
    E=XC*W-Y;
    E12=sqrt(sum(E.*E,2)+eps);
    W12=sqrt(sum(W.*W,2)+eps);
    obj(iter) = sum(E12) + alpha*sum(sum(K .* W)) + beta*sum(W12) + gamma*(norm((XC - X), 'fro'))^2 + lambda*sum(sum(N .* C));
    val=[val obj(iter)];
    cver = abs((obj(iter) - obji)/obji);
    if (obj(iter) - obji>0 && iter > 2)
        disp("error")
    end
    obji = obj(iter);
    iter = iter + 1;
    if (cver < 10^-3 && iter > 2 || iter > 50) , break, end
    % if (iter > 50) , break, end

end
time = toc(start);

end

