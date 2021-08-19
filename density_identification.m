tic
load(['AYM\textile.mat'])
A = cast(A,'double');
Y = Y';
ATA = A' * A;
rA = rank(A);
rT = rank(ATA);
[U,S,V] = svd(ATA);  % A'A = U*S*V'
rS = rank(S);
row = all(S<10e-5, 2);
col = all(S<10e-5, 1);
S(row,:) = []; 
S(:,col) = [];
U(:,row) = [];
V(:,col) = [];

rU = rank(U);
rV = rank(V);
sV = size(V);
H = 2 * S;
f = -(Y'*A*V)';
Z = quadprog(H,f,-V,zeros(sV(1),1));  
% x = quadprog(H,f,A,b) (note: not the same 'A'):
% 在 A*x ≤ b 的条件下求 x'*H*x/2 + f'*x 的最小值
X = V * Z;

% figure(1)
% plot(X)

u = zeros(m, m);
cnt=1; 
for i=1:m
    for j=1:m+1-i
        u(i,j)=X(cnt);
        cnt=cnt+1;
    end
end
figure(99)
heatmap(log(1e4*u+1))
delete('density.csv') 
writematrix(u, 'density.csv');
toc
    
