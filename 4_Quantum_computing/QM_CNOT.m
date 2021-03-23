clear all;

Nvec  =    16;
Nstep = 10000;
T     =     8;

x    = zeros(Nvec, Nstep);
y    = zeros(Nvec, Nstep);
prob = zeros(Nvec, Nstep);

CNOT  = [[1 0 0 0]
         [0 1 0 0]
         [0 0 0 1]
         [0 0 1 0]];
     
zero2 = [0 0]';
v0    = [1 0]';
v1    = [0 1]';

I4    = [[1 0 0 0]
         [0 1 0 0]
         [0 0 1 0]
         [0 0 0 1]];
     
I2    = [[1 0]
         [0 1]];

CNOT4 = kron(I4, CNOT);


v00   = [v0;    zero2];
v01   = [v1;    zero2];
v10   = [zero2; v0];
v11   = [zero2; v1];
zero4 = [zero2; zero2];

v000  = [v00;   zero4];
v001  = [v01;   zero4];
v010  = [v10;   zero4];
v011  = [v11;   zero4];
v100  = [zero4; v00];
v101  = [zero4; v01];
v110  = [zero4; v10];
v111  = [zero4; v11];
zero8 = [zero4; zero4];

v0000 = [v000;  zero8];
v0001 = [v001;  zero8];
v0010 = [v010;  zero8];
v0011 = [v011;  zero8];
v0100 = [v100;  zero8];
v0101 = [v101;  zero8];
v0110 = [v110;  zero8];
v0111 = [v111;  zero8];
v1000 = [zero8;  v000];
v1001 = [zero8;  v001];
v1010 = [zero8;  v010];
v1011 = [zero8;  v011];
v1100 = [zero8;  v100];
v1101 = [zero8;  v101];
v1110 = [zero8;  v110];
v1111 = [zero8;  v111];

c = [[0, 0]
     [1, 0]];

a = [[0, 1]
     [0, 0]];
 
c3 = kron(kron(kron(I2,I2),I2), c);
c2 = kron(kron(kron(I2,I2), c),I2);
c1 = kron(kron(kron(I2, c),I2),I2);
c0 = kron(kron(kron( c,I2),I2),I2);

a3 = kron(kron(kron(I2,I2),I2), a);
a2 = kron(kron(kron(I2,I2), a),I2);
a1 = kron(kron(kron(I2, a),I2),I2);
a0 = kron(kron(kron( a,I2),I2),I2);

M1 = CNOT4;

H = (c1 * a0 * M1) + (c1 * a0 * M1)';

U = expm(-i*H);

dt = T / (Nstep-1);
for i = 1:Nstep
    x(1:Nvec,i)=1:Nvec;
end
for i = 1:Nstep
    y(1:Nvec,i)=i;
end
for i = 1:Nstep
    t = (i-1) * dt;
    Ut = U^t;
    psi = Ut * v1011;
    prob(:,i) = abs(psi.*psi);
end

ticks = ['v0000';
         'v0001';
         'v0010';
         'v0011';
         'v0100';
         'v0101';
         'v0110';
         'v0111';
         'v1000';
         'v1001';
         'v1010';
         'v1011';
         'v1100';
         'v1101';
         'v1110';
         'v1111'];
     
contourf(x,y,prob);
xlabel('Vector');
ylabel('Time step');
set(gca,'XTick',1:16);
set(gca,'XTickLabel',ticks);
xticklabel_rotate;

     