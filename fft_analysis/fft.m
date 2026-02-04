data = rand(60000,1); % replace with your data
y = [];
for i = 1:120
  y(:,i) = data((i-1)*500+1:(i)*500);;
end
% perform the fft as;
L = 500;
Y = fft(y);
P2 = abs(Y/L);
P1 = P2(1:L/2+1,:);
P1(2:end-1,:) = 2*P1(2:end-1,:);
% obtain frequency matrix;
Fs = 1e3; % sampling frequency
f = Fs*(0:(L/2))/L;
% plot frequency domain
figure(1)
semilogy(f,P1)
% plot time domain
figure(2)
plot([1:L]/Fs,y)
