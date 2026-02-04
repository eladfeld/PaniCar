Fs = 30;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = 199;             % Length of signal
%t = (0:L-1)*T;        % Time vector



fname = '4_models_signals.xlsx';


C = xlsread(fname, 'C:C');
I = xlsread(fname, 'I:I');

Y = fft(C);
Y_1 = fft(I);

P1_1 = fft_signal(Y, L);
P1_2 = fft_signal(Y_1, L);

f = Fs*(0:(L/2))/L;
plot(f, P1_1);
hold on
plot(f, P1_2);


legend('Toyota Rav4- White','Toyota Rav4- Silver')

title("Retina net - Single-Sided Amplitude Spectrum of X(t)")
xlabel("f (Hz)")
ylabel("|P1(f)|")
ylim([0 1.2])

function sig = fft_signal(y, l)
    P2_1 = abs(y/l);
    P1_1 = P2_1(1:l/2+1);
    P1_1(2:end-1) = 2*P1_1(2:end-1);
    sig = P1_1;
end