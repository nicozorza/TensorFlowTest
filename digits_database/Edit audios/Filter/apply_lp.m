function signal = apply_lp(signal, fs, fc, delta_fc)
	
	wp = pi*(fc-delta_fc)/fs;
	ws = pi*(fc+delta_fc)/fs;

	Deltaw = ws-wp;
	omegac = (ws+wp)/2;

	hlp = @(n,w0,N) sinc(w0/pi*(n-N/2))*w0/pi;

	M = round((8*pi)/Deltaw); % orden del filtro

	% La ventana queda:
	w = hamming(M+1);
	n = 0:M;

	lp = hlp(n,omegac,M); % filtro ideal (truncado a 0:M)
	h = w.'.*lp; % filtro obtenido

	
	signal = filter(h,1,signal);
endfunction

