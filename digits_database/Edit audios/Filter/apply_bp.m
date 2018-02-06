function signal = apply_bp(signal, fs, fc1, fc2, delta_fc)
	
	wp1 = pi*(fc1+delta_fc)/fs;
	ws1 = pi*(fc1-delta_fc)/fs;
	
	wp2 = pi*(fc2-delta_fc)/fs;
	ws2 = pi*(fc2+delta_fc)/fs;
	
	Deltaw = min(wp1-ws1, ws2-wp2);
	omegac1 = (ws1+wp1)/2;
	omegac2 = (ws2+wp2)/2;

	hlp = @(n,w0,N) sinc(w0/pi*(n-N/2))*w0/pi;
	hhp=@(n,w0,N) sinc(n-N./2) - w0./pi .* sinc( w0.*(n-N./2)./ pi );

	M = round((8*pi)/Deltaw); % orden del filtro

	% La ventana queda:
	w = hamming(M+1);
	n = 0:M;

	lp = hlp(n,omegac2,M); % filtro ideal (truncado a 0:M)
	hp = hhp(n,omegac1,M);
	lp = w.'.*(lp); % filtro obtenido
	hp = w.'.*(hp);

	signal = filter(hp,1,filter(lp,1,signal));

endfunction

