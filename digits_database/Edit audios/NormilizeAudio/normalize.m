clc
close all
clear all


audio_list = glob("9/*.wav");

for i=1:1:length(audio_list)

	[speech, fs] = wavread( audio_list(i){1} );
	
	val_max = abs(max(speech));
	val_min = abs(min(speech));
	factor = max(val_max, val_min);
	speech /= factor;
	aux=sprintf('converted/%s',audio_list(i){1});
	wavwrite(speech,fs,aux);

endfor

