clc
close all
clear all

mkdir("converted");
mkdir("converted/wav");
for i=0:1:9
	mkdir(sprintf("converted/wav/%i",i));
	audio_list = glob(sprintf("wav/%i/*.wav",i));
	for i=1:1:length(audio_list)
		[speech, fs] = wavread(audio_list(i){1});
	
		val_max = abs(max(speech));
		val_min = abs(min(speech));
		factor = max(val_max, val_min);
		speech /= factor;
		aux=sprintf('converted/%s',audio_list(i){1});
		wavwrite(speech,fs,aux);
	endfor
	disp('Done');
endfor
