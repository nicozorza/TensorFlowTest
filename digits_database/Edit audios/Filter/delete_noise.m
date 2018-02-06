clc
clear all
close all

fc1=1e3;
fc2=3e3;
delta_fc=100;
delta=0.001;

mkdir("converted");
mkdir("converted/wav");
for i=0:1:9
	mkdir(sprintf("converted/wav/%i",i));
	audio_list = glob(sprintf("wav/%i/*.wav",i));
	for i=1:1:length(audio_list)
		[speech, fs] = wavread(audio_list(i){1});
	
		speech = apply_bp(speech, fs, fc1, fc2, delta_fc, delta);
		
		aux=sprintf('converted/%s',audio_list(i){1});
		wavwrite(speech,fs,aux);
	endfor
	disp('Done');
endfor
