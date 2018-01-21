#!/bin/sh

search_dir=wav

for dir in "$search_dir"/*
do
  echo "$dir"
  for entry in "$dir"/*
  do
  	echo "$entry"
  	sox "$entry" out.wav silence -l 1 0.01 0.01% reverse silence 1 0.01 0.01% reverse
  	mv out.wav "$entry"
  done
   
  
done
