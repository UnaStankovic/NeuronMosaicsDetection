#!/bin/bash

convertsecs() {
	((h=${1}/3600))
	((m=(${1}%3600)/60))
	((s=${1}%60))
 	printf "%02d:%02d:%02d" $h $m $s
}

rm -r $1 2> /dev/null;
mkdir $1 2> /dev/null;

i=1;
start=0;
end=0;
total=0;
while [ $i -lt $# ]; do
	clear;
	
	if [ $total -gt 0 ]; then
		echo "[$((i-1))] Finished in: $(convertsecs interval).";
		echo "Time elapsed: $(convertsecs total).";
		echo "Time remaining: $(convertsecs `echo "scale=0; $(($#-i))*${total}/$((i-1))" | bc -l`)";
		echo;
	fi

	i=$((i+1));
	echo "Running feature extraction for ${!i}... [$((i-1))/$(($#-1)) | "$(echo "scale=2; $((i-1))*100/$(($#-1))" | bc -l)"%]"
	start=$(date +'%s');
	python extract_features.py ${!i} $1/data_$((i-1)).csv > /dev/null;
	end=$(date +'%s');

	interval=$((end-start));
	total=$((total+interval));
done

clear;

echo "[$((i-1))] Finished in: $(convertsecs interval).";
echo "Time elapsed: $(convertsecs total).";
echo;

echo 'Done.';
