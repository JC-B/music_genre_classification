#!/bin/bash
#BASH script to collect all of the subset into one file

echo "Aggrigation script starting..."

touch $1/all.txt

DIR=/home/ubuntu/music_genre_classification

#append attribute names to tope of file
python $DIR/MSongsDB/PythonSrc/display_song.py /mnt/snap/data/A/C/G/TRACGAM128F92FBA44.h5 | cut -d":" -f1 | tr '\n' '|' > $1/all.txt 
echo "" >> $1/all.txt
echo "" >> $1/all.txt

echo "Starting Traveral"

for i in $(find $1 -name "*.h5" -type f); do
      python $DIR/MSongsDB/PythonSrc/display_song.py $i | cut -d":" -f2 | tr '\n' '|' >> $1/all.txt
      echo "" >> $1/all.txt
done

