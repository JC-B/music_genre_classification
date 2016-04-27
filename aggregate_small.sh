#!/bin/bash
#BASH script to collect all of the subset into one file

echo "Aggrigation script starting..."

#append attribute names to tope of file
python MSongsDB/PythonSrc/display_song.py /mnt/snap/data/A/C/G/TRACGAM128F92FBA44.h5 | cut -d":" -f1 | tr '\n' '|' > /mnt/snap/all.txt 
echo "" >> /mnt/snap/all.txt
echo "" >> /mnt/snap/all.txt

PERCENT=0
LINES=0
TOTAL=10000000

echo "Starting Traveral"

for i in $(find /mnt/snap/data -name "*.h5" -type f); do
      python MSongsDB/PythonSrc/display_song.py $i | cut -d":" -f2 | tr '\n' '|' >> /mnt/snap/all.txt
      echo "" >> /mnt/snap/all.txt
      LINES=`cat test.txt | wc -l`
      #echo $LINES
      NEW_PERCENT=`awk -v lines="$LINES" -v total="$TOTAL" 'BEGIN { rounded = sprintf("%.0f", lines/total*100); print rounded }'`
      #echo $NEW_PERCENT
      if [ "$PERCENT" -ne "$NEW_PERCENT" ]
      then
         PERCENT=$NEW_PERCENT
         echo "$PERCENT%"
      fi
done

