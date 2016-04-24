#!/bin/bash
#BASH script to collect all of the subset into one file


for i in $(find /mnt/snap/data -name "all.txt" -type f); do
      cat $i >> entire.txt
      echo "poop" >> entire.txt
done

