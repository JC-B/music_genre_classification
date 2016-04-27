#!/usr/bin/env python

import subprocess
import csv
import sys
COLUMNSTOREMOVE = [
    "artist_7digitalid",
    "artist_id",
    "artist_location",
    "artist_mbtags",
    "artist_mbtags_count",
    "artist_mbid",
    "artist_name",
    "artist_playmeid",
    "audio_md5",
    "release",
    "release_7digitalid",
    "similar_artists",
    "song_id",
    "title",
    "track_7digitalid",
    "track_id",
    "unknown",
    "DONE, showed song 0 / 0 in file"
]

with open(sys.argv[1]) as infile, open(sys.argv[2], "wb") as outfile:
    r = csv.DictReader(infile, delimiter="|")
    firstrow = next(r)  # Need to read the first row so we know the fieldnames
    fields = r.fieldnames
    w = csv.DictWriter(outfile, 
                       [field for field in fields if not field in COLUMNSTOREMOVE], 
                       extrasaction="ignore",delimiter="|")
    w.writeheader()
    w.writerow(firstrow)
    for row in r:
        w.writerow(row)

subprocess.call(['sed','-i','/None$/d',sys.argv[2]])
