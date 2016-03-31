### Script for attaching genre tags to songs
print("Starting script...")
genres = {}

genre_file = open('genres.txt','r')

print("Reading in genres...")
for line in genre_file:
    song,genre = line.split('\t')
    genres[song.strip()] = genre.strip()

genre_file.close()

combined_file = open('combined.txt','w')
song_file = open('songs.txt','r')

print("Reading in songs and combining with genres...")
for line in song_file:
    data = line.split('\t')
    for attribute in data:
        attribute.strip()
    try:
        name = data[52]
        name = name[1:]
    except:
        print("NO NAME")
        name = "o"
    try:
        data.append(genres[name])
        
        for attribute in data:
            combined_file.write(attribute.strip())
            combined_file.write('\t')
        combined_file.write('\n')
    except:
        print("NO GENRE FOR SONG {0}".format(name))

print("Finished!")
