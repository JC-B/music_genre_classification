### Script for attaching genre tags to songs

genre_file = open('GenreTags.txt','r')

count = 0

for line in genre_file.readline():
    if line[0] == 'T':
        count += 1

print(count)
