DIR=/mnt/snap/data

COUNT=0

for i in $(ls $DIR); do
    cd $DIR/$i
    pwd
    bash /home/ubuntu/music_genre_classification/p_aggregate.sh $DIR/$i &
    COUNT=$(($COUNT+1))
done
                                
echo $COUNT
