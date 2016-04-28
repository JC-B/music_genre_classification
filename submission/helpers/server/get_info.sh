DIR=/mnt/snap/data                                       

SIZE_TOTAL=0
LINES_TOTAL=0

for i in $(ls $DIR); do
        cd $DIR/$i
        THIS_DIR=`pwd`
	LINESS=$(wc -l $DIR/$i/all.txt |  awk '{print $1}')
	SIZE=$(ls -lh $DIR/$i/all.txt |  awk '{print $5}')
	echo FILE: $THIS_DIR/all.txt   SIZE: $SIZE   LINES: $LINESS
	SIZE=$(ls -l $DIR/$i/all.txt |  awk '{print $5}')
        SIZE_TOTAL=$(($SIZE+$SIZE_TOTAL))
        LINES_TOTAL=$(($LINESS+$LINES_TOTAL))
done
echo TOTAL SIZE: $SIZE_TOTAL
echo TOTAL LINE: $LINES_TOTAL

