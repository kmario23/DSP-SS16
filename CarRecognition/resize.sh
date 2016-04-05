for f in `find . -name "*.jpg"`
do
    convert $f -resize 96x96\!  $f
done
