for f in `find . -name "*.jpg"`
do
    convert $f -resize 256x256^ -gravity Center -crop 224x224+0+0  $f
done
