set -e
make -j4
./main "../img/pin.png" "../img/pin_1.png" "../img/pin_2.png"
mv *.png ../../images_gpu/out_imgs
cd ../../images_gpu/
git add *
git commit -am "foo1"
git push origin main
