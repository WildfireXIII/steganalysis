#!/bin/bash
pushd ../

#echo "Copying boss images into wow directory..."
#rsync -ahr --info=progress2 data/raw/boss/cover/* external/wow/WOW_linux_make_v10/images_cover

echo "Running wow stego algorithm..."
#pushd external/wow/WOW_linux_make_v10/executable

mkdir data/cache data/cache/wow

external/wow/WOW_linux_make_v10/executable/WOW -v -I data/raw/boss/cover/ -O data/cache/wow -a 0.4

echo "Done!"

popd
