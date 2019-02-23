#!/bin/bash
pushd ../ #project root directory

#echo "Copying boss images into s-uniward directory..."
#rsync -ahr --info=progress2 data/raw/boss/cover/* external/s-uniward/S-UNIWARD_linux_make_v10/images_cover/
#./S-UNIWARD -v -I ../images_cover -O ../images_stego -a 0.4

echo "Running s-uniward stego algorithm..."
#pushd external/s-uniward/S-UNIWARD_linux_make_v10/executable

mkdir data/cache data/cache/s-uniward

for i in {1..20}; do
	external/s-uniward/S-UNIWARD_linux_make_v10/executable/S-UNIWARD -v -I data/cache/boss_split/$i -O data/cache/s-uniward -a 0.4
done

echo "Done!"

popd
