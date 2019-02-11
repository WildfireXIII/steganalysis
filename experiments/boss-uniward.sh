#!/bin/bash
pushd ../

echo "Copying boss images into s-uniward directory..."
rsync -ahr --info=progress2 data/raw/boss/cover/* external/s-uniward/S-UNIWARD_linux_make_v10/images_cover/

echo "Running s-uniward stego algorithm..."
pushd external/s-uniward/S-UNIWARD_linux_make_v10/executable

bash example_default.sh

echo "Done!"

popd
