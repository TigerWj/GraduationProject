PWD=$(pwd)
mkdir -p ./tmp/test_images
cp -r ${sourceDir} ./tmp/test_images
docker run -it -v ${PWD}/tmp/test_images:/home/U-2-Net/test_data/test_images --name GetTest bigwj/u2net911:test python u2net_test.py
docker rm GetTest