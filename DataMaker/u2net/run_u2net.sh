PWD=$(pwd)
SOURCEDIR=${1}
rm -rf ./tmp/test_images
mkdir -p ./tmp/test_images
cp  ${SOURCEDIR}/* ./tmp/test_images
echo ${SOURCEDIR}
docker run -it -v ${PWD}/tmp:/home/U-2-Net/test_data --name GetTest bigwj/u2net911:test python u2net_test.py
docker rm GetTest