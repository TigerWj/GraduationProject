# GraduationProject

## some image tramsformation demo

MSL rigid deformation
```
cd DataMaker
python3 transformDemo.py <imageDir> <keyPointFileDir>
```

Local area deformation
```
cd DataMaker
python3 ./warp/test_warp.py <imageDir>
```

Run u2net  
```
docker pull bigwj/u2net911:test
cd DataMaker/u2net
sh ./run_u2net.sh <imageFolderDir>
```
results would be saved in DataMaker/u2net/tmp/

MSL quote from https://github.com/Jarvis73/Moving-Least-Squares
