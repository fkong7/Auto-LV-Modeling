#!/bin/bash
mydir=$PWD
echo $mydir
folder=$mydir/meshes_MACS40282_20150504_pm/surfaces
code_dir='/Users/fanweikong/ALE-FEniCS/tools'


array=()
while IFS=  read -r -d $'\0'; do
    array+=("$REPLY")
done < <(find $folder -type f -name "*.vtk" -print0)

for i in "${array[@]}"
do :
    echo "Converting $i"
    python $code_dir/vtk_to_fenics.py -i $i -f hdf5 --num-boundary-faces 3 --topo-dim 2
done

