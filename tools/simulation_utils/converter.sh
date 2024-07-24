#!/bin/bash
for i in $(seq -f "%06g" 0 1175)
do
   /home/lxh/Documents/ThirdParty/blender/blender "/media/lxh/Data/CarModels/8262593/blend/${i}.blend" --background --python convert_blend_to_obj.py "/media/lxh/Data/CarModels/8262593/obj/${i}.obj"
done
