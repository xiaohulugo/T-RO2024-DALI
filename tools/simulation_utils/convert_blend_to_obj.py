import bpy
import sys
import os

argv = sys.argv

bpy.ops.export_scene.obj(filepath=argv[5], axis_forward='-Z', axis_up='Y')


# for ob in bpy.data.objects:
#     if ob.name == "Cylinder 1.001": bpy.data.window_managers["WinMan"].(type) = "3D Cylinders"
#     	print (ob.name,ob.dimensions,ob.location,ob.rotation_euler.z)

# ob = bpy.data.objects["Cube"]
