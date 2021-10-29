import bpy
import random
import uuid
import math
import datetime
import os

primitive_constructors = [
    bpy.ops.mesh.primitive_cube_add,
    bpy.ops.mesh.primitive_uv_sphere_add,
    bpy.ops.mesh.primitive_cylinder_add,
    bpy.ops.mesh.primitive_cone_add,
    bpy.ops.mesh.primitive_torus_add,
]

dpi = 2*math.pi

model_folder = "/tmp/models"
if not os.path.exists(model_folder):
    os.mkdir(model_folder)

for ctor in random.choices(primitive_constructors, k=10):
    ctor()
    bpy.context.active_object.select_set(True)
    transform_seed = datetime.datetime.now().microsecond % 100
    print(transform_seed)
    bpy.ops.object.randomize_transform(
        random_seed=transform_seed, 
        loc=(10, 10, 10),
        rot=(dpi, dpi, dpi),
        scale=(10, 10, 10),
    )
    bpy.ops.export_mesh.stl(
        filepath=os.path.join(model_folder, f"{uuid.uuid4()}.stl"), 
    )
    bpy.ops.object.delete()
