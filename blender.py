
cube = bpy.data.objects["Cube"]

cube.location = pose

matr = bpy.data.materials.new("Red")
matr.diffuse_color = (1,0,0,0.8)
cube.active_material = matr