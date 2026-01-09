import numpy as np
from pygltflib import (
    GLTF2, Scene, Node, Mesh, Primitive,
    Buffer, BufferView, Accessor,
    FLOAT, ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER
)

def create_wall_vertices(x1, y1, x2, y2, height, thickness):
    # Wall direction
    dx, dy = x2 - x1, y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    nx, ny = -dy / length, dx / length  # perpendicular

    t = thickness / 2

    # Bottom rectangle
    p1 = [x1 + nx*t, y1 + ny*t, 0]
    p2 = [x1 - nx*t, y1 - ny*t, 0]
    p3 = [x2 - nx*t, y2 - ny*t, 0]
    p4 = [x2 + nx*t, y2 + ny*t, 0]

    # Top rectangle
    p5 = [*p1[:2], height]
    p6 = [*p2[:2], height]
    p7 = [*p3[:2], height]
    p8 = [*p4[:2], height]

    vertices = np.array([
        p1, p2, p3, p4,  # bottom
        p5, p6, p7, p8   # top
    ], dtype=np.float32)

    indices = np.array([
        # sides
        0,1,5, 0,5,4,
        1,2,6, 1,6,5,
        2,3,7, 2,7,6,
        3,0,4, 3,4,7,
        # top
        4,5,6, 4,6,7,
        # bottom
        0,3,2, 0,2,1
    ], dtype=np.uint16)

    return vertices, indices

def build_gltf(walls, output="/tmp/walls.gltf"):
    gltf = GLTF2()
    buffer_data = bytearray()
    buffer_views = []
    accessors = []
    nodes = []

    for wall in walls:
        vertices, indices = create_wall_vertices(**wall)

        v_offset = len(buffer_data)
        buffer_data.extend(vertices.tobytes())
        i_offset = len(buffer_data)
        buffer_data.extend(indices.tobytes())

        v_view = BufferView(
            buffer=0,
            byteOffset=v_offset,
            byteLength=vertices.nbytes,
            target=ARRAY_BUFFER
        )
        i_view = BufferView(
            buffer=0,
            byteOffset=i_offset,
            byteLength=indices.nbytes,
            target=ELEMENT_ARRAY_BUFFER
        )

        v_accessor = Accessor(
            bufferView=len(buffer_views),
            componentType=FLOAT,
            count=len(vertices),
            type="VEC3",
            max=vertices.max(axis=0).tolist(),
            min=vertices.min(axis=0).tolist()
        )

        i_accessor = Accessor(
            bufferView=len(buffer_views) + 1,
            componentType=5123,  # UNSIGNED_SHORT
            count=len(indices),
            type="SCALAR"
        )

        buffer_views.extend([v_view, i_view])
        accessors.extend([v_accessor, i_accessor])

        mesh = Mesh(
            primitives=[Primitive(
                attributes={"POSITION": len(accessors) - 2},
                indices=len(accessors) - 1
            )]
        )

        gltf.meshes.append(mesh)
        node = Node(mesh=len(gltf.meshes) - 1)
        nodes.append(node)
        gltf.nodes.append(node)

    gltf.buffers.append(Buffer(byteLength=len(buffer_data)))
    gltf.bufferViews = buffer_views
    gltf.accessors = accessors
    gltf.scenes = [Scene(nodes=list(range(len(nodes))))]
    gltf.scene = 0
    gltf.set_binary_blob(buffer_data)
    gltf.save(output)
