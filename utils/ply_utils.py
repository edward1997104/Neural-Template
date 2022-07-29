import numpy as np
EPS = 1e-4
def read_ply_point_normal(shape_name):
    file = open(shape_name,'r')
    lines = file.readlines()

    start = 0
    while True:
        line = lines[start].strip()
        if line == "end_header":
            start += 1
            break
        line = line.split()
        if line[0] == "element":
            if line[1] == "vertex":
                vertex_num = int(line[2])
        start += 1

    vertices = np.zeros([vertex_num,3], np.float32)
    normals = np.zeros([vertex_num,3], np.float32)
    for i in range(vertex_num):
        line = lines[i+start].split()
        vertices[i,0] = float(line[0]) #X
        vertices[i,1] = float(line[1]) #Y
        vertices[i,2] = float(line[2]) #Z
        normals[i,0] = float(line[3]) #normalX
        normals[i,1] = float(line[4]) #normalY
        normals[i,2] = float(line[5]) #normalZ
    return vertices, normals


def triangulate_mesh_with_subdivide(vertices, faces, triangle_subdivide_cnt=3):
    vertices = np.array(vertices)
    new_faces = split_ply(faces)

    ### new
    cnt = 0
    final_vertices, final_faces = [], []
    for face in new_faces:
        face_vertices = vertices[face]

        base_1, base_2, base_3 = [face_vertices[0], face_vertices[1]], [face_vertices[1], face_vertices[2]], [
            face_vertices[0], face_vertices[2]]

        base_1_lin, base_2_lin, base_3_lin = np.linspace(base_1[0], base_1[1],
                                                         triangle_subdivide_cnt + 2), np.linspace(base_2[0],
                                                                                                  base_2[1],
                                                                                                  triangle_subdivide_cnt + 2), np.linspace(
            base_3[0], base_3[1], triangle_subdivide_cnt + 2)
        vertices_lin = [base_1_lin]
        for i in range(1, triangle_subdivide_cnt + 1):
            new_lin = np.linspace(base_3_lin[i], base_2_lin[i], triangle_subdivide_cnt + 2 - i)
            vertices_lin.append(new_lin)
        vertices_lin.append([face_vertices[2]])

        ## push
        vertices_to_append = np.zeros((0, 3))
        for vertex_lin in vertices_lin:
            vertices_to_append = np.concatenate((vertices_to_append, vertex_lin), axis=0)

        faces_to_append = []
        current_cnt = 0
        for i in range(triangle_subdivide_cnt + 1):
            for j in range(triangle_subdivide_cnt + 1 - i):
                faces_to_append.append(
                    [current_cnt + j, current_cnt + j + 1, current_cnt + (triangle_subdivide_cnt + 2 - i) + j])
                if i > 0:
                    faces_to_append.append(
                        [current_cnt + j, current_cnt - (triangle_subdivide_cnt + 1 - i) + j - 1,
                         current_cnt + j + 1])
            current_cnt += (triangle_subdivide_cnt + 2 - i)

        final_vertices.append(vertices_to_append)
        final_faces.append(np.array(faces_to_append) + cnt)
        cnt += vertices_to_append.shape[0]

    final_vertices = np.concatenate(tuple(final_vertices), axis=0)
    final_faces = np.concatenate(tuple(final_faces), axis=0)

    return final_vertices, final_faces

def split_ply(faces):

    result_faces = []
    for face in faces:
        for i in range(2, len(face)):
            result_faces.append([face[0], face[i-1], face[i]])

    return np.array(result_faces)
