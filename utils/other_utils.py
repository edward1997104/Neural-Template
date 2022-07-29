import torch
import numpy as np
import math
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

EPS = 1e-2



def save_off(filename, VPos, ITris, VColors=None):
    """
    Save a .off file
    Parameters
    ----------
    filename: string
        Path to which to write .off file
    VPos : ndarray (N, 3)
        Array of points in 3D
    VColors : ndarray(N, 3)
        Array of RGB colors
    ITris : ndarray (M, 3)
        Array of triangles connecting points, pointing to vertex indices
    """
    nV = VPos.shape[0]
    nF = ITris.shape[0]
    fout = open(filename, "w")
    if VColors.size == 0:
        fout.write("OFF\n%i %i %i\n"%(nV, nF, 0))
    else:
        fout.write("COFF\n%i %i %i\n"%(nV, nF, 0))
    for i in range(nV):
        fout.write("%g %g %g"%tuple(VPos[i, :]))
        if VColors is not None:
            fout.write(" %g %g %g"%tuple(VColors[i, :]))
        fout.write("\n")
    for i in range(nF):
        fout.write("3 %i %i %i\n"%tuple(ITris[i, :]))
    fout.close()


def compute_activate_plane(vertices, bsp_convex):
    input_vertices = np.array(vertices)
    input_vertices = np.concatenate((input_vertices, np.ones((input_vertices.shape[0], 1))), axis = 1)
    distance_values = input_vertices.dot(bsp_convex.T)

    results = [np.where(distance_values[i] <  1e-5)[0].tolist() for i in range(len(distance_values))]

    ## remap here first
    map_dict = {}
    cnt = 0
    for i in range(len(results)):
        for j in range(len(results[i])):
            if results[i][j] not in map_dict:
                map_dict[results[i][j]] = cnt
                results[i][j] = cnt
                cnt += 1
            else:
                results[i][j] = map_dict[results[i][j]]
    bsp_convex = np.array(bsp_convex)
    bsp_convex_active = bsp_convex[np.array(list(map_dict.keys()))]
    return results, bsp_convex_active

def reconstruct_plane_params(vertices, active_set, bsp_convex, vertices_weight):
    plane_count = max([max(active_planes) for active_planes in active_set]) + 1
    vertices = np.array(vertices)
    vertices_extended = np.concatenate((vertices, np.ones((vertices.shape[0], 1))), axis = 1)

    equation_cnt = 0
    row_i, row_j, values = [], [], []
    for i in range(len(active_set)):
        for j in range(len(active_set[i])):

            ## slow
            for k in range(4):
                row_i.append(equation_cnt)
                row_j.append(active_set[i][j] * 4 + k)
                values.append(vertices_extended[i][k] * vertices_weight[i])

            equation_cnt += 1

    ## initalize here
    WEIGHT = 1
    vector = np.zeros(equation_cnt)
    for i in range(len(bsp_convex)):
        for j in range(4):
            row_i.append(equation_cnt)
            row_j.append(i * 4 + j)
            values.append(WEIGHT)
            equation_cnt += 1

    row_i, row_j, values = np.array(row_i), np.array(row_j), np.array(values)

    ## construct sparse matrix
    matrix = coo_matrix((values, (row_i, row_j)), shape = (equation_cnt, plane_count * 4))

    ## construct the zero vector
    bsp_convex = np.array(bsp_convex).reshape(-1)
    vector = np.concatenate((vector, bsp_convex * WEIGHT))

    result = lsqr(matrix, vector)[0].reshape(-1, 4)

    return result




def get_mesh_watertight(bsp_convex_list):
    vertices = []
    polygons = []
    vertices_convex = []
    polygons_convex = []
    merge_threshold = 1e-4

    for k in range(len(bsp_convex_list)):
        vg, tg = digest_bsp(bsp_convex_list[k], bias=0)
        # if len(vg) <= 4:  ## break for only 1 plane
        #     continue
        biass = len(vertices)

        vertices_per_convex = []
        polygons_per_convex = []
        # merge same vertex
        mapping = np.zeros([len(vg)], np.int32)
        use_flag = np.zeros([len(vg)], np.int32)
        counter = 0
        for i in range(len(vg)):
            same_flag = -1
            for j in range(i):
                if abs(vg[i][0] - vg[j][0]) + abs(vg[i][1] - vg[j][1]) + abs(vg[i][2] - vg[j][2]) < merge_threshold:
                    same_flag = j
                    break
            if same_flag >= 0:
                mapping[i] = mapping[same_flag]
            else:
                mapping[i] = counter
                counter += 1
                use_flag[i] = True
        for i in range(len(vg)):
            if use_flag[i]:
                vertices.append(vg[i])
                vertices_per_convex.append(vg[i])
        for i in range(len(tg)):
            prev = mapping[tg[i][0]]
            tmpf = [prev + biass]
            for j in range(1, len(tg[i])):
                nowv = mapping[tg[i][j]]
                if nowv != prev:
                    tmpf.append(nowv + biass)
                    prev = nowv
            if tmpf[0] == tmpf[-1]:
                tmpf = tmpf[:-1]
            if len(tmpf) >= 3:
                polygons.append(tmpf)
                polygons_per_convex.append([value - biass for value in tmpf])

        vertices_convex.append(vertices_per_convex)
        polygons_convex.append(polygons_per_convex)

    return np.array(vertices), polygons, vertices_convex, polygons_convex


# Union parametric faces to form a mesh, output vertices and polygons
def digest_bsp(bsp_convex, bias):
    faces = []

    cnt = 0
    # carve out the mesh face by face
    for i in range(len(bsp_convex)):
        temp_face = get_polygon_from_params(bsp_convex[i])
        if temp_face is not None:
            faces = join_polygons(temp_face, faces)
        else:
            cnt += 1

    print(f"Empty plane cnt : {cnt}")

    vertices = []
    polygons = []

    # add "merge same vertex" in the future?
    v_count = bias
    for i in range(len(faces)):
        temp_face_idx = []
        for j in range(1, len(faces[i])):
            vertices.append(faces[i][j])
            temp_face_idx.append(v_count)
            v_count += 1
        polygons.append(temp_face_idx)

    return vertices, polygons


border_limit = 10.0
def get_polygon_from_params(params):
    epsilon = 1e-5
    face = []
    a,b,c,d = params
    sum = a*a+b*b+c*c
    if sum<epsilon:
        return None

    #detect intersection on the 12 edges of a box [-1000,1000]^3
    if abs(a)>=abs(b) and abs(a)>=abs(c):
        #x-direction (y,z) = (--,-+,++,+-)
        x1=-(b*(-border_limit)+c*(-border_limit)+d)/a
        x2=-(b*(-border_limit)+c*(border_limit)+d)/a
        x3=-(b*(border_limit)+c*(border_limit)+d)/a
        x4=-(b*(border_limit)+c*(-border_limit)+d)/a
        face.append([a,b,c,-d])
        if a>0:
            face.append(np.array([x1,-border_limit,-border_limit]))
            face.append(np.array([x2,-border_limit,border_limit]))
            face.append(np.array([x3,border_limit,border_limit]))
            face.append(np.array([x4,border_limit,-border_limit]))
        else:
            face.append(np.array([x4,border_limit,-border_limit]))
            face.append(np.array([x3,border_limit,border_limit]))
            face.append(np.array([x2,-border_limit,border_limit]))
            face.append(np.array([x1,-border_limit,-border_limit]))
    elif abs(b)>=abs(c):
        #y-direction (x,z) = (--,-+,++,+-)
        y1=-(a*(-border_limit)+c*(-border_limit)+d)/b
        y2=-(a*(-border_limit)+c*(border_limit)+d)/b
        y3=-(a*(border_limit)+c*(border_limit)+d)/b
        y4=-(a*(border_limit)+c*(-border_limit)+d)/b
        face.append([a,b,c,-d])
        if b<0:
            face.append(np.array([-border_limit,y1,-border_limit]))
            face.append(np.array([-border_limit,y2,border_limit]))
            face.append(np.array([border_limit,y3,border_limit]))
            face.append(np.array([border_limit,y4,-border_limit]))
        else:
            face.append(np.array([border_limit,y4,-border_limit]))
            face.append(np.array([border_limit,y3,border_limit]))
            face.append(np.array([-border_limit,y2,border_limit]))
            face.append(np.array([-border_limit,y1,-border_limit]))
    else:
        #z-direction (x,y) = (--,-+,++,+-)
        z1=-(a*(-border_limit)+b*(-border_limit)+d)/c
        z2=-(a*(-border_limit)+b*(border_limit)+d)/c
        z3=-(a*(border_limit)+b*(border_limit)+d)/c
        z4=-(a*(border_limit)+b*(-border_limit)+d)/c
        face.append([a,b,c,-d])
        if c>0:
            face.append(np.array([-border_limit,-border_limit,z1]))
            face.append(np.array([-border_limit,border_limit,z2]))
            face.append(np.array([border_limit,border_limit,z3]))
            face.append(np.array([border_limit,-border_limit,z4]))
        else:
            face.append(np.array([border_limit,-border_limit,z4]))
            face.append(np.array([border_limit,border_limit,z3]))
            face.append(np.array([-border_limit,border_limit,z2]))
            face.append(np.array([-border_limit,-border_limit,z1]))

    return face


# put a plane into the mesh
# split faces if necessary
def join_polygons(target_face, face_group):
    epsilon = 1e-5
    faces = []
    a, b, c, w = target_face[0]

    for i in range(len(face_group)):
        # split each face in face_group, if necessary
        # first detect whether split is needed
        face_i = face_group[i]
        front_flag = False
        back_flag = False
        vtypes = [-1]  # first element is a dummy
        for j in range(1, len(face_i)):
            dist = face_i[j][0] * a + face_i[j][1] * b + face_i[j][2] * c - w
            if dist < -epsilon:  # back--2
                back_flag = True
                vtypes.append(2)
            elif dist > epsilon:  # front--1
                front_flag = True
                vtypes.append(1)
            else:  # coplanar--0
                vtypes.append(0)

        if front_flag and back_flag:
            # split
            # only save front part
            face_i_new = [face_i[0]]
            for j in range(1, len(face_i)):
                j1 = j + 1
                if j1 == len(face_i):
                    j1 = 1
                if vtypes[j] != 2:
                    face_i_new.append(face_i[j])
                if vtypes[j] == 1 and vtypes[j1] == 2 or vtypes[j] == 2 and vtypes[j1] == 1:
                    dist1 = face_i[j][0] * a + face_i[j][1] * b + face_i[j][2] * c
                    dist2 = face_i[j1][0] * a + face_i[j1][1] * b + face_i[j1][2] * c
                    p = (w - dist1) * (face_i[j1] - face_i[j]) / (dist2 - dist1) + face_i[j]

                    dist1 = target_face[1][0] * a + target_face[1][1] * b + target_face[1][2] * c
                    dist2 = target_face[2][0] * a + target_face[2][1] * b + target_face[2][2] * c
                    dist3 = target_face[3][0] * a + target_face[3][1] * b + target_face[3][2] * c
                    dist4 = p[0] * a + p[1] * b + p[2] * c
                    face_i_new.append(p)
            faces.append(face_i_new)
        elif front_flag:
            faces.append(face_i)

    # also split target_face
    onsurface_flag = True
    result_face = []
    for k in range(len(target_face)):
        result_face.append(target_face[k])

    for i in range(len(face_group)):
        # first detect whether split is needed
        face_i = face_group[i]
        a, b, c, w = face_i[0]
        front_flag = False
        back_flag = False
        vtypes = [-1]  # first element is a dummy
        for j in range(1, len(result_face)):
            dist = result_face[j][0] * a + result_face[j][1] * b + result_face[j][2] * c - w
            if dist < -epsilon:  # back--2
                back_flag = True
                vtypes.append(2)
            elif dist > epsilon:  # front--1
                front_flag = True
                vtypes.append(1)
            else:  # coplanar--0
                vtypes.append(0)

        if front_flag and back_flag:
            # split
            # only save front part
            result_face_new = [result_face[0]]
            for j in range(1, len(result_face)):
                j1 = j + 1
                if j1 == len(result_face):
                    j1 = 1
                if vtypes[j] != 2:
                    result_face_new.append(result_face[j])
                if vtypes[j] == 1 and vtypes[j1] == 2 or vtypes[j] == 2 and vtypes[j1] == 1:
                    dist1 = result_face[j][0] * a + result_face[j][1] * b + result_face[j][2] * c
                    dist2 = result_face[j1][0] * a + result_face[j1][1] * b + result_face[j1][2] * c
                    p = (w - dist1) * (result_face[j1] - result_face[j]) / (dist2 - dist1) + result_face[j]
                    result_face_new.append(p)
            result_face = result_face_new
        elif back_flag:
            onsurface_flag = False
            break

    if onsurface_flag:
        faces.append(result_face)
    return faces

def write_ply_polygon(name, vertices, polygons):
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex "+str(len(vertices))+"\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("element face " + str(len(polygons)) + "\n")
    fout.write("property list uchar int vertex_index\n")
    fout.write("end_header\n")
    for ii in range(len(vertices)):
        fout.write(str(vertices[ii][0]) + " " + str(vertices[ii][1]) + " " + str(vertices[ii][2]) + "\n")
    for ii in range(len(polygons)):
        fout.write(str(len(polygons[ii])))
        for jj in range(len(polygons[ii])):
            fout.write(" " + str(polygons[ii][jj]))
        fout.write("\n")
    fout.close()


def write_ply_polygon_with_color(name, vertices, colors, polygons):
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex " + str(len(vertices)) + "\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("property uchar red\n")
    fout.write("property uchar green\n")
    fout.write("property uchar blue\n")
    fout.write("element face " + str(len(polygons)) + "\n")
    fout.write("property list uchar int vertex_index\n")
    fout.write("end_header\n")
    for ii in range(len(vertices)):
        fout.write(str(vertices[ii][0]) + " " + str(vertices[ii][1]) + " " + str(vertices[ii][2]) + " " + str(
            colors[ii][0]) + " " + str(colors[ii][1]) + " " + str(colors[ii][2]) + "\n")
    for ii in range(len(polygons)):
        fout.write(str(len(polygons[ii])))
        for jj in range(len(polygons[ii])):
            fout.write(" " + str(polygons[ii][jj]))
        fout.write("\n")
    fout.close()


def write_off_polygon(name, vertices, polygons):
    fout = open(name, 'w')
    fout.write("OFF\n")
    fout.write(f"{len(vertices)} {len(polygons)} 0\n")
    for ii in range(len(vertices)):
        fout.write(str(vertices[ii][0]) + " " + str(vertices[ii][1]) + " " + str(vertices[ii][2]) + "\n")
    for ii in range(len(polygons)):
        fout.write(str(len(polygons[ii])))
        for jj in range(len(polygons[ii])):
            fout.write(" " + str(polygons[ii][jj]))
        fout.write("\n")
    fout.close()

def sample_points_polygon(vertices, polygons, num_of_points):
    # convert polygons to triangles
    triangles, vertices = convert_triangles(polygons, vertices)

    return sample_points_triangles(num_of_points, triangles, vertices)


def convert_triangles(polygons, vertices):
    triangles = []
    for ii in range(len(polygons)):
        for jj in range(len(polygons[ii]) - 2):
            triangles.append([polygons[ii][0], polygons[ii][jj + 1], polygons[ii][jj + 2]])
    triangles = np.array(triangles, np.int32)
    vertices = np.array(vertices, np.float32)
    return triangles, vertices


def sample_points_triangles(num_of_points, triangles, vertices):
    small_step = 1.0 / 64
    epsilon = 1e-6
    triangle_area_list = np.zeros([len(triangles)], np.float32)
    triangle_normal_list = np.zeros([len(triangles), 3], np.float32)
    for i in range(len(triangles)):
        # area = |u x v|/2 = |u||v|sin(uv)/2
        a, b, c = vertices[triangles[i, 1]] - vertices[triangles[i, 0]]
        x, y, z = vertices[triangles[i, 2]] - vertices[triangles[i, 0]]
        ti = b * z - c * y
        tj = c * x - a * z
        tk = a * y - b * x
        area2 = math.sqrt(ti * ti + tj * tj + tk * tk)
        if area2 < epsilon:
            triangle_area_list[i] = 0
            triangle_normal_list[i, 0] = 0
            triangle_normal_list[i, 1] = 0
            triangle_normal_list[i, 2] = 0
        else:
            triangle_area_list[i] = area2
            triangle_normal_list[i, 0] = ti / area2
            triangle_normal_list[i, 1] = tj / area2
            triangle_normal_list[i, 2] = tk / area2
    triangle_area_sum = np.sum(triangle_area_list)
    sample_prob_list = (num_of_points / triangle_area_sum) * triangle_area_list
    triangle_index_list = np.arange(len(triangles))
    point_normal_list = np.zeros([num_of_points, 6], np.float32)
    count = 0
    watchdog = 0
    while (count < num_of_points):
        np.random.shuffle(triangle_index_list)
        watchdog += 1
        if watchdog > 100:
            print("infinite loop here!")
            return point_normal_list
        for i in range(len(triangle_index_list)):
            if count >= num_of_points: break
            dxb = triangle_index_list[i]
            prob = sample_prob_list[dxb]
            prob_i = int(prob)
            prob_f = prob - prob_i
            if np.random.random() < prob_f:
                prob_i += 1
            normal_direction = triangle_normal_list[dxb]
            u = vertices[triangles[dxb, 1]] - vertices[triangles[dxb, 0]]
            v = vertices[triangles[dxb, 2]] - vertices[triangles[dxb, 0]]
            base = vertices[triangles[dxb, 0]]
            for j in range(prob_i):
                # sample a point here:
                u_x = np.random.random()
                v_y = np.random.random()
                if u_x + v_y >= 1:
                    u_x = 1 - u_x
                    v_y = 1 - v_y
                point_normal_list[count, :3] = u * u_x + v * v_y + base
                point_normal_list[count, 3:] = normal_direction
                count += 1
                if count >= num_of_points: break
    return point_normal_list


def write_ply_point_normal(name, vertices, normals=None):
	fout = open(name, 'w')
	fout.write("ply\n")
	fout.write("format ascii 1.0\n")
	fout.write("element vertex "+str(len(vertices))+"\n")
	fout.write("property float x\n")
	fout.write("property float y\n")
	fout.write("property float z\n")
	fout.write("property float nx\n")
	fout.write("property float ny\n")
	fout.write("property float nz\n")
	fout.write("end_header\n")
	if normals is None:
		for ii in range(len(vertices)):
			fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+" "+str(vertices[ii,3])+" "+str(vertices[ii,4])+" "+str(vertices[ii,5])+"\n")
	else:
		for ii in range(len(vertices)):
			fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+" "+str(normals[ii,0])+" "+str(normals[ii,1])+" "+str(normals[ii,2])+"\n")
	fout.close()

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens
    return arr

## THIS BY GEOMETRIC AVERAGING
def compute_vertices_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    n = normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    norm = normalize_v3(norm)
    return norm

def load_off(filename):
    """
    Load in an OFF file, assuming it's a triangle mesh
    Parameters
    ----------
    filename: string
        Path to file
    Returns
    -------
    VPos : ndarray (N, 3)
        Array of points in 3D
    VColors : ndarray(N, 3)
        Array of RGB colors
    ITris : ndarray (M, 3)
        Array of triangles connecting points, pointing to vertex indices
    """
    fin = open(filename, 'r')
    nVertices = 0
    nFaces = 0
    lineCount = 0
    face = 0
    vertex = 0
    divideColor = False
    VPos = np.zeros((0, 3))
    VColors = np.zeros((0, 3))
    ITris = np.zeros((0, 3))
    for line in fin:
        lineCount = lineCount+1
        fields = line.split() #Splits whitespace by default
        if len(fields) == 0: #Blank line
            continue
        if fields[0][0] in ['#', '\0', ' '] or len(fields[0]) == 0:
            continue
        #Check section
        if nVertices == 0:
            if fields[0] == "OFF" or fields[0] == "COFF":
                if len(fields) > 2:
                    fields[1:4] = [int(field) for field in fields]
                    [nVertices, nFaces, nEdges] = fields[1:4]
                    #Pre-allocate vertex arrays
                    VPos = np.zeros((nVertices, 3))
                    VColors = np.zeros((nVertices, 3))
                    ITris = np.zeros((nFaces, 3))
                if fields[0] == "COFF":
                    divideColor = True
            else:
                fields[0:3] = [int(field) for field in fields]
                [nVertices, nFaces, nEdges] = fields[0:3]
                VPos = np.zeros((nVertices, 3))
                VColors = np.zeros((nVertices, 3))
                ITris = np.zeros((nFaces, 3))
        elif vertex < nVertices:
            fields = [float(i) for i in fields]
            P = [fields[0],fields[1], fields[2]]
            color = np.array([0.5, 0.5, 0.5]) #Gray by default
            if len(fields) >= 6:
                #There is color information
                if divideColor:
                    color = [float(c)/255.0 for c in fields[3:6]]
                else:
                    color = [float(c) for c in fields[3:6]]
            VPos[vertex, :] = P
            VColors[vertex, :] = color
            vertex = vertex+1
        elif face < nFaces:
            # Assume the vertices are specified in CCW order
            fields = [int(i) for i in fields]
            ITris[face, :] = fields[1:fields[0] + 1]
            face = face + 1
    fin.close()
    VPos = np.array(VPos, np.float64)
    VColors = np.array(VColors, np.float64)
    ITris = np.array(ITris, np.int32)
    return (VPos, VColors, ITris)


