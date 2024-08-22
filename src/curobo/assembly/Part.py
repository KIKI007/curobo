import numpy as np
from trimesh import Trimesh
class Part:
    def __init__(self, partID, mesh: Trimesh):
        self.partID = partID
        self.mesh = mesh

    def planarity_decompose(self, planarity_tol_indegree=15.0):
        """ Decompose this part's mesh into planar patches.
            Args:
                planarity_tol_indegree: Tolerance for planarity decomposition.
        """
        select_pairs = (self.mesh.face_adjacency_angles < (planarity_tol_indegree / 180 * math.pi))

        groups = []
        parents = {}
        for fid, _ in enumerate(self.mesh.faces):
            groups.append({fid})
            parents[fid] = len(groups) - 1

        # combine pair of faces that are almost in the same plane
        for id, faces in enumerate(self.mesh.face_adjacency[select_pairs]):
            f0 = faces[0]
            f1 = faces[1]

            g0 = parents[f0]
            g1 = parents[f1]

            if g0 != g1:
                g1_group = copy.deepcopy(groups[g1])
                groups[g1] = set()
                for face_id in g1_group:
                    groups[g0].add(face_id)
                    parents[face_id] = g0

        # collect non-empty group
        # each group composes of a planar decomposition of the input mesh
        count = 0
        label = np.ones(self.mesh.faces.shape[0], dtype=int) * -1
        areas = []
        for face_set in groups:
            if face_set != set():
                area = 0
                for face in face_set:
                    label[face] = count
                    area += self.mesh.area_faces[face]
                areas.append(area)
                count = count + 1

        # sorted the planar regions
        # give them a new label
        areas = np.array(areas)
        sort_ind = np.argsort(-areas)
        new_labels = np.ones(self.mesh.faces.shape[0], dtype=int) * -1
        for new_label, old_label in enumerate(sort_ind):
            for id in range(label.shape[0]):
                if label[id] == old_label:
                    new_labels[id] = new_label

        return new_labels

    def compute_grasp_points(self, tool_radius: float,
                             density_dist: float,
                             planar_tol_indegree=15):
        """ Sample a set of points on this part for the robot tool to pick
            Args:
                tool_radius, the minimal required contact area between the tool and part
                density_dist, the minimal distance between two pick points
                planar_tol_indegree, the minimal angle between two adjacent faces to be considered as planar
            Return: N x 6 array where each row is [point, normal]
        """

        face_labels = mesh_planarity_decompose(self.mesh, planar_tol_indegree)

        # compute connected components
        max_label = np.max(face_labels) + 1
        contacts = []

        for label in range(max_label):
            patch = trimesh.Trimesh()
            patch.vertices = np.copy(self.mesh.vertices)
            patch.faces = np.copy(self.mesh.faces)
            patch.update_faces(mask=(face_labels == label))

            patch = trimesh.Trimesh.subdivide_to_size(patch, 0.01)

            normal = np.mean(patch.face_normals, axis=0)
            if np.linalg.norm(normal) < 1E-6:
                continue
            normal /= np.linalg.norm(normal)
            centroid = patch.centroid.reshape(1, 3)

            # project patch onto 2D
            transform = trimesh.geometry.plane_transform(None, normal)
            project_centroid = trimesh.transformations.transform_points(centroid, transform)
            project_centroid[0, 2] = 0
            transform = np.linalg.inv(transform)
            back_project_centroid = trimesh.transformations.transform_points(project_centroid, transform)
            offset = centroid - back_project_centroid
            polygon = patch.projected(normal)

            # medial axis for computing inscribe circles
            segment = polygon.medial_axis()
            tree = KDTree(polygon.vertices)
            dist_boundary, _ = tree.query(segment.vertices, 1)
            dist_centroid = np.linalg.norm(segment.vertices - project_centroid[0, :2], axis=1)

            patch_contacts = []
            sort_inds = np.argsort(dist_centroid)

            for jd in sort_inds:
                if dist_boundary[jd] > tool_radius:

                    new_contact_center = segment.vertices[jd]
                    new_contact_center = np.array([*new_contact_center, 0]).reshape(1, -1)

                    new_contact_center = trimesh.transformations.transform_points(new_contact_center,
                                                                                  transform) + offset
                    new_contact_center = new_contact_center.reshape(-1)

                    duplicate = False
                    for exist_contact in patch_contacts:
                        if np.linalg.norm(exist_contact - new_contact_center) < density_dist:
                            duplicate = True
                            break
                    if not duplicate:
                        patch_contacts.append(new_contact_center)

            if len(patch_contacts) > 0:
                patch_contacts = np.array(patch_contacts)
                normal = normal.reshape(1, -1)
                normal = normal.repeat(patch_contacts.shape[0], axis=0)
                result = np.hstack([patch_contacts, normal])
                contacts.extend(result)

        if len(contacts) > 0:
            circles = np.array(contacts)
            return circles
        else:
            return None

    def _compute_grasp_frames(self, grasp_points,
                              n_sample = 8):
        """ Sample a set of frames on this part for the robot tool to pick
            Args:
                grasp_points, the points where the robot can grasp
            Return:
                N x 4 x 4 array where each row is a 4x4 transformation matrix
        """
        frames = []
        for id, pt in enumerate(grasp_points):
            ct = grasp_points[id, 0:3].reshape(-1)
            normal = -grasp_points[id, 3:6].reshape(-1)
            transform = trimesh.geometry.plane_transform(ct, normal)
            transform = np.linalg.inv(transform)
            frames.append(transform)

        frames = np.array(frames)
        new_frames = []
        for frame in frames:
            for id in range(0, n_sample):
                angle = math.pi * 2 / n_sample * id
                xaxis = frame[:3, 0] * math.cos(angle) + frame[:3, 1] * math.sin(angle)
                yaxis = np.cross(frame[:3, 2], xaxis)
                new_frame = np.copy(frame)
                new_frame[:3, 0] = xaxis
                new_frame[:3, 1] = yaxis
                new_frames.append(new_frame)
        return new_frames

    def mesh_grasp_frames(self,
                          tool_radius: float,
                          density_dist: float,
                          planar_tol_indegree: float,
                          n_sample: int):
        """ Compute possible picking places of a mesh
            Args:
                tool_radius, the minimal required contact area between the tool and part
                density_dist, the minimal distance between two pick points
                planar_tol_indegree, the minimal angle between two adjacent faces to be considered as planar
                n_sample, the number of sample of in-plane rotation for each pick point
            Return:
                N x 4 x 4 array where each row is a 4x4 transformation matrix
        """

        grasp_points = mesh_grasp_points(mesh,
                                         tool_radius=tool_radius,
                                         density_dist=density_dist,
                                         planar_tol_indegree=planar_tol_indegree)
        return _mesh_grasp_frames(grasp_points, n_sample)

    def approx_spheres(n_spheres=50,
                       sample_surface=False):
        """ Approximate a given obstacle using spheres
            Args:
                n_spheres, number of spheres
                fit_type, type of sphere fitting algorithm to use
            Return:
                N x 4, a list of points where each row is [point, radius]
        """
        # sample object's spheres
        pts = np.array([])
        n_sample = n_spheres
        while pts.shape[0] < n_spheres:
            n_sample *= 2
            if sample_surface:
                pts, face_index = trimesh.sample.sample_surface(part_mesh, count=n_sample)
                pts -= part_mesh.face_normals[face_index] * 1E-4
                contained = part_mesh.contains(pts)
                pts = pts[contained]
            else:
                pts = trimesh.sample.volume_mesh(part_mesh, count=n_sample)
        query = trimesh.proximity.ProximityQuery(part_mesh)
        _, distance, _ = query.on_surface(pts)
        object_spheres = np.ones((pts.shape[0], 4), dtype=np.float32)
        object_spheres[:, :3] = pts
        object_spheres[:, 3] = distance - 1E-6
        return object_spheres
