import numpy as np
from curobo.wrap.model.curobo_robot_world import RobotWorld
from curobo.assembly.Part import Part
import pyikfastABB2600 as abb2600
import pyikfastgofa as gofa
class IKSolver:
    def __init__(self, robot_dict, robot_cfg):
        self.robot_dict = robot_dict
        self.robot_cfg = robot_cfg
        self.load_robot()

    def load_robot(self, robot_name):
        # IK
        planning = self.robot_dict.get("planning", {})
        if planning["ik_solver"] == "abb2600":
            self.solver = abb2600
        if planning["ik_solver"] == "gofa":
            self.solver = gofa
        self.jnds = np.array(planning["jnds"])

        # Base
        self.base_link_names = planning["base_link_names"]
        self.compute_baselink_frames()

    def compute_baselink_frames(self):
        self.inv_base_Ts = {}
        for robot_id, base_link_name in enumerate(self.base_link_names):
            idx = self.robot_cfg.kinematics.kinematics_config.link_name_to_idx_map[base_link_name]
            T = self.robot_cfg.kinematics.kinematics_config.fixed_transforms[idx]
            self.robot_cfg.kinematics.kinematics_config.fixed_transforms[idx] = T.detach().clone()
            T = T.detach().cpu().numpy()
            invT = np.linalg.inv(T)
            self.inv_base_Ts[base_link_name] = invT

    def solve(self, robotID, ee_frames):
        """ Compute inverse kinematics solution for given goal_frames
                    Args:
                        robot_id (int): robot id
                        goal_frames (np.array): robot ending effector frames for computing IK
                """
        if self.solver is None:
            return None

        ik_sols = []
        for goal_frame in goal_frames:
            frame = (self.inv_base_Ts[self.base_link_names[robot_id]] @ goal_frame).reshape(4, 4)
            pos = list(frame[:3, 3])
            rot = list(frame[:3, :3].reshape(-1))
            ik_sol = self.ik_solver.inverse(pos, rot)

            for sol in ik_sol:
                # clip the angles to be within [-pi, pi]
                nj = len(sol)
                sol_zero = np.array(sol, dtype=float)
                sol_minus_2pi = sol - np.ones(nj, dtype=float) * (np.pi * 2)
                sol_plus_2pi = sol + np.ones(nj, dtype=float) * (np.pi * 2)
                sol = np.zeros(nj, dtype=float)
                for js in [sol_zero, sol_plus_2pi, sol_minus_2pi]:
                    sol += js * (js >= -np.pi) * (js <= np.pi)

                # ensure the joint angles are within the predefine joint limits
                if ((sol >= self.joint_bounds_init[0, self.jnds[robot_id]]).all() \
                        and (sol <= self.joint_bounds_init[1, self.jnds[robot_id]]).all()):
                    ik_sols.append(sol)
        return np.array(ik_sols)

    def nj(self, robot_id):
        return len(self.jnds[robot_id])

class RobotWorldCollisionSolver:
    def __int__(self, robot_dict, world_cfg):
        self.robot_dict = robot_dict
        self.world_cfg = world_cfg

        # load robot
        self.load_robot()

        # create a RobotWorld for checking collision
        self.ccheck = RobotWorld(RobotWorld.load_from_config(self.robot_cfg, self.world_cfg, self.tensor_args))
        self.update_tools({})

        # initial joint angles
        self.retract_config_init = self.to_numpy(self.ccheck.kinematics.retract_config.detach())
        self.joint_bounds_init = self.to_numpy(self.ccheck.kinematics.kinematics_config.joint_limits.position)

    def to_numpy(self, x: torch.tensor):
        """Converts a tensor x to a numpy array.
        """
        return x.detach().clone().cpu().numpy()

    def to_torch(self, x: np.ndarray):
        """ Convert a numpy array x into GPU (torch tensor)
        """
        return torch.tensor(x, device=self.tensor_args.device, dtype=self.tensor_args.dtype)

    def nj(self):
        """ Total number of movable joints
        """
        return len(self.ccheck.kinematics.kinematics_config.joint_names)

    def overwrite_tool_spheres_config(self):
        """
        Adding more zero-size spheres to the tools of each robot
        This is required by the curobo.
        Once the robots are loaded on GPU, it is impossible to change the number of spheres for each joints
        This is problematic if we want to attach an object to a robot,
        which increases the number of spheres for the tool link.
        Thus, we first requires a list of empty spheres as reverse
        and later fill them with additional parts' spheres.
        """
        planning = self.robot_dict.get("planning", {})

        self.n_robot = planning.get("n_robot", 1)
        self.tool_link_names = planning.get("tool_link_names", ["tool"])
        self.jnds = np.array(planning["jnds"])

        # reserve GPU space for additional tool spheres
        tools = planning.get("tools", {})
        n_tool_spheres = tools.get("n_spheres", [50] * n_robot)
        collision_data = self.robot_cfg_dict["kinematics"]["collision_spheres"]
        for robot_id, tool_name in enumerate(tool_link_names):
            collision_data[tool_name] = []
            for id in range(n_tool_spheres[robot_id]):
                collision_data[tool_name].append({'center': [0.0, 0.0, -0.0], 'radius': 0.00})

    def load_robot(self):
        """ load robot
        """
        self.robot_cfg_dict = self.robot_dict["robot_cfg"]
        self.overwrite_tool_spheres_config()
        self.robot_cfg = RobotConfig.from_dict(self.robot_cfg_dict)
        self.tensor_args = self.robot_cfg.tensor_args

    def create_world(self, world_name):
        """ load world yaml file
            Args: world_name, a yaml file stores world geometry
        """
        if world_name is not None:
            yaml_file = join_path(get_world_configs_path(), world_name)
            self.world_config_dict = load_yaml(yaml_file)
            self.assembly = AssemblyGUI()
            self.assembly.from_yaml(get_world_configs_path(), get_assets_path(), world_name)
            world_cfg = WorldConfig.from_dict(self.world_config_dict)
            self.world_cfg = WorldConfig.create_collision_support_world(world_cfg)
        else:
            self.world_cfg = None

    def compute_tool_spheres(self,
                             robot_id,
                             grasp_frame: np.ndarray,
                             grasp_part: Part):
        """ Approximate a given tool using spheres
            Args:
                robot_id, the robot ID
                grasp_frame: the pose of robots' tool for grasping the part
                grasp_mesh: the part which robots to grasp
         """
        np.random.seed(12)
        curr_id = 0
        n_spheres = self.tools["n_spheres"][robot_id]
        spheres = np.zeros((n_spheres, 4), dtype=np.float32)
        if self.tools["shapes"][robot_id] == "L":
            spheres[0, :] = np.array([0, 0, 0.025, 0.075])
            spheres[1, :] = np.array([0, 0.05, 0.025, 0.065])
            spheres[2, :] = np.array([0, 0, 0.075, 0.07])
            spheres[3, :] = np.array([-0.01, -0.1, 0.05, 0.045])
            curr_id = 4
            radius = 0.02
            for id in range(0, 6):
                spheres[curr_id, :] = np.array([0.00 + id * 0.022, 0, 0.144, radius])
                curr_id = curr_id + 1

        if grasp_frame is not None and grasp_part is not None:

            n_object_spheres = n_spheres - curr_id
            sphs = self.grasp_part.approx_spheres(n_object_spheres)

            # transform the spheres
            radius = np.copy(sphs[:, 3])
            sphs[:, 3] = 1
            sphs = sphs @ np.linalg.inv(grasp_frame).T
            sphs[:, 3] = radius

            # add object spheres to robot's tool
            object_spheres = (sphs).tolist()
            object_spheres.sort(key=lambda sp: sp[3], reverse=True)
            for id in range(len(object_spheres)):
                if curr_id < n_spheres:
                    obj_sph = object_spheres[id]
                    spheres[curr_id, :3] = np.array(obj_sph[:3])
                    spheres[curr_id, 3] = obj_sph[3]
                    curr_id += 1

        return spheres

    def update_tools(self, tool_spheres={}):
        """ Updating robots' tool spheres
            Args:
                tool_spheres: a dict of list of spheres,
                each list stores spheres that approximate the robot's tool and attached part
        """
        for robot_id in range(self.n_robot):

            # remove tool spheres
            tool_link_name = self.tool_link_names[robot_id]
            self.checker.kinematics.kinematics_config.detach_object(tool_link_name)

            # get new spheres
            if robot_id in tool_spheres:
                spheres = tool_spheres[robot_id]
            else:
                spheres = self.tool_spheres(robot_id, task)

            # add spheres
            if spheres is not None:
                spheres_tensor = self.to_torch(spheres)
                self.checker.kinematics.kinematics_config.attach_object(sphere_tensor=spheres_tensor,
                                                                        link_name=tool_link_name)

    def update_world_with_offset_parts(self,
                                       part_status: np.array,
                                       parts_offset: dict):
        """ Adding new parts into the collision check
        by translating them using vectors stored in part_offset
            Args:
                part_status: the status of each part (installed or not)
                parts_offset: the translation vectors for parts to be added
        """
        new_world_config = WorldConfig()
        for part_id in range(part_status.shape[0]):
            obj_name = f"obj_{part_id}"

            if part_status[part_id] == 0:
                continue

            if part_id in parts_offset:
                for kd, v in enumerate(parts_offset[part_id]):
                    new_object = copy.deepcopy(self.world_cfg.get_obstacle(obj_name))
                    new_object.pose[: 3] = v
                    new_object.name += f"_{kd}"
                    new_world_config.add_obstacle(new_object)
                    new_object.texture = "visible"
            else:
                object = copy.deepcopy(self.world_cfg.get_obstacle(obj_name))
                object.color = self.obstacle_color(part_id, part_status)
                new_world_config.add_obstacle(object)
                object.texture = "visible"

        self.checker = RobotWorld(RobotWorld.load_from_config(self.robot_cfg,
                                                              new_world_config,
                                                              self.tensor_args))

    def update_world(self, part_status: np.array):
        """ Disable parts in collision check according to their part_status
            Args:
                part_status: the status of each part (installed or not)
        """
        for part_id in range(part_status.shape[0]):
            obj_name = f"obj_{part_id}"
            object_ccheck = self.ccheck.world_model.world_model.get_obstacle(obj_name)
            object_world = self.world_cfg.get_obstacle(obj_name)
            object_world.color = object_ccheck.color = self.obstacle_color(part_id, part_status)
            if part_status[part_id] == 0:
                self.ccheck.world_model.enable_obstacle(obj_name, False)
                object_world.texture = object_ccheck.texture = "invisible"
            else:
                self.ccheck.world_model.enable_obstacle(obj_name, True)
                object_world.texture = object_ccheck.texture = "visible"

    def check(self, sub_robotIDs: list[int], sub_js: list[np.ndarray]):
        js = np.repeat(self.retract_config_init, (sub_js.shape[0], 1))
        for robot_id in sub_robotIDs:
            js[:, self.jnds[robot_id]] = sub_js[robot_id]
        torch_js = self.to_torch(js)
        mask = self.checker.validate(torch_js).view(-1)
        mask = self.to_numpy(mask)
        return mask
