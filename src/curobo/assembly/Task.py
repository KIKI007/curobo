import copy
import numpy as np
import itertools
from curobo.assembly.Solver import IKSolver, RobotWorldCollisionChecker
from curobo.assembly.Part import Part

class RobotGraspTask():
    def __init__(self,
                 robot_id: int,
                 part_id : int,
                 part_pose: np.ndarray,
                 ik_solver: IKSolver,
                 checker: RobotWorldCollisionChecker,
                 part_status):

        self.robot_id = robot_id
        self.ik_solver = ik_solver
        self.collision_checker = checker
        self.part_status = part_status

        self.part = Part(self.collision_checker.get_part(part_id))
        self.part_pose = part_pose

        self.ik_sols = []
        self.offset_ik_sols = []
        self.n = 0

    def checkCollision(self, ik_sols):
        self.collision_checker.update_world(self.part_status)
        return checker.check([self.robot_id], ik)

    def computeGraspIK(self):
        ee_frames = self.part.compute_grasp_points(self.part_pose)

        self.ik_sols = []
        self.ee_frames = []
        for id, eeFrame in enumerate(ee_frames):
            ik_sol = self.ik_solver.solve(self.robot_id, [eeFrame])
            self.ik_sols.extend(ik_sol)
            self.ee_frames.extend([eeFrame] * len(ik_sol))

        # to numpy array
        self.ik_sols = np.array(self.ik_sols)
        self.ee_frames = np.array(self.ee_frames)
        self.n = self.ik_sols.shape[0]

    def computeOffsetGraspIK(self, offset_ee_frames, offset_cspace_tol = 1.0):
        self.offset_ik_sols = np.zeros(self.ik_sols.shape,dtype=float)
        for id, offset_ee_frame in enumerate(offset_ee_frames):
            offset_ik_sols = ik_func(robot_id, [offset_ee_frame])
            offset_ik_sols = np.array(offset_ik_sol)
            if offset_ik_sol.shape[0] > 0:
                ik_sol = self.ik_sols[id, :]
                dist = np.linalg.norm(offset_ik_sols - ik_sol, axis=1)
                min_ind = np.argmin(dist)
                if dist[min_ind] < offset_cspace_tol:
                    self.ik_sols[self.n] = ik_sol
                    self.offset_ik_sols[self.n] = offset_ik_sols[min_ind]
                    self.n += 1

        self.ik_sols = self.ik_sols[: self.n]

    def insert(self, drt: np.ndarray):
        ee_frames = self.computeGraspIK()
        offset_ee_frames = ee_frames[:, :3, 3] + drt
        self.computeOffsetIK(offset_ee_frames)

    def release(self, dist: float):
        ee_frames = self.computeGraspIK()
        offset_ee_frames = np.copy(self.ee_frames)
        offset_ee_frames[:, :3, 3] -= self.ee_frames[:, :3, 0] * dist
        self.computeOffsetIK(offset_ee_frames)

    def stay(self):
        self.computeGraspIK()
        self.offset_ik_sols = np.copy(self.ik_sols)

    def remove_collision(self, ts = [0, 1]):

        # compute all ik solutions
        nT = len(ts)
        nJ = self.ik_solver.nj(self.robot_id)
        ik = np.zeros((nT * self.n, nJ), dtype=float)
        for id, t in enumerate(ts):
            ik[id * self.n : (id + 1) * self.n ] = self.ik_sols * t + self.offset_ik_sols * (1 - t)

        # check collision
        mask_all = self.checkCollision(ik)

        # remove collision
        mask = np.ones((self.n, nJ), dtype=bool)
        for id, t in enumerate(ts):
            sub_mask = mask_all[id * self.n : (id + 1) * self.n ]
            mask = mask * sub_mask
        for var_name in ["ik_sols", "offset_ik_sols"]:
            var = self.__dict__[var_name]
            var = var[sub_mask]
        self.n = mask.sum()
        return mask.any()

class RobotPathTask():


class RobotPickandPlaceTask():

    def __init__(self, robotID):
        self.robotID = robotID
        self.start_ik_sols = []
        self.goal_ik_sols = []
        self.start_offset_ik_sols = []
        self.goal_offset_ik_sols = []
        self.n = 0
    def compute(self, start: RobotGraspTask, goal: RobotGraspTask):
        start_inds = np.arange(0, start.n)
        goal_inds = np.arange(0, goal.n)
        for pair in itertools.product([start_inds, goal_inds]):
            start_ind = pair[0]
            goal_ind = pair[1]
            self.start_ik_sols.append(start.ik_sols[start_ind])
            self.goal_ik_sols.append(goal.ik_sols[goal_ind])
            self.start_offset_ik_sols.append(start.offset_ik_sols[start_ind])
            self.goal_offset_ik_sols.append(goal.offset_ik_sols[goal_ind])

        for var in [self.start_ik_sols, self.goal_ik_sols,
                    self.start_offset_ik_sols, self.goal_offset_ik_sols]:
            var = np.array(var, dtype=float)

        self.n = self.start_ik_sols.shape[0]
class MultiRobotPickandPlaceTasks():
    def __init__(self, robotIDs, partIDs):
        self.start_ik_sols = []
        self.start_offset_ik_sols = []
        self.goal_ik_sols = []
        self.goal_offset_ik_sols = []

        self.robotIDs = robotIDs
        self.partIDs = partIDs

    def compute(self, tasks: list[RobotPathTask]):
        inds = []
        for task in tasks:
            inds.append(np.arange(0, task.n))

        for ind in itertools.product(*inds):
            for kd, task in enumerate(tasks):
                self.start_ik_sols.append(task.start_ik_sols[ind[kd]])
                self.goal_ik_sols.append(task.goal_ik_sols[ind[kd]])
                self.start_offset_ik_sols.append(task.start_offset_ik_sols[ind[kd]])
                self.goal_offset_ik_sols.append(task.goal_offset_ik_sols[ind[kd]])

        for var in [self.start_ik_sols, self.goal_ik_sols,
                    self.start_offset_ik_sols, self.goal_offset_ik_sols]:
            var = np.array(var, dtype=float)
        self.n = self.start_ik_sols.shape[0]

    def mask(self, flag):
        for var in [self.start_ik_sols, self.goal_ik_sols,
                    self.start_offset_ik_sols, self.goal_offset_ik_sols]:
            var = var[flag]
        self.n = flag.sum()
        return flag.any()

    def remove_collision(self,
                         checker: RobotWorldCollisionChecker,
                         sub_robot_ids = self.robotIDs,
                         var_names = ["start_ik_sols", "goal_ik_sols", "start_offset_ik_sols", "goal_offset_ik_sols"]):
        for var_name in var_names:
            var = self.__dict__[var_name]
            flag = checker.check(sub_robot_ids, var)
            if not self.mask(flag):
                return False
        return True




