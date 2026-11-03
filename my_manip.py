import time

import numpy as np
import robotic as ry


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

class My_KOMO_Helper:
    def __init__(self, info=""):
        self.info = info
        self.komo = ry.KOMO()

    def setup_inverse_kinematics(self, C, homing_scale=1e-1, accumulated_collisions=True, joint_limits=True, quaternion_norms=True):
        """
        setup a 1 phase single step problem
        """
        self.komo.setTiming(1.0, 1, 1.0, 0)
        self.komo.setConfig(C, accumulated_collisions)
        self.komo.addControlObjective([], 0, homing_scale)
        if accumulated_collisions:
            self.komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e0])
        if joint_limits:
            self.komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, [1e0])
        if quaternion_norms:
            self.komo.addQuaternionNorms()

    def setup_sequence(self, C, K, homing_scale=1e-2, velocity_scale=1e-1, accumulated_collisions=True, joint_limits=True, quaternion_norms=True):
        self.komo.setTiming(float(K), 1, 1.0, 1)
        self.komo.setConfig(C, accumulated_collisions)
        self.komo.addControlObjective([], 0, homing_scale)
        self.komo.addControlObjective([], 1, velocity_scale)
        if accumulated_collisions:
            self.komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e0])
        if joint_limits:
            self.komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, [1e0])
        if quaternion_norms:
            self.komo.addQuaternionNorms()

    def setup_motion(self, C, K, steps_per_phase, homing_scale=0.0, acceleration_scale=1e-1, collision_scale=1e0, joint_limits_scale=1e0, quaternion_norms=True):
        self.komo.setTiming(K, steps_per_phase, 1.0, 2)
        self.komo.setConfig(C, True)  # accumulated_collisions)
        if homing_scale > 0.0:
            self.komo.addControlObjective([], 0, homing_scale)
        self.komo.addControlObjective([], 2, acceleration_scale)
        if collision_scale > 0:
            self.komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [collision_scale])
        if joint_limits_scale:
            self.komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, [joint_limits_scale])
        if quaternion_norms:
            self.komo.addQuaternionNorms()

        # zero vel at end
        self.komo.addObjective([float(K)], ry.FS.qItself, [], ry.OT.eq, [1e0], [], 1)

    def setup_point_to_point_motion(self, C, q1, homing_scale=1e-2, acceleration_scale=1e-1, collision_scale=1e0, joint_limits_scale=1e0, quaternion_norms=False):
        """
        setup a 1 phase fine-grained motion problem with 2nd order (acceleration) control costs
        """
        self.setup_motion(C, 1, 50, homing_scale, acceleration_scale, collision_scale, joint_limits_scale, quaternion_norms)

        qs = np.empty((1,), dtype=object)  # so annying to create an array of arrays
        qs[0] = q1
        self.komo.initWithWaypoints(qs, 1, interpolate=True, qHomeInterpolate=(0.2 if homing_scale > 1e-2 else 0.0), verbose=0)
        self.komo.addObjective([1.0], ry.FS.qItself, [], ry.OT.eq, scale=[1e0], target=q1)

    def add_stable_frame(self, jointType, parent, name, initFrame=None, markerSize=-1.0):
        if isinstance(initFrame, str):
            f = self.komo.addFrameDof(name, parent, jointType, True, initFrame, None)
        else:
            f = self.komo.addFrameDof(name, parent, jointType, True, None, initFrame)
        if markerSize > 0.0:
            f.setShape(ry.ST.marker, [0.1])
            f.setColor([1.0, 0.0, 1.0])

        # f.joint.sampleSdv=1.
        # f.joint.setRandom(self.timeSlices.d1, 0)

    def grasp_attach(self, time, gripper, obj):
        # -- option 1: old-style mode switches: //a temporary free stable joint gripper -> object
        # self.komo.addModeSwitch([1.,-1.], ry.SY.stable, [gripper, obj], True)
        # -- option 2: a permanent free stable gripper->grasp joint; and a snap grasp->object
        self.add_stable_frame(ry.JT.free, gripper, "obj_grasp", initFrame=obj)
        self.komo.addRigidSwitch(time, ["obj_grasp", obj])
        # -- option 3: a permanent free stable object->grasp joint; and a snap gripper->grasp
        # self.komo.add_stable_frame(ry.JT.free, obj, 'obj_grasp', initFrame=obj)
        # self.snap_switch(1., gripper, 'obj_grasp')

    def grasp_box_constraints(self, time, gripper, obj, palm, grasp_direction="x", margin=0.02):
        """
        general grasp of a box, squeezing along provided grasp_axis (-> 3
        possible grasps of a box), where and angle of grasp is decided by
        inequalities on grasp plan and no-collision of box and palm
        """
        if grasp_direction == "x":
            xLine = np.array([[1, 0, 0]])
            yzPlane = np.array([[0, 1, 0], [0, 0, 1]])
            align = [ry.FS.scalarProductXY, ry.FS.scalarProductXZ]
        elif grasp_direction == "y":
            xLine = np.array([[0, 1, 0]])
            yzPlane = np.array([[1, 0, 0], [0, 0, 1]])
            align = [ry.FS.scalarProductXX, ry.FS.scalarProductXZ]
        elif grasp_direction == "z":
            xLine = np.array([[0, 0, 1]])
            yzPlane = np.array([[1, 0, 0], [0, 1, 0]])
            align = [ry.FS.scalarProductXX, ry.FS.scalarProductXY]
        else:
            raise ValueError("grasp_direction not defined:", grasp_direction)

        boxSize = self.komo.getConfig().getFrame(obj).getSize()[:3]

        # position: center in inner target plane X-specific
        self.komo.addObjective([time], ry.FS.positionRel, [gripper, obj], ry.OT.eq, xLine * 1e1)
        self.komo.addObjective([time], ry.FS.positionRel, [gripper, obj], ry.OT.ineq, yzPlane * 1e1, 0.5 * boxSize - margin)
        self.komo.addObjective([time], ry.FS.positionRel, [gripper, obj], ry.OT.ineq, yzPlane * (-1e1), -0.5 * boxSize + margin)

        # orientation: grasp axis orthoginal to target plane X-specific
        self.komo.addObjective([time - 0.2, time], align[0], [gripper, obj], ry.OT.eq, [1e0])
        self.komo.addObjective([time - 0.2, time], align[1], [gripper, obj], ry.OT.eq, [1e0])

        # no collision with palm
        self.komo.addObjective([time - 0.3, time], ry.FS.distance, [palm, obj], ry.OT.ineq, [1e1], [-0.001])

    def no_collisions(self, time_interval, objs, margin=0.001, scale=1e1):
        """
        inequality on distance between pairs of objects
        """
        for i in range(0, len(objs), 2):
            self.komo.addObjective(time_interval, ry.FS.negDistance, [objs[i], objs[i + 1]], ry.OT.ineq, [scale], [-margin])

    def up(self, time_interval, gripper, acc=5.):
        self.komo.addObjective(time_interval, ry.FS.position, [gripper], ry.OT.eq, [1e1, 1e1, 0], order=1)
        self.komo.addObjective(time_interval, ry.FS.angularVel, [gripper], ry.OT.eq, [1e0], order=1)
        self.komo.addObjective(time_interval, ry.FS.position, [gripper], ry.OT.sos, [1e1], [0,0,acc], order=2)

    def down(self, time_interval, gripper, acc=5.):
        self.komo.addObjective(time_interval, ry.FS.position, [gripper], ry.OT.eq, [1e1, 1e1, 0], order=1)
        self.komo.addObjective(time_interval, ry.FS.angularVel, [gripper], ry.OT.eq, [1e0], order=1)
        self.komo.addObjective(time_interval, ry.FS.position, [gripper], ry.OT.sos, [1e1], [0,0,acc], order=2)

    def retract(self, time_interval, gripper, acc=5.):
        f = self.komo.getFrame(gripper, time_interval[0])
        z = f.getRotationMatrix()[:,2]
        Z = skew(z)

        # self.komo.addObjective(time_interval, ry.FS.position, [gripper], ry.OT.eq, 1e0 * Z, order=1)
        # self.komo.addObjective(time_interval, ry.FS.angularVel, [gripper], ry.OT.eq, [1e0], order=1)
        self.komo.addObjective(time_interval, ry.FS.position, [gripper], ry.OT.eq, [1e0], acc*z, order=2)

    def approach(self, time_interval, gripper, acc=5.):
        f = self.komo.getFrame(gripper, time_interval[1])
        z = f.getRotationMatrix()[:,2]
        Z = skew(z)

        # self.komo.addObjective(time_interval, ry.FS.position, [gripper], ry.OT.eq, 1e0 * Z, order=1)
        # self.komo.addObjective(time_interval, ry.FS.angularVel, [gripper], ry.OT.eq, [1e0], order=1)
        self.komo.addObjective(time_interval, ry.FS.position, [gripper], ry.OT.eq, [1e0], acc*z, order=2)

    def retract_alg(self, time_interval, gripper, dist=0.03):
        helper = f"_{gripper}_retract_{time_interval[0]}"
        f = self.komo.getFrame(gripper, time_interval[0])
        self.add_stable_frame(ry.JT.none, "", helper, f)

        self.komo.addObjective(time_interval, ry.FS.positionRel, [gripper, helper], ry.OT.eq, 1e2 * np.array([[1, 0, 0]]))
        self.komo.addObjective(time_interval, ry.FS.quaternionDiff, [gripper, helper], ry.OT.eq, [1e2])
        self.komo.addObjective([time_interval[1]], ry.FS.positionRel, [gripper, helper], ry.OT.ineq, -1e2 * np.array([[0, 0, 1]]), target=[0.0, 0.0, dist])

    def approach_alt(self, time_interval, gripper, dist=.05):
        helper = f"_{gripper}_approach_{time_interval[1]}"
        f = self.komo.getFrame(gripper, time_interval[1])
        self.add_stable_frame(ry.JT.none, "", helper, f)

        self.komo.addObjective(time_interval, ry.FS.positionRel, [gripper, helper], ry.OT.eq, 1e2 * np.array([[1, 0, 0]]))
        self.komo.addObjective(time_interval, ry.FS.quaternionDiff, [gripper, helper], ry.OT.eq, [1e2])
        self.komo.addObjective([time_interval[0]], ry.FS.positionRel, [gripper, helper], ry.OT.ineq, -1e2 * np.array([[0, 0, 1]]), target=[0.0, 0.0, dist])

    def solve(self, verbose=1):
        sol = ry.NLP_Solver()
        sol.setProblem(self.komo.nlp())
        sol.setOptions(damping=1e-1, verbose=verbose - 1, stopTolerance=1e-3, lambdaMax=100.0, stopInners=20, stopEvals=200)
        self.ret = sol.solve()
        if self.ret.feasible:
            self.path = self.komo.getPath()
        else:
            self.path = None
        if verbose > 0:
            if not self.ret.feasible:
                print(f"  -- infeasible:{self.info}\n     {self.ret}")
                if verbose > 1:
                    objs = self.komo.report()
                    # pprint.pp(objs)
                    for k, v in objs.items():
                        print(f"objective {k}: {v}")
                    self.komo.view(True, f"failed: {self.info}\n{self.ret}")
            else:
                print(f"  -- feasible:{self.info}\n     {self.ret}")
                if verbose > 2:
                    self.komo.view(True, f"success: {self.info}\n{self.ret}")

        return self.ret

    def view(self, pause):
        #     cout <<'  -- DEBUG: ' <<info <<endl
        #     cout <<'  == solver return: ' <<*ret <<endl
        #     cout <<'  == all KOMO objectives with increasing errors:\n' <<self.report(False, listObjectives, plotOverTime) <<endl
        # #  cout <<'  == objectives sorted by error and Lagrange gradient:\n' <<sol.reportLagrangeGradients(self.featureNames) <<endl
        #     cout <<'  == view objective errors over slices in gnuplot' <<endl
        #     cout <<'  == scroll through solution in display window using SHIFT-scroll' <<endl
        self.komo.view(pause, f"debug: {self.info}\n{self.ret}")

    def play(self, C: ry.Config, duration=1.0, savePngs=False):
        dofs = C.getJointIDs()
        path = self.komo.getPath(dofs)
        for t in range(self.path.shape[0]):
            C.setJointState(path[t])
            C.view(False, f"step {t}\n{self.info}")
            time.sleep(duration / self.path.shape[0])
            if savePngs:
                C.get_viewer().savePng()

    def sub_motion(self, phase, fixEnd=True, homing_scale=1e-2, acceleration_scale=1e-1, collision_scale=1e0):
        C, _, q1 = self.komo.getSubProblem(phase)
        komo = My_KOMO_Helper(f"sub_motion_{phase}--{self.info}")
        komo.setup_point_to_point_motion(C, q1, homing_scale, acceleration_scale, collision_scale)
        return komo

    def sub_rrt(self, phase, explicitCollisionPairs=None):
        (C, q0, q1) = self.komo.getSubProblem(phase)
        rrt = ry.RRT_PathFinder()
        rrt.setProblem(C, q0, q1)
        if explicitCollisionPairs is not None:
            rrt.setExplicitCollisionPairs(explicitCollisionPairs)
        return rrt

    @property
    def feasible(self):
        return self.ret.feasible
