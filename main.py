#!/usr/bin/env python3


import time

import robotic as ry

from my_manip import My_KOMO_Helper

table_h = 0.65
w = 0.039
l = 0.129
aruco_w = 9 * 0.004


def build_scene(num_blocks=20):
    C = ry.Config()
    C.addFile("$RAI_PATH/scenarios/pandaSingle.g")
    f = C.getFrame("panda_collCameraWrist")
    if f is not None:
        C.delFrame(f)
    for i in range(num_blocks):
        f = C.addFrame(f"block_{i}")
        f.setShape(ry.ST.ssBox, [w, w, l, 0.002])
        f.setColor([1.0, 0.8, 0.6])
        f.setPosition([0.4, -w * num_blocks + 2 * w * i, table_h + w / 2])
        f.setQuaternion([1, 0, -1, 0])
        f.setMass(0.1)
        f.setContact(True)

        a = C.addFrame(f"aruco_{i}", args=f"shape: quad, size: [{aruco_w}, {aruco_w}], aruco_id: {1}")
        # a = C.addFrame(f"aruco_{i}", args = f"shape: marker, size: [{aruco_w}]")
        a.setParent(f)
        a.setRelativePoseByText("d(90 1 0 0) d(90 0 1 0)")
        a.setRelativePosition([w / 2 + 0.0001, 0.0, l / 2 - aruco_w / 2 - 0.002])

    C.addFrame("camera_init", args='X: "t(0 2. 1.2) d(90 1 0 0) d(180 0 0 1) d(-10 1 0 0)"')
    return C


def build_goal_scene(C: ry.Config):
    Cgoal = ry.Config()
    Cgoal.addConfigurationCopy(C)

    for i in range(100):
        f = Cgoal.getFrame(f"block_{i}")
        if f == None:
            break
        f.setQuaternion([1, 0, 0, 0])
        f.setPosition([0, 0.2, table_h + l / 2 + i * l])

    return Cgoal


def plan(C, gripper, obj_name, goal_frame, objs, verbose=1):
    waypoints = My_KOMO_Helper()
    waypoints.setup_sequence(C, 2, 1e-2, 1e-1)
    waypoints.grasp_box_constraints(1.0, gripper, obj_name, "l_palm", grasp_direction="y", margin=0.015)
    waypoints.grasp_attach(1.0, gripper, obj_name)
    waypoints.komo.addObjective([2.0], ry.FS.poseDiff, [obj_name, goal_frame], ry.OT.eq)
    ret = waypoints.solve(0)
    print("grasp costs/feasibilities:", ret)  # this provides a metric for how good/feasible the grasp is kinematically; can be use to reject the grasp
    waypoints.view(verbose > 0)
    if not ret.feasible:
        return None, None

    motion1 = waypoints.sub_motion(0)
    motion1.approach([0.8, 1.0], gripper, dist=0.05)  ## this generates the motion to the first waypoint, with the last 20% constrained to be an 'approach' to the grasp
    for o in objs:
        motion1.no_collisions([0.2, 0.8], [gripper, o], margin=0.05, scale=1e2)
    ret = motion1.solve(0)
    motion1.view(verbose > 0)
    if not ret.feasible:
        return None, None

    motion2 = waypoints.sub_motion(1)
    for o in objs:
        if o != obj_name:
            motion2.no_collisions([0.1, 0.9], [obj_name, o], margin=0.02, scale=1e1)
    motion2.up([0.0, 0.15], gripper, acc=5.)
    motion2.down([0.85, 1.0], gripper, acc=5.)
    ret = motion2.solve(0)  ## no additional constraints at all on the motion between 1st and 2nd waypoint; later: up and down motion
    motion2.view(verbose > 0 or not ret.feasible)
    if not ret.feasible:
        return None, None

    return motion1.path, motion2.path


def execute(C, bot, path1, path2):
    bot.move(path1, [1.0])
    bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

    bot.gripperMove(ry.ArgWord._left, 0.03, 1.5)  # closing
    bot.wait(C, forKeyPressed=False, forTimeToEnd=False, forGripper=True)

    bot.move(path2, [2.0])
    bot.wait(C, forKeyPressed=False, forTimeToEnd=True)

    for t in range(5):
        bot.sync(C, 0.1)

    bot.gripperMove(ry.ArgWord._left, +1.0, 1.5)  # opening
    bot.wait(C, forKeyPressed=False, forTimeToEnd=False, forGripper=True)

    for t in range(5):
        bot.sync(C, 0.1)


def set_physx_parameters():
    # these are global parameters that influence friction, etc...
    ry.params_add(
        {
            # 'physx/angularDamping': 0.1,
            "physx/defaultFriction": 3.0,
            'physx/defaultRestitution': .2,
            # 'physx/motorKp': 1000.,
            # 'physx/motorKd': 100.,
            "botsim/hyperSpeed": 1.0,
            "botsim/verbose": 0,
        }
    )  # 1 to see simulation display (not just sync'ed config); 4 to see physx internal model


def main():
    set_physx_parameters()

    num_blocks = 4

    # -- setup the scene configuration:
    C = build_scene(num_blocks)
    C.view(False, "scene")
    Cgoal = build_goal_scene(C)
    Cgoal.view(False, "goal")

    goal = C.addFrame("goal_helper")
    # # this adds a helper frame to specify the predefined grasp, here hardcoded relative to the object; for AnyGrasp perhaps in world (no parent frame) or camera coordinates (camera as parent)
    # predefined_grasp = C.addFrame('predefined_grasp', 'obj')
    # predefined_grasp.setRelativePosition([.0,.0,.02])  #2cm above center
    # predefined_grasp.setShape(ry.ST.marker, [.1])

    objs = [f"block_{i}" for i in range(num_blocks)] + ["table"]

    # launch the robot
    bot = ry.BotOp(C, useRealRobot=False)

    for i in range(num_blocks):
        obj_name = f"block_{i}"
        goal.setPose(Cgoal.getFrame(obj_name).getPose())

        # plan
        t_start = time.perf_counter()
        path1, path2 = plan(C, "l_gripper", obj_name, goal.name, objs, verbose=0)
        print("planning time: ", time.perf_counter() - t_start)

        # execute
        if path1 is not None:
            # start a new robot sim each time (could do that in real)
            execute(C, bot, path1, path2)
        else:
            print("== fail ==")

    bot.home(C)


if __name__ == "__main__":
    main()
