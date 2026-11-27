#!/usr/bin/env python3

import time

import numpy as np
import robotic as ry

from aruco_utils import aruco_poses, detect_arucos, image_pts_to_3d_plane_fit
from my_manip import My_KOMO_Helper

table_height = 0.65
block_width = 0.039
block_length = 0.119
aruco_width = 9 * 0.004

f_cam = "camera_0"
f_cam_coll = "camera_0_coll"
f_gripper = "gripper"
f_palm = "palm"
f_table = "table"


def build_scene(num_blocks=20):
    C = ry.Config()
    # C.addFile("$RAI_PATH/scenarios/pandaSingle.g")
    C.addFile("../calibration/scenes/single_right/scene_calib.yml")
    C.addFrame("l_gripper", "gripper")
    for i in range(num_blocks):
        f = C.addFrame(f"block_{i}")
        f.setShape(ry.ST.ssBox, [block_width, block_width, block_length, 0.002])
        f.setColor([1.0, 0.8, 0.6])
        x = 0.05 + i // (num_blocks / 2) * 0.15
        y = 0.1 + 0.08 * (i % (num_blocks / 2))
        f.setPosition([x, y, table_height + block_width / 2 + 0.1])
        f.setQuaternion([1, 0, -1, 0] + 0.03 * np.random.randn(4))
        f.setMass(0.1)
        f.setContact(True)

        a = C.addFrame(f"aruco_{i}", args=f"shape: quad, size: [{aruco_width}, {aruco_width}], aruco_id: {1}")
        # a = C.addFrame(f"aruco_{i}", args = f"shape: marker, size: [{aruco_w}]")
        a.setParent(f)
        a.setRelativePoseByText("d(90 1 0 0) d(90 0 1 0)")
        a.setRelativePosition([block_width / 2 + 0.0001, 0.0, block_length / 2 - aruco_width / 2 - 0.002])

    C.addFrame("camera_init", args="X: [0.446389, 2.29262, 1.6405, 0.0649853, -0.086942, -0.795562, 0.596069]")
    return C


def set_marker(C: ry.Config, name: str, pos: np.ndarray, parent: str, rot=None):
    """Show a single marker frame 'name' attached to the parent frame, with (3,) position and optional (3, 3) rotation relative to parent."""
    if name in C.getFrameNames():
        f = C.getFrame(name)
    else:
        f = C.addFrame(name, parent)
        f.setShape(ry.ST.marker, [0.02])
    f.setRelativePosition(pos)
    if rot is not None:
        f.setRelativeRotationMatrix(rot)


def plan_look(C, camera, obj_name, gripper, objs, verbose=1):
    waypoints = My_KOMO_Helper("look")
    waypoints.setup_sequence(C, 1, 1e-1, 1e-1)  # significant homing bias
    waypoints.komo.addObjective([1.0], ry.FS.positionRel, [obj_name, camera], ry.OT.eq, [1e1, 1e1, 0])  # look at
    waypoints.komo.addObjective([1.0], ry.FS.positionRel, [obj_name, camera], ry.OT.ineq, [0, 0, -1e1], [0.25])  # keep dist
    waypoints.komo.addObjective([1.0], ry.FS.positionRel, [obj_name, camera], ry.OT.ineq, [0, 0, 1e1], [0.35])  # not too far to aruco?
    waypoints.komo.addObjective([1.0], ry.FS.positionRel, [camera, obj_name], ry.OT.sos, [0, 1e0, 1e0])  # straight from top
    waypoints.komo.addObjective([1.0], ry.FS.qItself, ["panda_joint5", "panda_joint7"], ry.OT.eq, [1e0], [0.0, 0.0])  # straight from top
    ret = waypoints.solve(verbose)
    if not ret.feasible:
        return None

    motion0 = waypoints.sub_motion(0, collision_scale=0.5)
    motion0.retract([0.0, 0.15], gripper, acc=5.0)
    motion0.no_collisions([], [f_cam_coll, f_table])
    for o in objs:
        motion0.no_collisions([0.2, 0.8], [gripper, o], margin=0.05, scale=1e2)
    ret = motion0.solve(verbose)
    if not ret.feasible:
        return None

    return motion0.path


def plan_pnp(C, gripper, obj_name, goal_frame, objs, verbose=1):
    waypoints = My_KOMO_Helper("pnp")
    waypoints.setup_sequence(C, 2, 1e-2, 1e-1)
    waypoints.grasp_box_constraints(1.0, gripper, obj_name, f_palm, grasp_direction="y", margin=0.02)
    waypoints.grasp_attach(1.0, gripper, obj_name)
    # waypoints.komo.addObjective([1.0], ry.FS.vectorZ, [gripper], ry.OT.sos, [1e0], [0, 0, 1.0])
    waypoints.komo.addObjective([2.0], ry.FS.poseDiff, [obj_name, goal_frame], ry.OT.eq, [1e1])
    waypoints.no_collisions([], [f_cam_coll, obj_name, f_cam_coll, f_table])
    ret = waypoints.solve(verbose)
    if not ret.feasible:
        return None, None

    motion0 = waypoints.sub_motion(0, collision_scale=0.0)
    # motion0.retract([0.0, 0.1], gripper, acc=5.0)
    motion0.approach([0.85, 1.0], gripper, acc=5.0)
    # for o in objs:
    #     motion0.no_collisions([0.2, 0.8], [gripper, o], margin=0.05, scale=1e2)
    ret = motion0.solve(verbose)
    if not ret.feasible:
        return None, None

    motion1 = waypoints.sub_motion(1, collision_scale=0.0)
    # for o in objs:
    #     if o != obj_name:
    #         motion1.no_collisions([0.1, 0.9], [obj_name, o], margin=0.02, scale=1e1)
    motion1.up([0.0, 0.15], gripper, acc=5.0)
    motion1.down([0.85, 1.0], gripper, acc=5.0)
    ret = motion1.solve(verbose)
    if not ret.feasible:
        return None, None

    return motion0.path, motion1.path


def execute_pnp(C: ry.Config, bot: ry.BotOp, path1, path2, gripper, obj, real):
    bot.move(path1, [3.0])
    key = bot.wait(C)
    if key == ord("q"):
        return False

    if not real:
        bot.gripperMove(ry.ArgWord._left, 0.01, 1.5)  # closing
    else:
        bot.gripperClose(0)
    key = bot.wait(C, forTimeToEnd=False, forGripper=True)
    if key == ord("q"):
        return False
    if real:
        C.getFrame(obj).setParent(C.getFrame(gripper), True)

    bot.move(path2, [4.0])
    key = bot.wait(C)
    if key == ord("q"):
        return False

    for _ in range(5):
        bot.sync(C, 0.1)

    # bot.wait(C, True, False)

    bot.gripperMove(ry.ArgWord._left, 0.06, 1.5)  # opening
    key = bot.wait(C, forTimeToEnd=False, forGripper=True)
    if key == ord("q"):
        return False
    if real:
        C.getFrame(obj).unLink()
    for _ in range(5):
        bot.sync(C, 0.1)

    return True


def set_physx_parameters(sim_speed):
    # these are global parameters that influence friction, etc...
    ry.params_add(
        {
            # 'physx/angularDamping': 0.1,
            "physx/defaultFriction": 5.0,
            "physx/defaultRestitution": 0.2,
            # 'physx/motorKp': 1000.,
            # 'physx/motorKd': 100.,
            "botsim/hyperSpeed": sim_speed,
            "botsim/verbose": 0,
        }
    )  # 1 to see simulation display (not just sync'ed config); 4 to see physx internal model

    ry.params_add(
        {
            "RealSense/autoExposure": False,
            "RealSense/exposure": 100,
            "RealSense/white": 3000,
        }
    )


def main(verbose=1, sim_speed=1, real=False):
    set_physx_parameters(sim_speed)

    num_blocks = 8

    # -- setup the scene configuration:
    C = build_scene(num_blocks)
    C.view(False, "scene")
    Cgoal = ry.Config()
    Cgoal.addFile("target.yml")
    Cgoal.view(False, "goal")

    goal = C.addFrame("goal_helper")

    objs = [f"block_{i}" for i in range(num_blocks)] + ["table"]

    # launch the robot
    bot = ry.BotOp(C, useRealRobot=real)
    bot.gripperMove(0, 0.06, 1.5)
    V = ry.ImageViewer()

    fxycxy = C.getFrame(f_cam).getAttributes()["fxycxy"]

    bot.home(C)
    if real:
        realsense_frames = [f for f in C.getFrames() if "realsense" in f.getAttributes()]
        bot.launch_MultiRealSense(realsense_frames, captureColor=True, captureDepth=True)

    print("Waiting for images")
    while True:
        rgb, depth = bot.getImageAndDepth(f_cam)
        bot.sync(C, 0.1)
        if len(rgb):
            break

    for i in range(num_blocks):
        obj_name = f"block_{i}"
        goal.setPose(Cgoal.getFrame(obj_name).getPose())

        perceive = True
        if perceive:
            # look (retry until the look plan is feasible)
            path0 = None
            while path0 is None:
                path0 = plan_look(C, f_cam, obj_name, f_gripper, objs, verbose=verbose)
            bot.move(path0, [4.0])
            key = bot.wait(C)
            if key == ord("q"):
                break

            # this is important: you need to sleep, otherwise you grab the image during motion from a pretty undefined pose;
            time.sleep(0.8)
            # for _ in range(8):
            # bot.sync(C, 0.1)

            # grab image
            q_rgb = bot.get_q()  # memorize the robot pose when taking image
            rgb, depth = bot.getImageAndDepth(f_cam)
            if not real:
                depth += 0.003 * np.random.randn(*depth.shape)

            # OpenCV finding of arucos
            pts, ids = detect_arucos(rgb)
            if verbose > 0:
                V.view(rgb, False)
            if ids.size == 0:
                print("== can't find aruco!")
                break
            C.setJointState(q_rgb)
            found = []  # indices of markers with a valid plane fit
            for k in range(len(ids)):
                corners3d = image_pts_to_3d_plane_fit(pts[k], depth, fxycxy)
                if np.isnan(corners3d).any():
                    print(f"== plane fit failed for marker {k} (id {ids[k]})")
                    continue
                center, rot = aruco_poses(corners3d)
                print("------------ rotation:", rot)
                set_marker(C, f"marker_{k}", center, f_cam, rot)
                found.append(k)
            C.view(False)
            if not found:
                print("== no valid aruco plane fit!")
                break

            # pick the perceived marker closest to this block's aruco frame
            aruco = C.getFrame(f"aruco_{i}")
            j = min(found, key=lambda k: np.linalg.norm(C.getFrame(f"marker_{k}").getPosition() - aruco.getPosition()))

            # place model block so that its aruco frame coincides with the perceived marker: X_block = X_marker * X_aruco_rel^-1
            X = C.getFrame(f"marker_{j}").getTransform() @ np.linalg.inv(aruco.getRelativeTransform())
            C.getFrame(obj_name).setPosition(X[:3, 3]).setRotationMatrix(X[:3, :3])
            C.view(False)

        # pick-and-place (retry until the plan is feasible)
        path1 = path2 = None
        while path2 is None:
            path1, path2 = plan_pnp(C, f_gripper, obj_name, goal.name, objs, verbose=verbose)
        good = execute_pnp(C, bot, path1, path2, f_gripper, obj_name, real)

        # on the real robot, assume the object is exactly as intended (otherwise, mini collisions kill the KOMO problems)
        if real:
            C.getFrame(obj_name).setPose(goal.getPose())

        if not good:
            break

    bot.home(C)


if __name__ == "__main__":
    main(verbose=1, sim_speed=5.0, real=False)
