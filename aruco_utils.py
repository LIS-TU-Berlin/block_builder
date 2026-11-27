"""OpenCV/ArUco helpers: marker detection and mapping image points to camera-relative 3D."""

import cv2
import numpy as np


_DETECTOR_PARAMS = cv2.aruco.DetectorParameters()
_DETECTOR_PARAMS.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # default is NONE: integer contour corners, biased ~0.5px inwards
_DETECTOR = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50), _DETECTOR_PARAMS)


def detect_arucos(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Detect all ArUco markers (DICT_5X5_50) in an RGB image.
    Returns (corners, ids): corners of shape (N, 4, 2) with the pixel coordinates (x, y) of the 4 corners of each of the N detected markers,
    and ids of shape (N,) with the integer marker ids. The detected markers (outline, first corner, id) are drawn into rgb in place."""
    corners, ids, _ = _DETECTOR.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    if ids is None:
        return np.zeros((0, 4, 2)), np.zeros((0,), dtype=int)
    cv2.aruco.drawDetectedMarkers(rgb, corners, ids, borderColor=(255, 0, 0))  # red in RGB
    return np.concatenate(corners, axis=0), ids.flatten().astype(int)  # each corners entry is (1, 4, 2), ids is (N, 1)


def _backproject(x, y, z, fxycxy) -> np.ndarray:
    """Pinhole back-projection of pixel coordinates (x, y) with depth z to (N, 3) camera-relative 3D points
    (same convention as ry.depthImage2PointCloud: z is the depth along the optical axis)."""
    fx, fy, cx, cy = fxycxy
    return np.stack([(x - cx) * z / fx, (y - cy) * z / fy, z], axis=1)


def image_pts_to_3d(pts: np.ndarray, depth: np.ndarray, fxycxy) -> np.ndarray:
    """Map (N, 2) image points (x, y) to (N, 3) camera-relative 3D points, using the depth image (meters per pixel)
    at the (rounded) pixel and fxycxy = [fx, fy, cx, cy]."""
    u = np.clip(np.round(pts[:, 0]).astype(int), 0, depth.shape[1] - 1)
    v = np.clip(np.round(pts[:, 1]).astype(int), 0, depth.shape[0] - 1)
    return _backproject(pts[:, 0], pts[:, 1], depth[v, u], fxycxy)


def image_pts_to_3d_plane_fit(pts: np.ndarray, depth: np.ndarray, fxycxy) -> np.ndarray:
    """Like image_pts_to_3d, but robust to depth noise at the corners: gathers all pixels inside the convex hull of the
    (N, 2) image points (e.g. the 4 ArUco corners), fits a 3D plane to their (valid, >0) depth values (least squares in
    inverse depth, see below), and intersects each given point's viewing ray with that plane.
    Returns pts3d (N, 3) camera-relative points on the plane; all nan if fewer than 3 valid depth pixels lie inside the hull."""

    # depth values of all pixels inside the convex hull of the given image points
    mask = np.zeros(depth.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.round(pts).astype(np.int32)), 1)
    v, u = np.nonzero(mask)
    z = depth[v, u].astype(float)
    valid = np.isfinite(z) & (z > 0)
    if valid.sum() < 3:
        return np.full((len(pts), 3), np.nan)

    # plane fit in inverse depth: a 3D plane n.P = d with P = z * ((u-cx)/fx, (v-cy)/fy, 1) satisfies 1/z = a*u + b*v + c,
    # i.e. 1/z is linear in the (noise-free) pixel coordinates
    A = np.stack([u, v, np.ones_like(u)], axis=1)[valid].astype(float)
    abc = np.linalg.lstsq(A, 1 / z[valid], rcond=None)[0]

    # depth of the given points on that plane, back-projected
    zp = 1 / (np.stack([pts[:, 0], pts[:, 1], np.ones(len(pts))], axis=1) @ abc)
    if np.any(~np.isfinite(zp) | (zp <= 0)):
        return np.full((len(pts), 3), np.nan)
    return _backproject(pts[:, 0], pts[:, 1], zp, fxycxy)


def aruco_poses(corners3d: np.ndarray, marker_size: float = 7 * 0.004) -> tuple[np.ndarray, np.ndarray]:
    """Fit a square of the given edge length to the (4, 3) 3D corners of a single marker (OpenCV order: top-left, top-right, bottom-right, bottom-left).
    Returns center (3,) and rotation matrix (3, 3), with x to the marker's right, y up, z out of the marker (Kabsch fit)."""
    h = marker_size / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
    center = corners3d.mean(axis=0)
    U, _, Vt = np.linalg.svd(obj.T @ (corners3d - center))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:  # avoid reflection
        R = Vt.T @ np.diag([1, 1, -1]) @ U.T
    return center, R
