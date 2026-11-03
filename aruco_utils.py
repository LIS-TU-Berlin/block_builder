"""OpenCV/ArUco helpers: marker detection and mapping image points to camera-relative 3D."""

import cv2
import numpy as np


def detect_arucos(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Detect all ArUco markers (DICT_5X5_50) in an RGB image.
    Returns (corners, ids): corners of shape (N, 4, 2) with the pixel coordinates (x, y) of the 4 corners of each of the N detected markers,
    and ids of shape (N,) with the integer marker ids. The detected markers (outline, first corner, id) are drawn into rgb in place."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50), cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(corners) == 0:
        return np.zeros((0, 4, 2)), np.zeros((0,), dtype=int)
    cv2.aruco.drawDetectedMarkers(rgb, corners, ids, borderColor=(255, 0, 0))  # red in RGB
    return np.concatenate(corners, axis=0), ids.flatten().astype(int)  # each corners entry is (1, 4, 2), ids is (N, 1)


def image_pts_to_3d(pts: np.ndarray, depth: np.ndarray, fxycxy) -> np.ndarray:
    """Map (N, 2) image points (x, y) to (N, 3) camera-relative 3D points, using the depth image (meters per pixel)
    and fxycxy = [fx, fy, cx, cy]. Same pinhole convention as ry.depthImage2PointCloud: z is the depth along the optical axis."""
    fx, fy, cx, cy = fxycxy
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    depth = np.asarray(depth).squeeze()
    u = np.clip(np.round(pts[:, 0]).astype(int), 0, depth.shape[1] - 1)
    v = np.clip(np.round(pts[:, 1]).astype(int), 0, depth.shape[0] - 1)
    z = depth[v, u]
    x = (pts[:, 0] - cx) * z / fx
    y = (pts[:, 1] - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def aruco_corners_3d(corners: np.ndarray, fxycxy, marker_size: float) -> np.ndarray:
    """Map (N, 4, 2) ArUco corner pixels (as returned by detect_arucos) to (N, 4, 3) camera-relative 3D corners,
    using cv2.solvePnP with the known marker side length (no depth needed). Same camera convention as image_pts_to_3d."""
    fx, fy, cx, cy = fxycxy
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    h = marker_size / 2
    # marker corners in the marker's own frame, in OpenCV's corner order: top-left, top-right, bottom-right, bottom-left
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=float)
    out = np.zeros((len(corners), 4, 3))
    for k, c in enumerate(np.asarray(corners, dtype=float).reshape(-1, 4, 2)):
        ok, rvec, tvec = cv2.solvePnP(obj, c, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            out[k] = np.nan
            continue
        R, _ = cv2.Rodrigues(rvec)
        out[k] = obj @ R.T + tvec.reshape(1, 3)
    return out


def aruco_poses(corners3d: np.ndarray, marker_size: float = 7 * 0.004) -> tuple[np.ndarray, np.ndarray]:
    """Fit a square of the given edge length to (N, 4, 3) 3D corners (OpenCV order: top-left, top-right, bottom-right, bottom-left).
    Returns centers (N, 3) and rotation matrices (N, 3, 3), with x to the marker's right, y up, z out of the marker (Kabsch fit)."""
    h = marker_size / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
    centers = corners3d.mean(axis=1)
    rots = np.zeros((len(corners3d), 3, 3))
    for k, P in enumerate(corners3d):
        U, _, Vt = np.linalg.svd(obj.T @ (P - centers[k]))
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        rots[k] = Vt.T @ np.diag([1, 1, d]) @ U.T
    return centers, rots
