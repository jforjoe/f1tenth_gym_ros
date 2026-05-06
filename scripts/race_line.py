"""
F1Tenth Race Line Waypoint Generator
pip install numpy opencv-python scipy scikit-image matplotlib
"""

import sys
import csv
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches


# ═══════════════════════════════════════════════════════════
#  CONFIG  —  edit these before running
# ═══════════════════════════════════════════════════════════

MAP_PATH      = "/home/joe/my_repo/Roborace/f1tenth_gym_ros/maps/Spielberg_map.png"
OUTPUT_CSV    = "waypoints.csv"

RESOLUTION    = 0.05
ORIGIN_X      = 0.0
ORIGIN_Y      = 0.0

N_WAYPOINTS   = 300
V_MAX         = 8.0
V_MIN         = 1.5

IMG_THRESHOLD = 200
SPLINE_SMOOTH = None


# ═══════════════════════════════════════════════════════════
#  STEP 1 — Load map and extract the drivable track corridor
# ═══════════════════════════════════════════════════════════

img = cv2.imread(MAP_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    sys.exit(f"[ERROR] Cannot read map: {MAP_PATH}")

IMG_H, IMG_W = img.shape
print(f"[INFO] Map loaded: {IMG_W}x{IMG_H} px")

_, free_thresh = cv2.threshold(img, IMG_THRESHOLD, 255, cv2.THRESH_BINARY)
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(free_thresh, connectivity=8)
area_label  = sorted([(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, num_labels)])
track_label = area_label[0][1]
track_mask  = (labels == track_label).astype(np.uint8)
print(f"[INFO] Track corridor: area={area_label[0][0]} px")


# ═══════════════════════════════════════════════════════════
#  STEP 2 — Skeletonize to get a 1-px-wide centerline
# ═══════════════════════════════════════════════════════════

skel     = skeletonize(track_mask).astype(np.uint8)
skel_pts = np.argwhere(skel > 0)
pts_xy   = skel_pts[:, ::-1].astype(float)
print(f"[INFO] Skeleton points: {len(pts_xy)}")


# ═══════════════════════════════════════════════════════════
#  STEP 3 — Order skeleton into a continuous closed loop
# ═══════════════════════════════════════════════════════════

def nn_order(pts):
    pts  = pts.copy()
    n    = len(pts)
    used = np.zeros(n, dtype=bool)
    order = [0]
    used[0] = True
    tree = cKDTree(pts)
    for _ in range(n - 1):
        _, idxs = tree.query(pts[order[-1]], k=min(20, n))
        for idx in idxs:
            if not used[idx]:
                order.append(idx)
                used[idx] = True
                break
    return pts[order]

ordered = nn_order(pts_xy)


# ═══════════════════════════════════════════════════════════
#  STEP 4 — Fit closed B-spline and resample evenly
# ═══════════════════════════════════════════════════════════

s = SPLINE_SMOOTH if SPLINE_SMOOTH is not None else len(ordered) * 3.0

try:
    tck, _ = splprep([ordered[:, 0], ordered[:, 1]], s=s, per=True, k=3)
except Exception as e:
    sys.exit(f"[ERROR] Spline fitting failed: {e}")

u = np.linspace(0, 1, N_WAYPOINTS, endpoint=False)
x_px, y_px = splev(u, tck)


# ═══════════════════════════════════════════════════════════
#  STEP 5 — Curvature-based velocity profile
# ═══════════════════════════════════════════════════════════

def curvature(x, y):
    dx, dy   = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    return (dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-8) ** 1.5

k     = curvature(x_px * RESOLUTION, y_px * RESOLUTION)
k_max = np.percentile(np.abs(k), 95)
v     = V_MAX * (1.0 - np.clip(np.abs(k) / k_max, 0, 1))
v     = np.clip(v, V_MIN, V_MAX)
v     = savgol_filter(v, window_length=min(15, N_WAYPOINTS - 1 | 1), polyorder=3)


# ═══════════════════════════════════════════════════════════
#  STEP 6 — Heading (yaw) from spline tangent
# ═══════════════════════════════════════════════════════════

dx, dy  = np.gradient(x_px), np.gradient(y_px)
heading = np.arctan2(-dy, dx)


# ═══════════════════════════════════════════════════════════
#  STEP 7 — Convert pixels → world coordinates (meters)
# ═══════════════════════════════════════════════════════════

xw = x_px * RESOLUTION + ORIGIN_X
yw = (IMG_H - y_px) * RESOLUTION + ORIGIN_Y


# ═══════════════════════════════════════════════════════════
#  STEP 8 — Interactive: click on the map to set start point
# ═══════════════════════════════════════════════════════════

clicked_px = [None]   # will hold (x_px, y_px) of click

def on_click(event):
    if event.inaxes and event.button == 1 and event.xdata is not None:
        cx, cy = float(event.xdata), float(event.ydata)
        clicked_px[0] = (cx, cy)

        dists   = np.sqrt((x_px - cx)**2 + (y_px - cy)**2)
        roll_by = int(np.argmin(dists))

        rx = np.roll(x_px, -roll_by)
        ry = np.roll(y_px, -roll_by)

        start_dot.set_data([rx[0]], [ry[0]])
        race_line.set_data(rx, ry)

        wx = rx[0] * RESOLUTION + ORIGIN_X
        wy = (IMG_H - ry[0]) * RESOLUTION + ORIGIN_Y
        ax.set_title(f"Click to set start  |  current: pixel=({cx:.0f},{cy:.0f})  world=({wx:.2f},{wy:.2f}) m\n"
                     f"Close window to confirm and save", fontsize=9)
        fig.canvas.draw_idle()

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img, cmap="gray", origin="upper")
race_line, = ax.plot(x_px, y_px, color="red",  linewidth=1.5)
start_dot, = ax.plot(x_px[0], y_px[0], "go",   markersize=10, zorder=5)
ax.set_title("Click anywhere on the race line to set the START point\nClose window to confirm and save", fontsize=10)
ax.axis("off")

green_patch = mpatches.Patch(color='green', label='Start waypoint')
red_patch   = mpatches.Patch(color='red',   label='Race line')
ax.legend(handles=[green_patch, red_patch], loc='lower right', fontsize=8)

fig.canvas.mpl_connect('button_press_event', on_click)
plt.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════
#  STEP 9 — Apply the chosen start and roll all arrays
# ═══════════════════════════════════════════════════════════

if clicked_px[0] is not None:
    cx, cy  = clicked_px[0]
    dists   = np.sqrt((x_px - cx)**2 + (y_px - cy)**2)
    roll_by = int(np.argmin(dists))
else:
    roll_by = 0
    print("[WARN] No click detected — using default start (index 0)")

x_px, y_px = np.roll(x_px, -roll_by), np.roll(y_px, -roll_by)
xw,   yw   = np.roll(xw,   -roll_by), np.roll(yw,   -roll_by)
v          = np.roll(v,    -roll_by)
heading    = np.roll(heading, -roll_by)
print(f"[INFO] Start set to pixel=({x_px[0]:.0f},{y_px[0]:.0f})  world=({xw[0]:.2f},{yw[0]:.2f}) m")


# ═══════════════════════════════════════════════════════════
#  STEP 10 — Save waypoints CSV
# ═══════════════════════════════════════════════════════════

with open(OUTPUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["x_m", "y_m", "velocity_mps", "heading_rad"])
    for xi, yi, vi, hi in zip(xw, yw, v, heading):
        w.writerow([f"{xi:.4f}", f"{yi:.4f}", f"{vi:.4f}", f"{hi:.4f}"])

print(f"[INFO] Saved {N_WAYPOINTS} waypoints → {OUTPUT_CSV}")


# ═══════════════════════════════════════════════════════════
#  STEP 11 — Final verification plot  (comment out once happy)
# ═══════════════════════════════════════════════════════════

dist     = distance_transform_edt(track_mask)
skel_vis = skel * 255

fig2, axes = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle("F1Tenth Race Line — verify then comment out Step 11", fontsize=13)

axes[0].set_title("A  Map + Race Line (pixels)")
axes[0].imshow(img, cmap="gray", origin="upper")
axes[0].plot(x_px, y_px, color="red", linewidth=1.5, label="race line")
axes[0].plot(x_px[0], y_px[0], "go", markersize=10, label="start")
axes[0].legend(fontsize=8)
axes[0].axis("off")

axes[1].set_title("B  Race Line — colored by velocity (m/s)")
norm   = plt.Normalize(V_MIN, V_MAX)
colors = cm.RdYlGn(norm(v))
for i in range(len(xw) - 1):
    axes[1].plot(xw[i:i+2], yw[i:i+2], color=colors[i], linewidth=2)
sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=axes[1], label="velocity (m/s)", shrink=0.8)
axes[1].plot(xw[0], yw[0], "go", markersize=10)
axes[1].set_aspect("equal")
axes[1].grid(True, alpha=0.3)
axes[1].set_xlabel("x (m)")
axes[1].set_ylabel("y (m)")

axes[2].set_title("C  Distance Transform + Skeleton")
axes[2].imshow(dist, cmap="plasma", origin="upper")
axes[2].imshow(np.ma.masked_where(skel_vis == 0, skel_vis), cmap="cool", alpha=0.8, origin="upper")
axes[2].plot(x_px, y_px, "w-", linewidth=1, alpha=0.7)
axes[2].axis("off")

plt.tight_layout()
plt.savefig("raceline_preview.png", dpi=150, bbox_inches="tight")
print("[INFO] Preview saved → raceline_preview.png")
plt.show()