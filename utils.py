import math

def get_wall_centers(wall_detections):
    """Return list of (cx, cy) for each wall bbox."""
    centers = []
    for w in wall_detections:
        x1, y1, x2, y2 = w['bbox']
        centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
    return centers

def euclid(a, b):
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def filter_right_side(main_pt, switch_pt, centers, min_walls=3):
    """
    Rotation‐invariant right‐side filter:
    Keep centers c where cross(main→switch, main→c) < 0 (clockwise = right).
    If that yields fewer than `min_walls`, return the original list.
    """
    mx, my = main_pt
    sx, sy = switch_pt
    vx, vy = (sx - mx, sy - my)
    right = []
    for c in centers:
        wx, wy = (c[0] - mx, c[1] - my)
        cross_z = vx * wy - vy * wx
        if cross_z < 0:
            right.append(c)
    return right if len(right) >= min_walls else centers.copy()

def compute_custom_path(main_pt, switch_pt, wall_detections):
    """
    1) Compute k=3 intermediate walls, but only from the 'right side' relative to main→switch.
    2) w1: among the two closest to main, pick the one farthest from switch.
    3) w2: among remaining, pick the one closest to w1.
    4) w3: among remaining, pick the one closest to switch.
    Path: [main, w1, w2?, w3?, switch].
    """
    centers = get_wall_centers(wall_detections)
    if not centers:
        return [main_pt, switch_pt], euclid(main_pt, switch_pt)

    # Filter walls to “right‐hand side” relative to main→switch
    usable = filter_right_side(main_pt, switch_pt, centers, min_walls=3)

    # 1) w1: among two closest to main, pick farthest from switch
    sorted_by_main = sorted(usable, key=lambda c: euclid(main_pt, c))
    top2 = sorted_by_main[:2]
    w1 = top2[0] if len(top2) == 1 else max(top2, key=lambda c: euclid(c, switch_pt))

    # Remove w1
    rem = [c for c in usable if c != w1]

    # 2) w2: closest to w1
    w2 = min(rem, key=lambda c: euclid(w1, c)) if rem else None
    if w2: rem.remove(w2)

    # 3) w3: closest to switch
    w3 = min(rem, key=lambda c: euclid(switch_pt, c)) if rem else None

    # Build path
    path = [main_pt, w1]
    if w2: path.append(w2)
    if w3: path.append(w3)
    path.append(switch_pt)

    # Total length
    total = sum(euclid(path[i], path[i+1]) for i in range(len(path)-1))
    return path, total
