import numpy as np
import cv2
import random
import shapely



def generate_deformed_circle(m_px, image_size=350, circle_radius=90, waviness=3, num_points=100, thickness=2,
                             blur_strength=6, blur_repeat=8, dot_probability=0.5, dot_offset=5,
                             ellipse_ratio_range=0.03):
    """
    Generates a float image (values 0.0–1.0) with a wavy deformed circle (or ellipse) and optional blurred center dot.
    """

    # --- scale factor logic ---
    use_scale = random.random() > 0.3
    scale_factor = 1.0
    if use_scale:
        scale_factor = max(0.1, min(1.0, 1.2 - 0.6 * m_px))

    # --- scaled parameters ---
    circle_radius = int(circle_radius * scale_factor)
    waviness = int(max(1, waviness * scale_factor))
    thickness = max(1, int(thickness * scale_factor))
    dot_offset = int(dot_offset * scale_factor)
    dot_radius = max(1, int(4 * scale_factor))

    # Create base image
    img = np.zeros((image_size, image_size), dtype=np.float32)
    center = (image_size // 2, image_size // 2)

    # --- ellipse deformation ratio ---
    ellipse_ratio = 1.0 + random.uniform(-ellipse_ratio_range, ellipse_ratio_range)

    # --- generate wavy outline ---
    def add_waviness(center, radius, num_points, waviness, ellipse_ratio):
        points = []
        for i in range(num_points):
            angle = (i / num_points) * 2 * np.pi
            r_offset = random.randint(-waviness, waviness)
            x = int(center[0] + (radius + r_offset) * np.cos(angle) * ellipse_ratio)
            y = int(center[1] + (radius + r_offset) * np.sin(angle) / ellipse_ratio)
            points.append((x, y))
        return points

    wavy_edges = add_waviness(center, circle_radius, num_points, waviness, ellipse_ratio)

    # Draw outline on temp uint8 image
    temp = np.zeros_like(img, dtype=np.uint8)
    wavy_edges = np.array([wavy_edges], dtype=np.int32)
    cv2.polylines(temp, wavy_edges, isClosed=True, color=255, thickness=thickness * 2)

    # Optional center dot
    if random.random() < dot_probability:
        dot_x = center[0] + random.randint(-dot_offset, dot_offset)
        dot_y = center[1] + random.randint(-dot_offset, dot_offset)
        cv2.circle(temp, (dot_x, dot_y), radius=dot_radius, color=255, thickness=-1)

    # Blur if needed
    if blur_strength > 0:
        for _ in range(blur_repeat):
            temp = cv2.GaussianBlur(temp, (blur_strength * 2 + 1, blur_strength * 2 + 1), 0)

    # Normalize to float32
    img = temp.astype(np.float32) / 255.0

    return img

def generate_deformed_circle2(m_px, image_size=350, circle_radius=90, waviness=2, num_points=100, thickness=2,
                             blur_strength=6, blur_repeat=8, dot_probability=0.5, dot_offset=5,
                             ellipse_ratio_range=0.03, fade_factor=1.0, intensity_variation=10):
    """
    Generates a float image (0.0–1.0) with a wavy deformed circle or ellipse, optional dot,
    and fade effect with intensity variation along the contour.
    """
    if random.random() > 0.7:
        thickness = thickness - 1
        fade_factor = random.uniform(0.2, 5)
        intensity_variation = random.randint(1, 1000)

    if random.random() > 0.5:
        blur_repeat = random.randint(2, 5)

    if random.random() > 0.5:
        waviness = random.randint(0, 3)

    # --- škálování ---
    base_scale = 1.25 - 0.63 * m_px
    scale_noise = random.uniform(0.8, 1.2)
    scale_factor = (base_scale if random.random() > 0.3 else 1.0) * scale_noise

    circle_radius = int(circle_radius * scale_factor)
    waviness = max(1, int(waviness * scale_factor))
    thickness = max(1, int(thickness * scale_factor))
    dot_offset = int(dot_offset * scale_factor)
    dot_radius = max(1, int(4 * scale_factor))
    brightness_variation = intensity_variation * scale_factor

    # --- inicializace ---
    img = np.zeros((image_size, image_size), dtype=np.float32)
    center = (image_size // 2, image_size // 2)
    ellipse_ratio = 1.0 + random.uniform(-ellipse_ratio_range, ellipse_ratio_range)

    # --- fade + waviness ---
    def add_waviness(center, radius, num_points, waviness, intensity_variation, fade_factor, ellipse_ratio):
        points = []
        base_intensity = max(0, 255 - intensity_variation // 2)
        intensities = np.clip(
            np.random.randint(base_intensity, base_intensity + intensity_variation, num_points + 1), 0, 255
        ) / 255.0

        for i in range(num_points + 1):
            angle = (i / num_points) * 2 * np.pi
            r_offset = random.randint(-waviness, waviness)
            x = int(center[0] + (radius + r_offset) * np.cos(angle) * ellipse_ratio)
            y = int(center[1] + (radius + r_offset) * np.sin(angle) / ellipse_ratio)
            t = (1 - np.cos(i / num_points * np.pi * fade_factor + random.uniform(-0.5, 0.5))) / 2
            points.append((x, y, intensities[i] * t))
        return points

    # --- generování a vykreslení ---
    edge = add_waviness(center, circle_radius, num_points, waviness, brightness_variation, fade_factor, ellipse_ratio)

    for j in range(len(edge) - 1):
        pt1 = (edge[j][0], edge[j][1])
        pt2 = (edge[j + 1][0], edge[j + 1][1])
        value = edge[j][2]
        cv2.line(img, pt1, pt2, color=value, thickness=thickness * 2)

    # --- případná středová tečka ---
    if random.random() < dot_probability:
        dot_x = center[0] + random.randint(-dot_offset, dot_offset)
        dot_y = center[1] + random.randint(-dot_offset, dot_offset)
        cv2.circle(img, (dot_x, dot_y), radius=dot_radius, color=1.0, thickness=-1)

    # --- rozmazání ---
    for _ in range(blur_repeat):
        img = cv2.GaussianBlur(img, (blur_strength * 2 + 1, blur_strength * 2 + 1), 0)

    # --- normalizace ---
    img = np.clip(img, 0.0, 1.0)
    if np.max(img) > 0:
        img /= np.max(img)

    return img




import numpy as np
import cv2
import random

import numpy as np
import cv2
import random

def generate_deformed_grid(m_px, image_size=350, grid_rows=None, grid_cols=None, point_radius=7, offset_range=3,
                           blur_strength=6, blur_repeat=8, rotation_range=45, spacing_variation=5,
                           brightness_variation=80, variation_scale=40, missing_point_prob=0.15):
    """
    Generates a float32 image (0.0–1.0) of a deformed point grid with optional blur, rotation, and noise.
    """
    if random.random() > 0.3:
        offset_range = random.randint(1, 10)
        point_radius = random.randint(1,10)
        spacing_variation = random.randint(1, 20)

    if random.random() > 0.3:
        point_radius = point_radius - 1
    if random.random() > 0.3:
        point_radius = point_radius + 1
    # --- Always include noise in scale ---
    base_scale = 1.2 - 0.6 * m_px
    scale_noise = random.uniform(0.85, 1.15)  # ±10 %
    if random.random() > 0.3:  # 70% pravděpodobnost použití m_px
        scale_factor = max(0.1, min(1.0, base_scale)) * scale_noise
    else:
        scale_factor = 1.0 * scale_noise

    # --- Apply scale ---
    point_radius = max(1, int(point_radius * scale_factor))
    offset_range = int(offset_range * scale_factor)
    spacing_variation = int(spacing_variation * scale_factor)
    brightness_variation = brightness_variation * scale_factor
    variation_scale = max(5, int(variation_scale * (1 / scale_factor)))

    # --- Grid size ---
    if grid_rows is None:
        grid_rows = random.randint(2, 5)
    if grid_cols is None:
        grid_cols = random.randint(2, 5)

    # --- Init images ---
    img = np.zeros((image_size, image_size), dtype=np.float32)
    temp = np.zeros_like(img, dtype=np.uint8)

    # --- Spacing ---
    x_spacings = [image_size // (grid_cols + 1) + random.randint(-spacing_variation, spacing_variation)
                  for _ in range(grid_cols)]
    y_spacings = [image_size // (grid_rows + 1) + random.randint(-spacing_variation, spacing_variation)
                  for _ in range(grid_rows)]

    # --- Grid points ---
    points = []
    y_position = y_spacings[0]
    for row in range(grid_rows):
        x_position = x_spacings[0]
        for col in range(grid_cols):
            if random.random() > missing_point_prob:
                x = x_position + random.randint(-offset_range, offset_range)
                y = y_position + random.randint(-offset_range, offset_range)
                points.append((x, y))
            x_position += x_spacings[col]
        y_position += y_spacings[row]

    # --- Rotate points ---
    points = np.array(points, dtype=np.float32)
    center = (image_size // 2, image_size // 2)
    angle = random.uniform(-rotation_range, rotation_range)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack([points, ones])
    rotated_points = rot_mat.dot(points_homogeneous.T).T

    # --- Draw points ---
    for x, y in rotated_points.astype(int):
        cv2.circle(temp, (x, y), radius=point_radius, color=255, thickness=-1)

    # --- Normalize mask ---
    mask = temp.astype(np.float32) / 255.0
    if blur_strength > 0:
        for _ in range(blur_repeat):
            mask = cv2.GaussianBlur(mask, (blur_strength * 2 + 1, blur_strength * 2 + 1), 0)
    if np.max(mask) > 0:
        mask /= np.max(mask)

    # --- Apply local noise ---
    noise_h = max(1, image_size // variation_scale)
    noise_small = np.random.normal(0, brightness_variation / 255.0, (noise_h, noise_h))
    noise = cv2.resize(noise_small, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    img = mask + noise * mask
    img = np.clip(img, 0.0, 1.0)

    # --- Náhodné celkové zmenšení výsledného objektu v 50 % případů ---
    if random.random() > 0.5:
        resize_factor = 1.0 / random.uniform(1.1, 4.0)
        new_size = max(1, int(image_size * resize_factor))
        img = cv2.resize(img, (new_size, new_size), interpolation=cv2.INTER_AREA)

        # Umístění zmenšeného obrázku doprostřed
        canvas = np.zeros((image_size, image_size), dtype=np.float32)
        y_offset = (image_size - new_size) // 2
        x_offset = (image_size - new_size) // 2
        canvas[y_offset:y_offset + new_size, x_offset:x_offset + new_size] = img
        img = canvas


    return img







def generate_deformed_trapezoid(m_px, image_size=350, base_size=150, top_size_ratio_range=(0.4, 1.4), height_range=(80, 200),
                                deformation=8, waviness=3, num_points=16, thickness=2,
                                skew=10, blur_strength=5, blur_repeat=7, missing_sides_prob=0.05, rotation_range=90):
    """
    Generates a float32 image (0.0–1.0) of a distorted trapezoid with optional open sides and blur.
    """

    # --- m_px škálování ---
    use_scale = random.random() > 0.3
    scale_factor = 1.0
    if use_scale:
        scale_factor = max(0.1, min(1.0, 1.2 - 0.6 * m_px))

    base_size = int(base_size * scale_factor)
    deformation = int(deformation * scale_factor)
    waviness = int(max(1, waviness * scale_factor))
    thickness = max(1, int(thickness * scale_factor))
    height_min = int(height_range[0] * scale_factor)
    height_max = int(height_range[1] * scale_factor)

    # --- Inicializace ---
    img = np.zeros((image_size, image_size), dtype=np.float32)
    center_x, center_y = image_size // 2, image_size // 2

    # --- Parametry tvaru ---
    top_size_ratio = random.uniform(*top_size_ratio_range)
    height = random.randint(height_min, height_max)
    base_half = base_size // 2
    top_half = int(base_half * top_size_ratio)

    # --- Rohy lichoběžníku ---
    corners = np.array([
        [center_x - base_half + random.randint(-deformation, deformation),
         center_y + height // 2 + random.randint(-deformation, deformation)],
        [center_x + base_half + random.randint(-deformation, deformation),
         center_y + height // 2 + random.randint(-deformation, deformation)],
        [center_x + top_half + random.randint(-deformation, deformation),
         center_y - height // 2 + random.randint(-deformation, deformation)],
        [center_x - top_half + random.randint(-deformation, deformation),
         center_y - height // 2 + random.randint(-deformation, deformation)]
    ], dtype=np.float32)

    # --- Rotace ---
    angle = random.uniform(-rotation_range, rotation_range)
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    corners = np.dot(rotation_matrix[:, :2], corners.T).T + rotation_matrix[:, 2]

    # --- Náhodně chybějící strany ---
    missing_sides = set()
    if random.random() < missing_sides_prob:
        missing_sides.add(random.randint(0, 3))
    if random.random() < missing_sides_prob and len(missing_sides) < 2:
        missing_sides.add(random.randint(0, 3))

    # --- Vlnění ---
    def add_waviness(p1, p2, num_points, waviness):
        points = []
        for i in range(num_points + 1):
            t = i / num_points
            x = int((1 - t) * p1[0] + t * p2[0] + random.randint(-waviness, waviness))
            y = int((1 - t) * p1[1] + t * p2[1] + random.randint(-waviness, waviness))
            points.append((x, y))
        return points

    # --- Vykreslení hran ---
    temp = np.zeros((image_size, image_size), dtype=np.uint8)
    for i in range(4):
        if i not in missing_sides:
            edge = add_waviness(corners[i], corners[(i + 1) % 4], num_points, waviness)
            edge = np.array([edge], dtype=np.int32)
            cv2.polylines(temp, edge, isClosed=False, color=255, thickness=thickness * 2)

    # --- Rozmazání ---
    if blur_strength > 0:
        for _ in range(blur_repeat):
            temp = cv2.GaussianBlur(temp, (blur_strength * 2 + 1, blur_strength * 2 + 1), 0)

    # --- Normalizace výstupu ---
    img = temp.astype(np.float32) / 255.0
    return img



def generate_deformed_faded_trapezoid(m_px, image_size=350, base_size=150, top_size_ratio_range=(0.4, 1.4), height_range=(80, 200),
                                      deformation=8, waviness=2, num_points=16, thickness=2,
                                      skew=10, blur_strength=6, blur_repeat=8, missing_sides_prob=0.15, rotation_range=90,
                                      intensity_variation=500, fade_factor=10):
    """
    Generates a float32 image (0.0–1.0) of a trapezoid with hand-drawn effect, fading edges, blur and rotation.
    """
    if random.random() > 0.5:
        fade_factor = random.uniform(0.2, 5)
        intensity_variation = random.randint(1, 500)
    elif random.random() > 0.6:
        fade_factor = random.randint(5, 10)
        intensity_variation = random.randint(1, 500)

    # --- škálování ---
    base_scale = 1.2 - 0.6 * m_px
    scale_noise = random.uniform(0.9, 1.1)
    scale_factor = (base_scale if random.random() > 0.3 else 1.0) * scale_noise

    base_size = int(base_size * scale_factor)
    deformation = int(deformation * scale_factor)
    waviness = int(max(1, waviness * scale_factor))
    thickness = max(1, int(thickness * scale_factor))
    height_min = int(height_range[0] * scale_factor)
    height_max = int(height_range[1] * scale_factor)

    # --- blur variabilita ---
    if random.random() > 0.5:
        blur_strength = max(1, int(blur_strength * random.uniform(0.5, 1.5)))
    else:
        blur_strength = max(1, int(blur_strength * scale_factor * random.uniform(0.8, 1.2)))

    blur_repeat = max(1, int(blur_repeat * scale_factor * random.uniform(0.7, 1.4)))
    if random.random() > 0.8:
        blur_repeat = random.randint(1, 10)

    # --- inicializace ---
    img = np.zeros((image_size, image_size), dtype=np.float32)
    center_x, center_y = image_size // 2, image_size // 2

    top_size_ratio = random.uniform(*top_size_ratio_range)
    height = random.randint(height_min, height_max)
    base_half = base_size // 2
    top_half = int(base_half * top_size_ratio)

    corners = np.array([
        [center_x - base_half + random.randint(-deformation, deformation),
         center_y + height // 2 + random.randint(-deformation, deformation)],
        [center_x + base_half + random.randint(-deformation, deformation),
         center_y + height // 2 + random.randint(-deformation, deformation)],
        [center_x + top_half + random.randint(-deformation, deformation),
         center_y - height // 2 + random.randint(-deformation, deformation)],
        [center_x - top_half + random.randint(-deformation, deformation),
         center_y - height // 2 + random.randint(-deformation, deformation)]
    ], dtype=np.float32)

    # --- rotace ---
    angle = random.uniform(-rotation_range, rotation_range)
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    corners = np.dot(rotation_matrix[:, :2], corners.T).T + rotation_matrix[:, 2]

    # --- chybějící strany ---
    missing_sides = set()
    if random.random() < missing_sides_prob:
        missing_sides.add(random.randint(0, 3))
    if random.random() < missing_sides_prob and len(missing_sides) < 2:
        missing_sides.add(random.randint(0, 3))

    # --- fade efekt ---
    def add_waviness(p1, p2, num_points, waviness, intensity_variation, fade_factor):
        points = []
        base_intensity = max(0, 255 - intensity_variation // 2)
        intensities = np.clip(np.random.randint(base_intensity, base_intensity + intensity_variation, num_points + 1), 0, 255)
        for i in range(num_points + 1):
            t_rel = i / num_points
            fade = (1 - np.cos(t_rel * np.pi * fade_factor + random.uniform(-0.3, 0.3))) / 2
            x = int((1 - t_rel) * p1[0] + t_rel * p2[0] + random.randint(-waviness, waviness))
            y = int((1 - t_rel) * p1[1] + t_rel * p2[1] + random.randint(-waviness, waviness))
            intensity = (intensities[i] / 255.0) * fade
            points.append((x, y, intensity))
        return points

    # --- vykreslení ---
    for i in range(4):
        if i in missing_sides:
            continue
        edge = add_waviness(corners[i], corners[(i + 1) % 4], num_points, waviness, intensity_variation, fade_factor)
        for j in range(len(edge) - 1):
            pt1 = (edge[j][0], edge[j][1])
            pt2 = (edge[j + 1][0], edge[j + 1][1])
            value = edge[j][2]
            cv2.line(img, pt1, pt2, color=value, thickness=thickness * 2)

    # --- blur ---
    for _ in range(blur_repeat):
        img = cv2.GaussianBlur(img, (blur_strength * 2 + 1, blur_strength * 2 + 1), 0)

    # --- normalizace ---
    if np.max(img) > 0:
        img /= np.max(img)

    return img



def generate_deformed_ellipse(m_px, image_size=350, radius=100, ellipse_ratio_range=(0.85, 1.15),
                              deformation=8, waviness=2, num_points=50, thickness=2,
                              blur_strength=6, blur_repeat=5, rotation_range=90,
                              intensity_variation=1500, fade_factor=1.0):
    """
    Generates a float32 image (0.0–1.0) of a hand-drawn-like ellipse with intensity variation, blur and rotation.
    """
    if random.random() > 0.4:
        if random.random() > 0.5:
            fade_factor = random.uniform(0.2, 5)
            intensity_variation = random.randint(1, 1500)
        elif random.random() > 0.6:
            fade_factor = random.randint(5, 10)
            intensity_variation = random.randint(1, 500)


    if random.random() > 0.3:
        thickness = thickness - 1

    radius = radius * random.uniform(0.6, 1.1)

    # --- škálování ---
    base_scale = 1.2 - 0.6 * m_px
    if random.random() > 0.5:
        base_scale = base_scale * 0.5
    blur_repeat = max(int(blur_repeat * base_scale), 1)
    if random.random() > 0.3:
        blur_strength = max(int(blur_strength * base_scale), 1)
    scale_noise = random.uniform(0.9, 1.1)
    if random.random() > 0.3:
        scale_factor = max(0.1, min(1.0, base_scale)) * scale_noise
    else:
        scale_factor = 1.0 * scale_noise

    radius = int(radius * scale_factor)
    waviness = int(max(1, waviness * scale_factor))
    thickness = max(1, int(thickness * scale_factor))
    deformation = int(deformation * scale_factor)  # rezervováno pro budoucí použití

    # --- inicializace ---
    img = np.zeros((image_size, image_size), dtype=np.float32)
    center_x, center_y = image_size // 2, image_size // 2
    ellipse_ratio = random.uniform(*ellipse_ratio_range)

    # --- generování hrany ---
    def add_waviness(center, radius, num_points, waviness, intensity_variation, fade_factor, ellipse_ratio):
        points = []
        base_intensity = 255 - intensity_variation // 2
        intensities = np.clip(
            np.random.randint(base_intensity, base_intensity + intensity_variation, num_points + 1), 0, 255
        ) / 255.0

        for i in range(num_points + 1):
            angle = (i / num_points) * 2 * np.pi
            r_offset = random.randint(-waviness, waviness)
            x = int(center[0] + (radius + r_offset) * np.cos(angle) * ellipse_ratio)
            y = int(center[1] + (radius + r_offset) * np.sin(angle))
            t = (1 - np.cos(i / num_points * np.pi * fade_factor + random.uniform(-0.5, 0.5))) / 2
            points.append((x, y, intensities[i]))
        return points

    wavy_edge = add_waviness((center_x, center_y), radius, num_points, waviness,
                             intensity_variation, fade_factor, ellipse_ratio)

    # --- rotace ---
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), random.uniform(-rotation_range, rotation_range), 1.0)
    rotated_points = np.array([rotation_matrix @ [x, y, 1] for x, y, _ in wavy_edge], dtype=int)

    # --- vykreslení čar ---
    for j in range(len(rotated_points) - 1):
        pt1 = (rotated_points[j][0], rotated_points[j][1])
        pt2 = (rotated_points[j + 1][0], rotated_points[j + 1][1])
        value = wavy_edge[j][2]
        cv2.line(img, pt1, pt2, color=value, thickness=thickness * 2)

    # --- rozmazání ---
    if blur_strength > 0:
        for _ in range(blur_repeat):
            img = cv2.GaussianBlur(img, (blur_strength * 2 + 1, blur_strength * 2 + 1), 0)

    # --- normalizace výsledku ---
    if np.max(img) > 0:
        img /= np.max(img)

    return img



def generate_deformed_rectangle(m_px, image_size=350, width_range=(50, 180), height_range=(80, 200),
                                deformation=8, waviness=2, num_points=16, thickness=2,
                                blur_strength=6, blur_repeat=6, rotation_range=90,
                                intensity_variation=400, fade_factor=1.0):
    """
    Generates a float32 image (0.0–1.0) of a deformed rectangle with fading edges, blur and rotation.
    """
    if random.random() > 0.4:
        if random.random() > 0.5:
            fade_factor = random.uniform(0.2, 5)
            intensity_variation = random.randint(1, 1500)
        elif random.random() > 0.6:
            fade_factor = random.randint(5, 10)
            intensity_variation = random.randint(1, 500)



    # --- škálování ---
    base_scale = 1.3 - 0.7 * m_px
    if random.random() > 0.6:
        base_scale = base_scale * max(random.random(), 0.3)
    scale_noise = random.uniform(0.9, 1.1)
    if random.random() > 0.3:
        scale_factor = max(0.1, min(1.0, base_scale)) * scale_noise
    else:
        scale_factor = 1.0 * scale_noise

    if random.random() > 0.5:
        blur_strength = max(1, int(blur_strength * scale_factor))
    blur_repeat = max(1, int(blur_repeat * scale_factor))

    deformation = int(deformation * scale_factor)
    waviness = int(max(1, waviness * scale_factor))
    thickness = max(1, int(thickness * scale_factor))

    width = random.randint(*width_range)
    height = random.randint(*height_range)
    width = int(width * scale_factor)
    height = int(height * scale_factor)

    center_x, center_y = image_size // 2, image_size // 2
    half_w, half_h = width // 2, height // 2

    # --- Rohy s deformací ---
    corners = np.array([
        [center_x - half_w + random.randint(-deformation, deformation),
         center_y - half_h + random.randint(-deformation, deformation)],
        [center_x + half_w + random.randint(-deformation, deformation),
         center_y - half_h + random.randint(-deformation, deformation)],
        [center_x + half_w + random.randint(-deformation, deformation),
         center_y + half_h + random.randint(-deformation, deformation)],
        [center_x - half_w + random.randint(-deformation, deformation),
         center_y + half_h + random.randint(-deformation, deformation)]
    ], dtype=np.float32)

    # --- Rotace ---
    angle = random.uniform(-rotation_range, rotation_range)
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    corners = np.dot(rotation_matrix[:, :2], corners.T).T + rotation_matrix[:, 2]

    # --- Fade okrajů ---
    def add_waviness(p1, p2, num_points, waviness, intensity_variation, fade_factor):
        points = []
        base_intensity = max(0, 255 - intensity_variation // 2)
        intensities = np.clip(
            np.random.randint(base_intensity, base_intensity + intensity_variation, num_points + 1),
            0, 255
        ) / 255.0  # float intenzity
        for i in range(num_points + 1):
            t = (1 - np.cos(i / num_points * np.pi * fade_factor + random.uniform(-0.5, 0.5))) / 2
            x = int((1 - t) * p1[0] + t * p2[0] + random.randint(-waviness, waviness))
            y = int((1 - t) * p1[1] + t * p2[1] + random.randint(-waviness, waviness))
            points.append((x, y, intensities[i]))
        return points

    img = np.zeros((image_size, image_size), dtype=np.float32)

    # --- Vykreslení ---
    for i in range(len(corners)):
        edge = add_waviness(corners[i], corners[(i + 1) % 4], num_points, waviness, intensity_variation, fade_factor)
        for j in range(len(edge) - 1):
            pt1 = (edge[j][0], edge[j][1])
            pt2 = (edge[j + 1][0], edge[j + 1][1])
            value = edge[j][2]
            cv2.line(img, pt1, pt2, color=value, thickness=thickness * 2)

    # --- Rozmazání ---
    if blur_strength > 0:
        for _ in range(blur_repeat):
            img = cv2.GaussianBlur(img, (blur_strength * 2 + 1, blur_strength * 2 + 1), 0)

    # --- Normalizace ---
    if np.max(img) > 0:
        img /= np.max(img)

    return img


def generate_deformed_rounded_rectangle(m_px, image_size=350, width_range=(80, 200), height_range=(80, 200),
                                deformation=1, waviness=2, num_points=16, thickness=2,
                                blur_strength=6, blur_repeat=7, rotation_range=90,
                                intensity_variation=1000, fade_factor=1.0, corner_rounding=16):
    """
    Generates a float32 image (0.0–1.0) of a rounded, deformed rectangle with fading edges and blur.
    """


    if random.random() > 0.3:
        thickness = thickness - 1
    if random.random() > 0.3:
        intensity_variation = intensity_variation * random.uniform(0.1, 10)
    else:
        intensity_variation = 100 * random.uniform(0.1, 3)
        intensity_variation = intensity_variation * random.uniform(0.1, 3)
        fade_factor = random.uniform(0.2, 5)

    # --- škálování ---
    base_scale = 1.2 - 0.6 * m_px
    if random.random() > 0.6:
        base_scale = base_scale * max(random.random(), 0.5)
    scale_noise = random.uniform(0.9, 1.1)
    if random.random() > 0.3:
        scale_factor = max(0.15, min(1.0, base_scale)) * scale_noise
    else:
        scale_factor = 1.0 * scale_noise
    # print(scale_factor)

    if random.random() > 0.3:
        blur_strength = max(1, int(blur_strength * scale_factor))
    blur_repeat = max(1, int(blur_repeat * scale_factor))

    if random.random() > 0.6:
        blur_repeat = max(1, int(blur_repeat/2))

    deformation = int(deformation * scale_factor)
    waviness = int(max(1, waviness * scale_factor))
    thickness = max(1, int(thickness * scale_factor))
    corner_rounding = max(1, int(corner_rounding * scale_factor))

    width = int(random.randint(*width_range) * scale_factor)
    height = int(random.randint(*height_range) * scale_factor)

    # --- Inicializace ---
    img = np.zeros((image_size, image_size), dtype=np.float32)
    center_x, center_y = image_size // 2, image_size // 2
    half_w, half_h = width // 2, height // 2

    # --- Náhodné zaoblení rohů ---
    corner_radii = [random.randint(0, corner_rounding) for _ in range(4)]

    # --- Rohy s deformací a roundingem ---
    corners = np.array([
        [center_x - half_w + random.randint(-deformation, deformation) + corner_radii[0],
         center_y - half_h + random.randint(-deformation, deformation) + corner_radii[0]],
        [center_x + half_w + random.randint(-deformation, deformation) - corner_radii[1],
         center_y - half_h + random.randint(-deformation, deformation) + corner_radii[1]],
        [center_x + half_w + random.randint(-deformation, deformation) - corner_radii[2],
         center_y + half_h + random.randint(-deformation, deformation) - corner_radii[2]],
        [center_x - half_w + random.randint(-deformation, deformation) + corner_radii[3],
         center_y + half_h + random.randint(-deformation, deformation) - corner_radii[3]]
    ], dtype=np.float32)

    # --- Rotace ---
    angle = random.uniform(-rotation_range, rotation_range)
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    corners = np.dot(rotation_matrix[:, :2], corners.T).T + rotation_matrix[:, 2]

    # --- Funkce pro vlnění + fade ---
    def add_waviness(p1, p2, num_points, waviness, intensity_variation, fade_factor):
        points = []
        base_intensity = 255 - intensity_variation // 2
        intensities = np.clip(
            np.random.randint(base_intensity, base_intensity + intensity_variation, num_points + 1),
            0, 255
        ) / 255.0  # float

        for i in range(num_points + 1):
            t = (1 - np.cos(i / num_points * np.pi * fade_factor + random.uniform(-0.5, 0.5))) / 2
            x = int((1 - t) * p1[0] + t * p2[0] + random.randint(-waviness, waviness))
            y = int((1 - t) * p1[1] + t * p2[1] + random.randint(-waviness, waviness))
            points.append((x, y, intensities[i]))
        return points

    # --- Vykreslení ---
    for i in range(len(corners)):
        edge = add_waviness(corners[i], corners[(i + 1) % 4], num_points, waviness, intensity_variation, fade_factor)
        for j in range(len(edge) - 1):
            pt1 = (edge[j][0], edge[j][1])
            pt2 = (edge[j + 1][0], edge[j + 1][1])
            value = edge[j][2]
            cv2.line(img, pt1, pt2, color=value, thickness=thickness * 2)

    # --- Blur ---
    if blur_strength > 0:
        for _ in range(blur_repeat):
            img = cv2.GaussianBlur(img, (blur_strength * 2 + 1, blur_strength * 2 + 1), 0)

    # --- Normalizace ---
    if np.max(img) > 0:
        img /= np.max(img)

    return img



def generate_deformed_dotted_rectangle(m_px, image_size=350, width_range=(80, 200), height_range=(80, 200),
                              deformation=2, dot_radius=7, dot_spacing=40, dot_offset=2,
                              blur_strength=6, blur_repeat=7, rotation_range=90,
                              intensity_variation=1000, brightness_variation=0,
                              variation_scale=10, missing_corner_prob=0.2, missing_adjacent_prob=0.2,
                              missing_point_prob=0.35):
    """
    Generates a float32 image (0.0–1.0) of a dotted rectangle with missing points, blur, and noise.
    """
    if random.random() > 0.7 and m_px < 0.5:
        dot_radius = random.randint(3, 10)

    brightness_variation = random.uniform(0, 10)
    # --- škálování ---
    base_scale = 1.25 - 0.63 * m_px

    if m_px > 1.5:
        base_scale = 0.18 + random.uniform(0.0, 0.2)
        if random.random() > 0.6:
            base_scale = 0.18 + random.uniform(0.0, 0.6)


    scale_noise = random.uniform(0.9, 1.1)
    if random.random() > 0.3:
        scale_factor = max(0.1, min(1.0, base_scale)) * scale_noise
    else:
        scale_factor = 1.0 * scale_noise

    if random.random() > 0.3:
        blur_strength = max(1, int(blur_strength * scale_factor))
    blur_repeat = max(1, int(blur_repeat * scale_factor))

    if random.random() > 0.7:
        blur_repeat = max(1, int(blur_repeat/2))

    if random.random() > 0.8:
        dot_offset = 3

    deformation = int(deformation * scale_factor)
    dot_radius = max(1, int(dot_radius * scale_factor))
    dot_spacing = max(5, int(dot_spacing * scale_factor))
    if random.random() > 0.8:
        dot_spacing = int(dot_spacing * random.uniform(0.5, 2))
    dot_offset = int(dot_offset * scale_factor)
    brightness_variation *= scale_factor
    variation_scale = max(5, int(variation_scale * (1 / scale_factor)))

    if m_px > 1.5 and random.random() > 0.2:
        if blur_repeat > 2 and random.random() > 0.2:
            blur_repeat = random.randint(1, 2)

    width = int(random.randint(*width_range) * scale_factor)
    height = int(random.randint(*height_range) * scale_factor)

    # --- Inicializace ---
    img = np.zeros((image_size, image_size), dtype=np.float32)
    mask = np.zeros_like(img, dtype=np.float32)
    center_x, center_y = image_size // 2, image_size // 2
    half_w, half_h = width // 2, height // 2

    # --- Rohy s deformací ---
    corners = np.array([
        [center_x - half_w + random.randint(-deformation, deformation),
         center_y - half_h + random.randint(-deformation, deformation)],
        [center_x + half_w + random.randint(-deformation, deformation),
         center_y - half_h + random.randint(-deformation, deformation)],
        [center_x + half_w + random.randint(-deformation, deformation),
         center_y + half_h + random.randint(-deformation, deformation)],
        [center_x - half_w + random.randint(-deformation, deformation),
         center_y + half_h + random.randint(-deformation, deformation)]
    ], dtype=np.float32)

    # --- Rotace ---
    angle = random.uniform(-rotation_range, rotation_range)
    rot_mat = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    corners = np.dot(rot_mat[:, :2], corners.T).T + rot_mat[:, 2]

    # --- Chybějící rohy ---
    missing_corners = {i for i in range(4) if random.random() < missing_corner_prob}

    # --- Generování bodů ---
    def generate_dots(p1, p2, dot_spacing, dot_offset, missing_point_prob, missing_first, missing_last):
        dots = []
        dist = np.linalg.norm(np.array(p2) - np.array(p1))
        num_dots = max(1, int(dist // dot_spacing))
        for i in range(num_dots + 2):
            t = i / (num_dots + 1)
            x = int((1 - t) * p1[0] + t * p2[0] + random.randint(-dot_offset, dot_offset))
            y = int((1 - t) * p1[1] + t * p2[1] + random.randint(-dot_offset, dot_offset))
            if not ((missing_first and i == 0) or (missing_last and i == num_dots + 1)) and random.random() > missing_point_prob:
                dots.append((x, y))
        return dots

    dotted_points = []
    for i in range(4):
        missing_first = i in missing_corners and random.random() < missing_adjacent_prob
        missing_last = (i + 1) % 4 in missing_corners and random.random() < missing_adjacent_prob
        dotted_points.extend(generate_dots(corners[i], corners[(i + 1) % 4],
                                           dot_spacing, dot_offset, missing_point_prob, missing_first, missing_last))

    # --- Kreslení teček ---
    for x, y in dotted_points:
        cv2.circle(img, (x, y), dot_radius, color=1.0, thickness=-1)
        cv2.circle(mask, (x, y), dot_radius, color=1.0, thickness=-1)

    # --- Noise pouze v maskovaných místech ---
    noise_small = np.random.normal(0, brightness_variation / 255.0,
                                   (image_size // variation_scale, image_size // variation_scale))
    noise = cv2.resize(noise_small, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    img = img + noise * mask

    # --- Blur ---
    if blur_strength > 0:
        for _ in range(blur_repeat):
            img = cv2.GaussianBlur(img, (blur_strength * 2 + 1, blur_strength * 2 + 1), 0)

    # --- Normalizace ---
    img = np.clip(img, 0.0, 1.0)
    if np.max(img) > 0:
        img /= np.max(img)

    return img



# def generate_dotted_rectangle(m_px, image_size=350, width_range=(80, 200), height_range=(80, 200),
#                               deformation=2, dot_radius=2, dot_spacing_range=(30, 60), dot_offset=2,
#                               blur_strength=6, blur_repeat=6, rotation_range=90,
#                               intensity_variation=1000, brightness_variation=100,
#                               variation_scale=20, missing_corner_prob=0.5, missing_adjacent_prob=0.5,
#                               missing_point_prob=0.2):
#     """
#     Generates a float32 image (0.0–1.0) of a dotted rectangle with missing points, blur, and noise.
#     """
#
#     # --- škálování ---
#     base_scale = 1.25 - 0.63 * m_px
#     scale_noise = random.uniform(0.9, 1.1)
#     if random.random() > 0.3:
#         scale_factor = max(0.1, min(1.0, base_scale)) * scale_noise
#     else:
#         scale_factor = 1.0 * scale_noise
#
#     if random.random() > 0.3:
#         blur_strength = max(1, int(blur_strength * scale_factor))
#     blur_repeat = max(1, int(blur_repeat * scale_factor))
#
#     deformation = int(deformation * scale_factor)
#     dot_radius = max(1, int(dot_radius * scale_factor))
#     dot_offset = int(dot_offset * scale_factor)
#     brightness_variation *= scale_factor
#     variation_scale = max(5, int(variation_scale * (1 / scale_factor)))
#     dot_spacing_range = (max(5, int(dot_spacing_range[0] * scale_factor)),
#                          max(6, int(dot_spacing_range[1] * scale_factor)))
#
#     width = int(random.randint(*width_range) * scale_factor)
#     height = int(random.randint(*height_range) * scale_factor)
#
#     # --- Inicializace ---
#     img = np.zeros((image_size, image_size), dtype=np.float32)
#     mask = np.zeros_like(img, dtype=np.float32)
#     center_x, center_y = image_size // 2, image_size // 2
#     half_w, half_h = width // 2, height // 2
#
#     # --- Rohy s deformací ---
#     corners = np.array([
#         [center_x - half_w + random.randint(-deformation, deformation),
#          center_y - half_h + random.randint(-deformation, deformation)],
#         [center_x + half_w + random.randint(-deformation, deformation),
#          center_y - half_h + random.randint(-deformation, deformation)],
#         [center_x + half_w + random.randint(-deformation, deformation),
#          center_y + half_h + random.randint(-deformation, deformation)],
#         [center_x - half_w + random.randint(-deformation, deformation),
#          center_y + half_h + random.randint(-deformation, deformation)]
#     ], dtype=np.float32)
#
#     # --- Rotace ---
#     angle = random.uniform(-rotation_range, rotation_range)
#     rot_mat = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
#     corners = np.dot(rot_mat[:, :2], corners.T).T + rot_mat[:, 2]
#
#     # --- Chybějící rohy ---
#     missing_corners = {i for i in range(4) if random.random() < missing_corner_prob}
#
#     # --- Generování bodů ---
#     def generate_dots(p1, p2, dot_spacing_range, dot_offset, missing_point_prob, missing_first, missing_last):
#         dots = []
#         dist = np.linalg.norm(np.array(p2) - np.array(p1))
#         spacing = random.randint(*dot_spacing_range)
#         num_dots = max(1, int(dist // spacing))
#         for i in range(num_dots + 2):
#             t = i / (num_dots + 1)
#             x = int((1 - t) * p1[0] + t * p2[0] + random.randint(-dot_offset, dot_offset))
#             y = int((1 - t) * p1[1] + t * p2[1] + random.randint(-dot_offset, dot_offset))
#             if not ((missing_first and i == 0) or (missing_last and i == num_dots + 1)) and random.random() > missing_point_prob:
#                 dots.append((x, y))
#         return dots
#
#     dotted_points = []
#     for i in range(4):
#         missing_first = i in missing_corners and random.random() < missing_adjacent_prob
#         missing_last = (i + 1) % 4 in missing_corners and random.random() < missing_adjacent_prob
#         dotted_points.extend(generate_dots(corners[i], corners[(i + 1) % 4],
#                                            dot_spacing_range, dot_offset, missing_point_prob,
#                                            missing_first, missing_last))
#
#     # --- Vykreslení teček ---
#     for x, y in dotted_points:
#         cv2.circle(img, (x, y), dot_radius, color=1.0, thickness=-1)
#         cv2.circle(mask, (x, y), dot_radius, color=1.0, thickness=-1)
#
#     # --- Noise pouze v masce ---
#     noise_small = np.random.normal(0, brightness_variation / 255.0,
#                                    (image_size // variation_scale, image_size // variation_scale))
#     noise = cv2.resize(noise_small, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
#     img += noise * mask
#
#     # --- Blur ---
#     if blur_strength > 0:
#         for _ in range(blur_repeat):
#             img = cv2.GaussianBlur(img, (blur_strength * 2 + 1, blur_strength * 2 + 1), 0)
#
#     # --- Normalizace ---
#     img = np.clip(img, 0.0, 1.0)
#     if np.max(img) > 0:
#         img /= np.max(img)
#
#     return img


def generate_deformed_filled_patch(m_px, image_size=350, width_range=(60, 140), height_range=(60, 140),
                                   deformation=0, waviness=2, num_points=20, blur_strength=5,
                                   blur_repeat=8, rotation_range=180, intensity_variation=1000,
                                   corner_rounding=12, bulge_factor=18,
                                   brightness_variation=70, variation_scale=20):
    """
    Generates a float32 image (0.0–1.0) of a filled, hand-drawn-like patch with bulging edges and noise.
    """

    if m_px > 1.1:
        print("generate_deformed_filled_patch Does not support this m_px value")
        return None
    if random.random() > 0.8:
        width_range = (90, 180)
        height_range = (90, 180)
    elif random.random() > 0.7 and m_px < 0.5:
        width_range = (40, 120)
        height_range = (40, 120)
    # --- úprava range velikostí podle m_px ---
    size_scale = 1.2 - 0.6 * m_px
    size_scale = np.clip(size_scale, 0.4, 1.0)
    width_range_scaled = (int(width_range[0] * size_scale), int(width_range[1] * size_scale))
    height_range_scaled = (int(height_range[0] * size_scale), int(height_range[1] * size_scale))

    # --- škálování ---
    base_scale = (1.4 - 0.8 * m_px) / 2
    scale_noise = random.uniform(0.9, 1.1)
    scale_factor = max(0.1, min(1.0, base_scale)) * scale_noise if random.random() > 0.3 else 1.0 * scale_noise

    if random.random() > 0.3:
        blur_strength = max(1, int(blur_strength * scale_factor))
    blur_repeat = max(1, int(blur_repeat * scale_factor))
    if random.random() > 0.7:
        blur_repeat = max(1, blur_repeat // 2)

    if random.random() > 0.35:
        bulge_factor = random.randint(0, 20)
    if random.random() > 0.65:
        blur_repeat = random.randint(1, 4)

    deformation = int(deformation * scale_factor)
    waviness = max(1, int(waviness * scale_factor))
    corner_rounding = max(1, int(corner_rounding * scale_factor))
    bulge_factor = int(bulge_factor * scale_factor)
    brightness_variation *= scale_factor
    variation_scale = max(5, int(variation_scale * (1 / scale_factor)))

    width = int(random.randint(*width_range_scaled) * scale_factor)
    height = int(random.randint(*height_range_scaled) * scale_factor)

    # --- Inicializace ---
    img = np.zeros((image_size, image_size), dtype=np.float32)
    mask = np.zeros_like(img, dtype=np.float32)
    center_x, center_y = image_size // 2, image_size // 2
    half_w, half_h = width // 2, height // 2

    # --- Rohy s náhodným zaoblením ---
    corner_radii = [random.randint(0, corner_rounding) for _ in range(4)]
    corners = np.array([
        [center_x - half_w + corner_radii[0], center_y - half_h + corner_radii[0]],
        [center_x + half_w - corner_radii[1], center_y - half_h + corner_radii[1]],
        [center_x + half_w - corner_radii[2], center_y + half_h - corner_radii[2]],
        [center_x - half_w + corner_radii[3], center_y + half_h - corner_radii[3]]
    ], dtype=np.float32)

    # --- Rotace ---
    angle = random.uniform(-rotation_range, rotation_range)
    rot_mat = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    corners = np.dot(rot_mat[:, :2], corners.T).T + rot_mat[:, 2]

    # --- Vlnící se hrany s vyboulením ---
    def add_waviness(p1, p2, num_points, waviness, bulge_factor):
        points = []
        midpoint_x = (p1[0] + p2[0]) / 2
        midpoint_y = (p1[1] + p2[1]) / 2
        bulge_x = midpoint_x + bulge_factor * ((p2[1] - p1[1]) / (height + 1e-5))
        bulge_y = midpoint_y + bulge_factor * ((p1[0] - p2[0]) / (width + 1e-5))

        for i in range(num_points + 1):
            t = i / num_points
            bx = (1 - t)**2 * p1[0] + 2 * (1 - t) * t * bulge_x + t**2 * p2[0]
            by = (1 - t)**2 * p1[1] + 2 * (1 - t) * t * bulge_y + t**2 * p2[1]
            x = int(bx + random.randint(-waviness, waviness))
            y = int(by + random.randint(-waviness, waviness))
            points.append((x, y))
        return points

    edge_points = []
    for i in range(4):
        edge_points.extend(add_waviness(corners[i], corners[(i + 1) % 4],
                                        num_points, waviness, bulge_factor))

    poly = np.array([edge_points], dtype=np.int32)
    cv2.fillPoly(img, poly, color=1.0)
    cv2.fillPoly(mask, poly, color=1.0)

    # --- Noise uvnitř objektu ---
    noise_small = np.random.normal(0, brightness_variation / 255.0,
                                   (image_size // variation_scale, image_size // variation_scale))
    noise = cv2.resize(noise_small, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    img += noise * mask

    # --- Blur ---
    if blur_strength > 0:
        for _ in range(blur_repeat):
            img = cv2.GaussianBlur(img, (blur_strength * 2 + 1, blur_strength * 2 + 1), 0)

    # --- Normalizace ---
    img = np.clip(img, 0.0, 1.0)
    if np.max(img) > 0:
        img /= np.max(img)

    return img




def generate_deformed_filled_patch_multi(m_px, image_size=350, width_range=(40, 100), height_range=(40, 100),
                                         deformation=0, waviness=2, num_points=20, blur_strength=5,
                                         blur_repeat=8, rotation_range=180, intensity_variation=1000,
                                         corner_rounding=12, bulge_factor=18,
                                         brightness_variation=70, variation_scale=20,
                                         poisson_lambda=9):
    """
    Generates a float32 image (0.0–1.0) with multiple non-overlapping filled patches near the center.
    """
    if random.random() > 0.8 and m_px < 0.5:
        poisson_lambda = 4

    if m_px > 1.1:
        print("generate_deformed_filled_patch_multi Does not support this m_px value")
        return None

    n_patches = 1 + np.random.poisson(lam=poisson_lambda)

    if random.random() > 0.9:
        width_range = (90, 180)
        height_range = (90, 180)

    # --- úprava range velikostí podle m_px ---
    size_scale = 1.2 - 0.6 * m_px
    size_scale = np.clip(size_scale, 0.4, 1.0)
    width_range_scaled = (int(width_range[0] * size_scale), int(width_range[1] * size_scale))
    height_range_scaled = (int(height_range[0] * size_scale), int(height_range[1] * size_scale))

    # --- škálování ---
    base_scale = (1.4 - 0.8 * m_px) / 2
    scale_noise = random.uniform(0.9, 1.1)
    scale_factor = max(0.1, min(1.0, base_scale)) * scale_noise if random.random() > 0.3 else 1.0 * scale_noise

    if random.random() > 0.3:
        blur_strength = max(1, int(blur_strength * scale_factor))
    blur_repeat = max(1, int(blur_repeat * scale_factor))
    if random.random() > 0.7:
        blur_repeat = max(1, blur_repeat // 2)
    if random.random() > 0.35:
        bulge_factor = random.randint(0, 20)
    if random.random() > 0.65:
        blur_repeat = random.randint(1, 4)

    deformation = int(deformation * scale_factor)
    waviness = max(1, int(waviness * scale_factor))
    corner_rounding = max(1, int(corner_rounding * scale_factor))
    bulge_factor = int(bulge_factor * scale_factor)
    brightness_variation *= scale_factor
    variation_scale = max(5, int(variation_scale * (1 / scale_factor)))

    img = np.zeros((image_size, image_size), dtype=np.float32)
    mask_total = np.zeros_like(img, dtype=np.float32)

    def generate_single_patch(cx, cy, width, height):
        corner_radii = [random.randint(0, corner_rounding) for _ in range(4)]
        half_w, half_h = width // 2, height // 2
        corners = np.array([
            [cx - half_w + corner_radii[0], cy - half_h + corner_radii[0]],
            [cx + half_w - corner_radii[1], cy - half_h + corner_radii[1]],
            [cx + half_w - corner_radii[2], cy + half_h - corner_radii[2]],
            [cx - half_w + corner_radii[3], cy + half_h - corner_radii[3]]
        ], dtype=np.float32)

        angle = random.uniform(-rotation_range, rotation_range)
        rot_mat = cv2.getRotationMatrix2D((float(cx), float(cy)), angle, 1.0)
        corners = np.dot(rot_mat[:, :2], corners.T).T + rot_mat[:, 2]

        def add_waviness(p1, p2):
            points = []
            midpoint_x = (p1[0] + p2[0]) / 2
            midpoint_y = (p1[1] + p2[1]) / 2
            bulge_x = midpoint_x + bulge_factor * ((p2[1] - p1[1]) / (height + 1e-5))
            bulge_y = midpoint_y + bulge_factor * ((p1[0] - p2[0]) / (width + 1e-5))

            for i in range(num_points + 1):
                t = i / num_points
                bx = (1 - t)**2 * p1[0] + 2 * (1 - t) * t * bulge_x + t**2 * p2[0]
                by = (1 - t)**2 * p1[1] + 2 * (1 - t) * t * bulge_y + t**2 * p2[1]
                x = int(bx + random.randint(-waviness, waviness))
                y = int(by + random.randint(-waviness, waviness))
                points.append((x, y))
            return points

        edge_points = []
        for i in range(4):
            edge_points.extend(add_waviness(corners[i], corners[(i + 1) % 4]))
        return np.array([edge_points], dtype=np.int32)

    attempts = 0
    max_attempts = n_patches * 10
    while n_patches > 0 and attempts < max_attempts:
        attempts += 1
        width = int(random.randint(*width_range_scaled) * scale_factor)
        height = int(random.randint(*height_range_scaled) * scale_factor)
        spread = image_size // 6
        cx = int(image_size // 2 + np.random.normal(0, spread))
        cy = int(image_size // 2 + np.random.normal(0, spread))
        cx = np.clip(cx, 0, image_size - 1)
        cy = np.clip(cy, 0, image_size - 1)

        patch_poly = generate_single_patch(cx, cy, width, height)
        temp_mask = np.zeros_like(mask_total, dtype=np.uint8)
        cv2.fillPoly(temp_mask, patch_poly, color=255)

        if np.any(cv2.bitwise_and(mask_total.astype(np.uint8), temp_mask)):
            continue

        cv2.fillPoly(mask_total, patch_poly, color=1.0)
        cv2.fillPoly(img, patch_poly, color=1.0)
        n_patches -= 1

    # --- Noise ---
    noise_small = np.random.normal(0, brightness_variation / 255.0,
                                   (image_size // variation_scale, image_size // variation_scale))
    noise = cv2.resize(noise_small, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    img += noise * mask_total

    # --- Blur ---
    if blur_strength > 0:
        for _ in range(blur_repeat):
            img = cv2.GaussianBlur(img, (blur_strength * 2 + 1, blur_strength * 2 + 1), 0)

    # --- Normalizace ---
    img = np.clip(img, 0.0, 1.0)
    if np.max(img) > 0:
        img /= np.max(img)

    return img



def generate_deformed_arc(m_px, image_size=350, deformation=2, waviness=1, num_points=42,
                          blur_strength=10, blur_repeat=2, intensity_variation=1000,
                          brightness_variation=100, variation_scale=20, thickness=2,
                          rotation_range=180, flattening_range=(0.85, 1.15)):
    """
    Generates a float32 image (0.0–1.0) with a hand-drawn arc that spans across the image,
    with blur, rotation, and optional intensity noise.
    """
    if random.random() > 0.7:
        waviness = 2
    elif random.random() > 0.7:
        waviness = 3
    # --- základní init ---
    img = np.zeros((image_size, image_size), dtype=np.float32)
    mask = np.zeros_like(img)

    # --- škálování ---
    base_scale = 1.25 - 0.63 * m_px
    scale_noise = random.uniform(0.9, 1.1)
    scale_factor = (base_scale if random.random() > 0.3 else 1.0) * scale_noise
    blur_strength = max(1, int(blur_strength * scale_factor)) if random.random() > 0.3 else blur_strength
    blur_repeat = max(1, int(blur_repeat * scale_factor))
    if random.random() > 0.7:
        blur_repeat = max(1, blur_repeat // 2)
    elif random.random() > 0.8:
        blur_repeat = blur_repeat + 1


    brightness_variation *= scale_factor
    variation_scale = max(5, int(variation_scale * (1 / scale_factor)))
    waviness = max(1, int(waviness * scale_factor))
    thickness = max(1, int(thickness * scale_factor))

    # --- geometrie oblouku ---
    # fixní velký radius přes obraz + mírná náhodná složka
    base_radius = image_size * random.uniform(0.8, 1.2)
    radius = int(base_radius / m_px)  # měřítkově závislý
    radius = int(np.clip(radius, image_size * 0.6, image_size * 1.8))

    # --- úhlová délka oblouku závislá na m_px ---
    # max_angle_deg = int(np.clip(180 - m_px * 80, 60, 160))  # např. při m_px=1.5 jen ~60°
    max_angle_deg = int(np.clip(180 - m_px * 60, 100, 175))

    arc_angle_deg = random.uniform(max_angle_deg * 0.8, max_angle_deg)

    start_angle = random.uniform(0, 360 - arc_angle_deg)
    end_angle = start_angle + arc_angle_deg
    flattening_factor = random.uniform(*flattening_range)

    # --- výpočet středu tak, aby oblouk byl přes střed obrázku ---
    mid_angle = np.radians((start_angle + end_angle) / 2)
    center_x = image_size // 2 - int(radius * np.cos(mid_angle))
    center_y = image_size // 2 - int(radius * np.sin(mid_angle) * flattening_factor)

    # --- rotace celého tvaru ---
    rotation_angle = random.uniform(-rotation_range, rotation_range)
    rot_mat = cv2.getRotationMatrix2D((float(image_size // 2), float(image_size // 2)), rotation_angle, 1.0)

    # --- generování bodů oblouku ---
    def add_waviness(center, radius, start_angle, end_angle, num_points, waviness, flattening_factor):
        points = []
        for i in range(num_points + 1):
            angle = np.radians(start_angle + (end_angle - start_angle) * (i / num_points))
            r_offset = random.randint(-waviness, waviness)
            x = int(center[0] + (radius + r_offset) * np.cos(angle))
            y = int(center[1] + (radius + r_offset) * np.sin(angle) * flattening_factor)
            points.append([x, y])
        return np.array(points, dtype=np.float32)

    arc_pts = add_waviness((center_x, center_y), radius, start_angle, end_angle,
                           num_points, waviness, flattening_factor)

    # --- aplikace rotace ---
    ones = np.ones((arc_pts.shape[0], 1), dtype=np.float32)
    arc_homog = np.hstack([arc_pts, ones])
    rotated_pts = rot_mat @ arc_homog.T
    rotated_pts = rotated_pts[:2].T.astype(int)

    # --- vykreslení čáry ---
    for i in range(len(rotated_pts) - 1):
        pt1 = tuple(rotated_pts[i])
        pt2 = tuple(rotated_pts[i + 1])
        cv2.line(img, pt1, pt2, color=1.0, thickness=thickness * 2)
        cv2.line(mask, pt1, pt2, color=1.0, thickness=thickness * 2)

    # --- přidání šumu pouze do oblouku ---
    noise_small = np.random.normal(0, brightness_variation / 255.0,
                                   (image_size // variation_scale, image_size // variation_scale))
    noise = cv2.resize(noise_small, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    img += noise * mask

    # --- rozmazání ---
    for _ in range(blur_repeat):
        img = cv2.GaussianBlur(img, (blur_strength * 2 + 1, blur_strength * 2 + 1), 0)

    # --- normalizace ---
    img = np.clip(img, 0.0, 1.0)
    if np.max(img) > 0:
        img /= np.max(img)

    return img

def generate_deformed_linear_border(m_px, image_size=350, deformation=2, waviness=1, num_points=4,
                          blur_strength=10, blur_repeat=2, intensity_variation=1000,
                          brightness_variation=100, variation_scale=20, thickness=2,
                          rotation_range=180, flattening_range=(0.85, 1.15)):
    """
    Generates a float32 image (0.0–1.0) with a hand-drawn arc that spans across the image,
    with blur, rotation, and optional intensity noise.
    """

    if random.random() > 0.3:
        waviness = random.randint(0, 10)

    if random.random() > 0.5:
        num_points = random.randint(2, 6)
        waviness = random.randint(10, 80)
    # --- základní init ---
    img = np.zeros((image_size, image_size), dtype=np.float32)
    mask = np.zeros_like(img)

    # --- škálování ---
    base_scale = 1.25 - 0.63 * m_px
    scale_noise = random.uniform(0.9, 1.1)
    scale_factor = (base_scale if random.random() > 0.3 else 1.0) * scale_noise
    blur_strength = max(1, int(blur_strength * scale_factor)) if random.random() > 0.3 else blur_strength
    blur_repeat = max(1, int(blur_repeat * scale_factor))
    if random.random() > 0.7:
        blur_repeat = max(1, blur_repeat // 2)
    elif random.random() > 0.8:
        blur_repeat = blur_repeat + 1


    brightness_variation *= scale_factor
    variation_scale = max(5, int(variation_scale * (1 / scale_factor)))
    waviness = max(1, int(waviness * scale_factor))
    thickness = max(1, int(thickness * scale_factor))

    # --- geometrie oblouku ---
    # fixní velký radius přes obraz + mírná náhodná složka
    base_radius = image_size * random.uniform(0.8, 1.2)
    radius = int(base_radius / m_px)  # měřítkově závislý
    if random.random() > 0.5:
        radius = int(radius/2)
    radius = int(np.clip(radius, image_size * 0.6, image_size * 1.8))

    # --- úhlová délka oblouku závislá na m_px ---
    # max_angle_deg = int(np.clip(180 - m_px * 80, 60, 160))  # např. při m_px=1.5 jen ~60°
    max_angle_deg = int(np.clip(180 - m_px * 60, 100, 175))

    arc_angle_deg = random.uniform(max_angle_deg * 0.8, max_angle_deg)

    start_angle = random.uniform(0, 360 - arc_angle_deg)
    end_angle = start_angle + arc_angle_deg
    flattening_factor = random.uniform(*flattening_range)

    # --- výpočet středu tak, aby oblouk byl přes střed obrázku ---
    mid_angle = np.radians((start_angle + end_angle) / 2)
    center_x = image_size // 2 - int(radius * np.cos(mid_angle))
    center_y = image_size // 2 - int(radius * np.sin(mid_angle) * flattening_factor)

    # --- rotace celého tvaru ---
    rotation_angle = random.uniform(-rotation_range, rotation_range)
    rot_mat = cv2.getRotationMatrix2D((float(image_size // 2), float(image_size // 2)), rotation_angle, 1.0)

    # --- generování bodů oblouku ---
    def add_waviness(center, radius, start_angle, end_angle, num_points, waviness, flattening_factor):
        points = []
        for i in range(num_points + 1):
            angle = np.radians(start_angle + (end_angle - start_angle) * (i / num_points))
            r_offset = random.randint(-waviness, waviness)
            x = int(center[0] + (radius + r_offset) * np.cos(angle))
            y = int(center[1] + (radius + r_offset) * np.sin(angle) * flattening_factor)
            points.append([x, y])
        return np.array(points, dtype=np.float32)

    arc_pts = add_waviness((center_x, center_y), radius, start_angle, end_angle,
                           num_points, waviness, flattening_factor)

    # --- aplikace rotace ---
    ones = np.ones((arc_pts.shape[0], 1), dtype=np.float32)
    arc_homog = np.hstack([arc_pts, ones])
    rotated_pts = rot_mat @ arc_homog.T
    rotated_pts = rotated_pts[:2].T.astype(int)

    # --- vykreslení čáry ---
    for i in range(len(rotated_pts) - 1):
        pt1 = tuple(rotated_pts[i])
        pt2 = tuple(rotated_pts[i + 1])
        cv2.line(img, pt1, pt2, color=1.0, thickness=thickness * 2)
        cv2.line(mask, pt1, pt2, color=1.0, thickness=thickness * 2)

    # --- přidání šumu pouze do oblouku ---
    noise_small = np.random.normal(0, brightness_variation / 255.0,
                                   (image_size // variation_scale, image_size // variation_scale))
    noise = cv2.resize(noise_small, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    img += noise * mask

    # --- rozmazání ---
    for _ in range(blur_repeat):
        img = cv2.GaussianBlur(img, (blur_strength * 2 + 1, blur_strength * 2 + 1), 0)

    # --- normalizace ---
    img = np.clip(img, 0.0, 1.0)
    if np.max(img) > 0:
        img /= np.max(img)

    return img

import numpy as np
import cv2
import random

def generate_parallel_wavy_arcs(m_px, image_size=350, deformation=2, waviness=2,
                                wave_amplitude=10, wave_frequency=0.1, smoothness=32,
                                blur_strength=6, blur_repeat=6, intensity_variation=400,
                                brightness_variation=50, variation_scale=20, thickness=3,
                                rotation_range=180, flattening_range=(0.85, 1.2),
                                base_distance=30, distance_variation=10):
    """
    Generates a float32 image (0.0–1.0) with two parallel wavy arcs affected by m_px, with noise, blur and variation.
    """
    if random.random() > 0.7:
        wave_amplitude = random.randint(0, 20)
        wave_frequency = random.uniform(0.05, 0.2)
    # --- Fading parameters ---
    fade_factor = 1.0
    if random.random() > 0.7:
        fade_factor = random.uniform(0.2, 0.5)
        intensity_variation = random.randint(1, 400)
    elif random.random() > 0.5:
        fade_factor = random.uniform(3, 12)

    if random.random() > 0.7:
        base_distance = random.randint(5, 30)
        distance_variation = random.randint(0, base_distance)
    # if random.random() > 0.5:
    #     fade_factor = random.uniform(0.2, 4)
    #     intensity_variation = random.randint(500, 1500)

    # --- škálování ---
    base_scale = 1.25 - 0.63 * m_px
    scale_noise = random.uniform(0.9, 1.1)
    scale_factor = (base_scale if random.random() > 0.3 else 1.0) * scale_noise
    blur_strength = max(1, int(blur_strength * scale_factor)) if random.random() > 0.3 else blur_strength
    blur_repeat = max(1, int(blur_repeat * scale_factor))
    if random.random() > 0.7:
        blur_repeat = max(1, blur_repeat // 2)
    elif random.random() > 0.8:
        blur_repeat += 1

    if random.random() > 0.3:
        waviness = random.randint(1, 4)

    if random.random() > 0.5 and m_px >= 0.4:
        blur_repeat += random.randint(1, 3)
    elif random.random() > 0.7:
        blur_repeat = random.randint(3, 7)

    waviness = max(1, int(waviness * scale_factor))
    wave_amplitude = int(wave_amplitude * scale_factor)
    thickness = max(1, int(thickness * scale_factor))
    brightness_variation *= scale_factor
    variation_scale = max(5, int(variation_scale * (1 / scale_factor)))

    # --- délka oblouku ---
    max_angle = int(np.clip(180 - m_px * 80, 40, 160))
    arc_angle = random.uniform(max_angle * 0.8, max_angle)
    start_angle = random.uniform(0, 360 - arc_angle)
    end_angle = start_angle + arc_angle

    # --- poloměr ---
    if random.random() > 0.7:
        radius = int((image_size * random.uniform(0.8, 1.2)) / m_px)
    else:
        radius = int(image_size * m_px * random.uniform(0.7, 1.2))

    radius = int(np.clip(radius, image_size * 0.6, image_size * 1.8))
    flattening_factor = random.uniform(*flattening_range)
    adjusted_distance = int(base_distance * (1 / m_px) + random.randint(-distance_variation, distance_variation))

    img = np.zeros((image_size, image_size), dtype=np.float32)
    mask = np.zeros_like(img)

    # --- výpočet středu ---
    mid_angle = np.radians((start_angle + end_angle) / 2)
    center_x = image_size // 2 - int(radius * np.cos(mid_angle))
    center_y = image_size // 2 - int(radius * np.sin(mid_angle) * flattening_factor)

    # --- rotace ---
    rotation_angle = random.uniform(-rotation_range, rotation_range)
    rot_mat = cv2.getRotationMatrix2D((image_size // 2, image_size // 2), rotation_angle, 1.0)

    # --- generování bodů oblouků ---
    def generate_faded_arc(center, radius, start_angle, end_angle, num_points,
                           waviness, wave_amplitude, wave_frequency,
                           flattening_factor, offset=0):
        pts = []
        intensities = np.clip(
            np.random.randint(255 - intensity_variation // 2,
                              255 + intensity_variation // 2 + 1, num_points + 1), 0, 255) / 255.0
        for i in range(num_points + 1):
            angle = np.radians(start_angle + (end_angle - start_angle) * (i / num_points))
            r_offset = random.randint(-waviness, waviness) + offset + int(wave_amplitude * np.sin(i * wave_frequency))
            x = int(center[0] + (radius + r_offset) * np.cos(angle))
            y = int(center[1] + (radius + r_offset) * np.sin(angle) * flattening_factor)
            t = (1 - np.cos(i / num_points * np.pi * fade_factor + random.uniform(-0.5, 0.5))) / 2
            pts.append((x, y, intensities[i] * t))
        return pts

    arc1 = generate_faded_arc((center_x, center_y), radius, start_angle, end_angle,
                              smoothness, waviness, wave_amplitude, wave_frequency,
                              flattening_factor)
    arc2 = generate_faded_arc((center_x, center_y), radius + adjusted_distance, start_angle, end_angle,
                              smoothness, waviness, wave_amplitude, wave_frequency,
                              flattening_factor)

    # --- rotace ---
    def apply_rotation(points):
        rotated = []
        for x, y, intensity in points:
            vec = np.array([x, y, 1.0])
            new = rot_mat @ vec
            rotated.append((int(new[0]), int(new[1]), intensity))
        return rotated

    arc1 = apply_rotation(arc1)
    arc2 = apply_rotation(arc2)

    # --- vykreslení ---
    for arc in (arc1, arc2):
        for i in range(len(arc) - 1):
            pt1 = (arc[i][0], arc[i][1])
            pt2 = (arc[i + 1][0], arc[i + 1][1])
            val = arc[i][2]
            cv2.line(img, pt1, pt2, color=val, thickness=thickness * 2)
            cv2.line(mask, pt1, pt2, color=val, thickness=thickness * 2)

    # --- noise ---
    noise_small = np.random.normal(0, brightness_variation / 255.0,
                                   (image_size // variation_scale, image_size // variation_scale))
    noise = cv2.resize(noise_small, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    img += noise * mask

    # --- blur ---
    for _ in range(blur_repeat):
        img = cv2.GaussianBlur(img, (blur_strength * 2 + 1, blur_strength * 2 + 1), 0)

    # --- normalizace ---
    img = np.clip(img, 0.0, 1.0)
    if np.max(img) > 0:
        img /= np.max(img)

    return img



def generate_parallel_linear_borders(m_px, image_size=350, deformation=2, waviness=2,
                                wave_amplitude=10, wave_frequency=0.1, num_points=4,
                                blur_strength=6, blur_repeat=6, intensity_variation=400,
                                brightness_variation=50, variation_scale=20, thickness=3,
                                rotation_range=180, flattening_range=(0.85, 1.2),
                                base_distance=10, distance_variation=10, fade_factor=1.0):
    """
    Generates a float32 image (0.0–1.0) with two pairs of parallel wavy arcs affected by m_px,
    with noise, blur and variation. Includes a parameter `num_points` to control lomenost.
    """

    def randomize_parameters():
        nonlocal wave_amplitude, wave_frequency, fade_factor
        nonlocal intensity_variation, base_distance, distance_variation
        nonlocal blur_strength, blur_repeat, waviness
        if random.random() > 0.5:
            num_points = random.randint(2, 6)
        if random.random() > 0.7:
            wave_amplitude = random.randint(0, 20)
            wave_frequency = random.uniform(0.05, 0.2)

        fade_factor = 1.0
        if random.random() > 0.7:
            fade_factor = random.uniform(0.2, 0.5)
            intensity_variation = random.randint(1, 400)
        elif random.random() > 0.5:
            fade_factor = random.uniform(3, 12)

        if random.random() > 0.7:
            base_distance = random.randint(5, 30)
            distance_variation = random.randint(0, base_distance)

        if random.random() > 0.3:
            waviness = random.randint(1, 4)

    def generate_arc_parameters(offset_x=0, offset_y=0):
        max_angle = int(np.clip(180 - m_px * 80, 40, 160))
        arc_angle = random.uniform(max_angle * 0.8, max_angle)
        start_angle = random.uniform(0, 360 - arc_angle)
        end_angle = start_angle + arc_angle

        if random.random() > 0.7:
            radius = int((image_size * random.uniform(0.8, 1.2)) / m_px)
        else:
            radius = int(image_size * m_px * random.uniform(0.7, 1.2))
        radius = int(np.clip(radius, image_size * 0.6, image_size * 1.8))
        if random.random() > 0.7:
            radius = int(radius / 1.66)
        flattening_factor = random.uniform(*flattening_range)
        adjusted_distance = int(base_distance * (1 / m_px) + random.randint(-distance_variation, distance_variation))

        mid_angle = np.radians((start_angle + end_angle) / 2)
        center_x = image_size // 2 - int(radius * np.cos(mid_angle)) + offset_x
        center_y = image_size // 2 - int(radius * np.sin(mid_angle) * flattening_factor) + offset_y

        rotation_angle = random.uniform(-rotation_range, rotation_range)

        return center_x, center_y, radius, start_angle, end_angle, flattening_factor, adjusted_distance, rotation_angle

    def generate_arc_pair(center_x, center_y, radius, start_angle, end_angle,
                          flattening_factor, adjusted_distance, rotation_angle,
                          num_points):

        rot_mat = cv2.getRotationMatrix2D((image_size // 2, image_size // 2), rotation_angle, 1.0)

        def generate_faded_arc(center, radius, start_angle, end_angle, num_points,
                               waviness, wave_amplitude, wave_frequency,
                               flattening_factor, offset=0):
            pts = []
            intensities = np.clip(
                np.random.randint(255 - intensity_variation // 2,
                                  255 + intensity_variation // 2 + 1, num_points + 1), 0, 255) / 255.0
            for i in range(num_points + 1):
                angle = np.radians(start_angle + (end_angle - start_angle) * (i / num_points))
                r_offset = random.randint(-waviness, waviness) + offset + int(wave_amplitude * np.sin(i * wave_frequency))
                x = int(center[0] + (radius + r_offset) * np.cos(angle))
                y = int(center[1] + (radius + r_offset) * np.sin(angle) * flattening_factor)
                t = (1 - np.cos(i / num_points * np.pi * fade_factor + random.uniform(-0.5, 0.5))) / 2
                pts.append((x, y, intensities[i] * t))
            return pts

        def apply_rotation(points):
            rotated = []
            for x, y, intensity in points:
                vec = np.array([x, y, 1.0])
                new = rot_mat @ vec
                rotated.append((int(new[0]), int(new[1]), intensity))
            return rotated

        arc1 = generate_faded_arc((center_x, center_y), radius, start_angle, end_angle,
                                  num_points, waviness, wave_amplitude, wave_frequency,
                                  flattening_factor)
        arc2 = generate_faded_arc((center_x, center_y), radius + adjusted_distance, start_angle, end_angle,
                                  num_points, waviness, wave_amplitude, wave_frequency,
                                  flattening_factor)

        return apply_rotation(arc1), apply_rotation(arc2)

    # --- náhodné úpravy ---
    randomize_parameters()
    base_scale = 1.25 - 0.63 * m_px
    scale_noise = random.uniform(0.9, 1.1)
    scale_factor = (base_scale if random.random() > 0.3 else 1.0) * scale_noise
    blur_strength = max(1, int(blur_strength * scale_factor)) if random.random() > 0.3 else blur_strength
    blur_repeat = max(1, int(blur_repeat * scale_factor))
    if random.random() > 0.7:
        blur_repeat = max(1, blur_repeat // 2)
    elif random.random() > 0.8:
        blur_repeat += 1

    if random.random() > 0.5 and m_px >= 0.4:
        blur_repeat += random.randint(1, 3)
    elif random.random() > 0.7:
        blur_repeat = random.randint(3, 7)

    waviness = max(1, int(waviness * scale_factor))
    wave_amplitude = int(wave_amplitude * scale_factor)
    thickness = max(1, int(thickness * scale_factor))
    brightness_variation *= scale_factor
    variation_scale = max(5, int(variation_scale * (1 / scale_factor)))

    img = np.zeros((image_size, image_size), dtype=np.float32)
    mask = np.zeros_like(img)

    # --- první sada ---
    arc_params_1 = generate_arc_parameters()
    arc1, arc2 = generate_arc_pair(*arc_params_1, num_points=num_points)


    # --- druhá sada s posunem ---
    arc_params_2 = generate_arc_parameters(offset_x=random.randint(-30, 30), offset_y=random.randint(-30, 30))
    arc3, arc4 = generate_arc_pair(*arc_params_2, num_points=num_points)

    # --- vykreslení ---
    counter = 0
    for arc in (arc1, arc2, arc3, arc4):
        if counter >= 2 and random.random() > 0.0:
            break
        for i in range(len(arc) - 1):
            pt1 = (arc[i][0], arc[i][1])
            pt2 = (arc[i + 1][0], arc[i + 1][1])
            val = arc[i][2]
            cv2.line(img, pt1, pt2, color=val, thickness=thickness * 2)
            cv2.line(mask, pt1, pt2, color=val, thickness=thickness * 2)
        counter += 1

    # --- šum ---
    noise_small = np.random.normal(0, brightness_variation / 255.0,
                                   (image_size // variation_scale, image_size // variation_scale))
    noise = cv2.resize(noise_small, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    img += noise * mask

    # --- rozmazání ---
    for _ in range(blur_repeat):
        img = cv2.GaussianBlur(img, (blur_strength * 2 + 1, blur_strength * 2 + 1), 0)

    # --- normalizace ---
    img = np.clip(img, 0.0, 1.0)
    if np.max(img) > 0:
        img /= np.max(img)

    return img


def generate_parallel_wavy_lines(m_px, image_size=350, angle_range=(-10, 10),
                                 waviness=2, smoothness=50,
                                 wave_amplitude_range=(10, 80), wave_frequency_range=(0.04, 0.21),
                                 amplitude_variation=0.9, frequency_variation=0.02,
                                 blur_strength=6, blur_repeat=5, brightness_variation=80,
                                 variation_scale=20, thickness=3, rotation_range=180,
                                 distance_range=(20, 85), fade_factor=0.5):
    """
    Generates a float32 image (0.0–1.0) with two parallel hand-drawn-like wavy lines affected by m_px, with blur, noise, fade and variability.
    """

    if random.random() > 0.5:
        waviness = random.randint(1, 3)

    base_scale = 1.25 - 0.63 * m_px
    scale_noise = random.uniform(0.9, 1.1)
    scale_factor = (base_scale if random.random() > 0.3 else 1.0) * scale_noise

    blur_strength = max(1, int(blur_strength * scale_factor)) if random.random() > 0.3 else blur_strength
    blur_repeat = max(1, int(blur_repeat * scale_factor))
    if random.random() > 0.7:
        blur_repeat = max(1, blur_repeat // 2)

    waviness = max(1, int(waviness * scale_factor))
    thickness = max(1, int(thickness * scale_factor))
    brightness_variation *= scale_factor
    variation_scale = max(5, int(variation_scale * (1 / scale_factor)))

    line_length = int(image_size * (1.0 / m_px) * random.uniform(0.9, 1.1))

    img = np.zeros((image_size, image_size), dtype=np.float32)
    mask = np.zeros_like(img)

    base_wave_amplitude = random.uniform(*wave_amplitude_range) * scale_factor
    base_wave_frequency = random.uniform(*wave_frequency_range) * scale_factor

    wave_amplitude1 = base_wave_amplitude + random.uniform(-amplitude_variation, amplitude_variation)
    wave_frequency1 = base_wave_frequency + random.uniform(-frequency_variation, frequency_variation)
    wave_amplitude2 = base_wave_amplitude + random.uniform(-amplitude_variation, amplitude_variation)
    wave_frequency2 = base_wave_frequency + random.uniform(-frequency_variation, frequency_variation)

    distance = random.uniform(*distance_range) * scale_factor

    start_x = image_size // 2 - line_length // 2
    start_y = image_size // 2
    angle = np.radians(random.uniform(*angle_range))

    def generate_wavy_path(start_x, start_y, length, angle, waviness, smoothness, wave_amplitude, wave_frequency):
        points = []
        for i in range(smoothness + 1):
            t = i / smoothness
            x = int(start_x + t * length * np.cos(angle) + random.uniform(-waviness, waviness))
            y = int(start_y + t * length * np.sin(angle) + random.uniform(-waviness, waviness)
                    + wave_amplitude * np.sin(i * wave_frequency))
            points.append((x, y))
        return points

    line1 = generate_wavy_path(start_x, start_y, line_length, angle, waviness, smoothness, wave_amplitude1, wave_frequency1)
    line2 = generate_wavy_path(start_x, start_y + int(distance), line_length, angle, waviness, smoothness, wave_amplitude2, wave_frequency2)

    rotation_angle = random.uniform(-rotation_range, rotation_range)
    rotation_matrix = cv2.getRotationMatrix2D((image_size // 2, image_size // 2), rotation_angle, 1.0)

    def rotate_points(points, rotation_matrix):
        points_np = np.array(points, dtype=np.float32)
        ones = np.ones((points_np.shape[0], 1))
        points_np = np.hstack([points_np, ones])
        rotated_points = rotation_matrix @ points_np.T
        return rotated_points[:2].T.astype(int)

    line1 = rotate_points(line1, rotation_matrix)
    line2 = rotate_points(line2, rotation_matrix)

    for i in range(len(line1) - 1):
        fade = (1 - np.cos(i / (len(line1) - 1) * np.pi * fade_factor)) / 2  # fade 0→1→0
        value = float(fade)
        cv2.line(img, tuple(line1[i]), tuple(line1[i + 1]), color=value, thickness=thickness * 2)
        cv2.line(mask, tuple(line1[i]), tuple(line1[i + 1]), color=value, thickness=thickness * 2)
        cv2.line(img, tuple(line2[i]), tuple(line2[i + 1]), color=value, thickness=thickness * 2)
        cv2.line(mask, tuple(line2[i]), tuple(line2[i + 1]), color=value, thickness=thickness * 2)

    noise_small = np.random.normal(0, brightness_variation / 255.0,
                                   (image_size // variation_scale, image_size // variation_scale))
    noise = cv2.resize(noise_small, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    img += noise * mask

    for _ in range(blur_repeat):
        img = cv2.GaussianBlur(img, (blur_strength * 2 + 1, blur_strength * 2 + 1), 0)

    img = np.clip(img, 0.0, 1.0)
    if np.max(img) > 0:
        img /= np.max(img)

    return img



# def generate_parallel_wavy_lines2(m_px, image_size=350, angle_range=(-10, 10),
#                                  waviness=2, smoothness=50,
#                                  wave_amplitude_range=(5, 30), wave_frequency_range=(0.04, 0.21),
#                                  amplitude_variation=0.9, frequency_variation=0.02,
#                                  blur_strength=6, blur_repeat=5, brightness_variation=80,
#                                  variation_scale=20, thickness=3, rotation_range=180,
#                                  distance_range=(20, 85), fade_factor=1.0):
#     """
#     Generates a float32 image (0.0–1.0) with two parallel hand-drawn-like wavy lines affected by m_px, with fade, noise, blur and variability.
#     """
#     if m_px > 1.5:
#         print("generate_parallel_wavy_lines2 Does not support this m_px value")
#         return None
#     # --- škálování ---
#     base_scale = 1.25 - 0.63 * m_px
#     scale_noise = random.uniform(0.9, 1.1)
#     scale_factor = (base_scale if random.random() > 0.3 else 1.0) * scale_noise
#
#     blur_strength = max(1, int(blur_strength * scale_factor)) if random.random() > 0.3 else blur_strength
#     blur_repeat = max(1, int(blur_repeat * scale_factor))
#     if random.random() > 0.7:
#         blur_repeat = max(1, blur_repeat // 2)
#
#     waviness = max(1, int(waviness * scale_factor))
#     thickness = max(1, int(thickness * scale_factor))
#     brightness_variation *= scale_factor
#     variation_scale = max(5, int(variation_scale * (1 / scale_factor)))
#
#     # --- délka linky škálovaná podle m_px ---
#     line_length = int(image_size * (1.0 / m_px) * random.uniform(0.9, 1.1))
#     line_length = int(image_size * np.clip((1.2 / m_px) * random.uniform(0.9, 1.1), 0.6, 3.5))
#     # --- pozice středu ---
#     center_x, center_y = image_size // 2, image_size // 2
#     start_x = center_x - line_length // 2
#     start_y = center_y
#
#     # --- vlnění ---
#     base_wave_amplitude = random.uniform(*wave_amplitude_range) * scale_factor
#     base_wave_frequency = random.uniform(*wave_frequency_range) * scale_factor
#
#     wave_amplitude1 = base_wave_amplitude + random.uniform(-amplitude_variation, amplitude_variation)
#     wave_frequency1 = base_wave_frequency + random.uniform(-frequency_variation, frequency_variation)
#     wave_amplitude2 = base_wave_amplitude + random.uniform(-amplitude_variation, amplitude_variation)
#     wave_frequency2 = base_wave_frequency + random.uniform(-frequency_variation, frequency_variation)
#
#     distance = random.uniform(*distance_range) * scale_factor
#
#     angle = np.radians(random.uniform(*angle_range))
#
#     img = np.zeros((image_size, image_size), dtype=np.float32)
#     mask = np.zeros_like(img)
#
#     # --- trajektorie ---
#     def generate_wavy_path(start_x, start_y, length, angle, waviness, smoothness, wave_amplitude, wave_frequency, fade_factor):
#         points = []
#         intensities = []
#         for i in range(smoothness + 1):
#             t = i / smoothness
#             x = int(start_x + t * length * np.cos(angle) + random.uniform(-waviness, waviness))
#             y = int(start_y + t * length * np.sin(angle) + random.uniform(-waviness, waviness)
#                     + wave_amplitude * np.sin(i * wave_frequency))
#             fade = (1 - np.cos(t * np.pi * fade_factor)) / 2
#             intensity = np.clip(fade, 0.0, 1.0)
#             points.append((x, y))
#             intensities.append(intensity)
#         return points, intensities
#
#     line1, intensities1 = generate_wavy_path(start_x, start_y, line_length, angle, waviness, smoothness, wave_amplitude1, wave_frequency1, fade_factor)
#     line2, intensities2 = generate_wavy_path(start_x, start_y + int(distance), line_length, angle, waviness, smoothness, wave_amplitude2, wave_frequency2, fade_factor)
#
#     # --- rotace ---
#     rotation_angle = random.uniform(-rotation_range, rotation_range)
#     rot_mat = cv2.getRotationMatrix2D((float(center_x), float(center_y)), rotation_angle, 1.0)
#
#     def rotate_points_with_intensity(points, intensities, rot_mat):
#         pts_np = np.array(points, dtype=np.float32)
#         ones = np.ones((pts_np.shape[0], 1))
#         pts_h = np.hstack([pts_np, ones])
#         pts_rot = rot_mat @ pts_h.T
#         pts_rot = pts_rot[:2].T.astype(int)
#         return list(zip(pts_rot, intensities))
#
#     line1 = rotate_points_with_intensity(line1, intensities1, rot_mat)
#     line2 = rotate_points_with_intensity(line2, intensities2, rot_mat)
#
#     # --- vykreslení ---
#     for i in range(len(line1) - 1):
#         pt1, v1 = line1[i]
#         pt2, _ = line1[i + 1]
#         cv2.line(img, pt1, pt2, color=v1, thickness=thickness * 2)
#         cv2.line(mask, pt1, pt2, color=v1, thickness=thickness * 2)
#
#         pt1, v2 = line2[i]
#         pt2, _ = line2[i + 1]
#         cv2.line(img, pt1, pt2, color=v2, thickness=thickness * 2)
#         cv2.line(mask, pt1, pt2, color=v2, thickness=thickness * 2)
#
#     # --- šum v masce ---
#     noise_small = np.random.normal(0, brightness_variation / 255.0,
#                                    (image_size // variation_scale, image_size // variation_scale))
#     noise = cv2.resize(noise_small, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
#     img += noise * mask
#
#     # --- rozmazání ---
#     for _ in range(blur_repeat):
#         img = cv2.GaussianBlur(img, (blur_strength * 2 + 1, blur_strength * 2 + 1), 0)
#
#     # --- normalizace ---
#     img = np.clip(img, 0.0, 1.0)
#     if np.max(img) > 0:
#         img /= np.max(img)
#
#     return img

def generate_wavy_ellipses(m_px, image_size=350, num_ellipses_range=(2, 7), ellipse_size_range=(8, 13),
                           flattening_range=(0.6, 1.3), waviness=5, num_points=50, fade_factor=1.0,
                           blur_strength=6, blur_repeat=4, brightness_variation=70,
                           variation_scale=20, rotation_range=180, distance_range=(15, 35),
                           regularity_factor=0.35):
    """
    Generates a float32 image (0.0–1.0) with several wavy ellipses modulated by m_px, blur and noise.
    """

    if random.random() > 0.5:
        regularity_factor = max(0.1, random.random())
        regularity_factor = min(0.8, regularity_factor)
    if random.random() > 0.65:
        num_ellipses_range = (6, 15)
    if m_px >= 1.0 and random.random() > 0.3:
        num_ellipses_range = (4, 20)

    if random.random() > 0.5:
        waviness = random.randint(1, 5)
    # --- škálování ---
    base_scale = 1.25 - 0.63 * m_px
    scale_noise = random.uniform(0.9, 1.1)
    scale_factor = (base_scale if random.random() > 0.3 else 1.0) * scale_noise

    blur_strength = max(1, int(blur_strength * scale_factor)) if random.random() > 0.3 else blur_strength
    blur_repeat = max(1, int(blur_repeat * scale_factor))
    if random.random() > 0.7:
        blur_repeat = max(1, blur_repeat // 2)

    elif random.random() > 0.5:
        blur_repeat = blur_repeat - 1


    brightness_variation *= scale_factor
    variation_scale = max(5, int(variation_scale * (1 / scale_factor)))
    waviness = max(1, int(waviness * scale_factor))
    fade_factor *= random.uniform(0.8, 1.2)

    low = int(ellipse_size_range[0] * scale_factor)
    high = int(ellipse_size_range[1] * scale_factor)

    # Ošetření prázdného rozsahu
    if high <= low:
        high = low + 1

    ellipse_size_range_scaled = (low, high)

    if random.random() > 0.6 and m_px <= 0.25:
        val1 = int(12 * random.uniform(0.85, 1.25))
        val2 = int(16 * random.uniform(0.85, 1.25))
        ellipse_size_range_scaled = (min(val1, val2), max(val1, val2))
        if ellipse_size_range_scaled[0] == ellipse_size_range_scaled[1]:
            ellipse_size_range_scaled = (ellipse_size_range_scaled[0], ellipse_size_range_scaled[0] + 1)

    # --- inicializace ---
    img = np.zeros((image_size, image_size), dtype=np.float32)
    mask = np.zeros_like(img)

    num_ellipses = random.randint(*num_ellipses_range)
    center_x, center_y = image_size // 2, image_size // 2
    radius = random.uniform(distance_range[0], distance_range[1]) * num_ellipses / 2 * scale_factor

    # --- centra elips ---
    centers = []
    for i in range(num_ellipses):
        angle = (i / num_ellipses) * 2 * np.pi + random.uniform(-np.pi * regularity_factor, np.pi * regularity_factor)
        r_offset = random.uniform(-radius * regularity_factor, radius * regularity_factor)
        x = int(center_x + (radius + r_offset) * np.cos(angle))
        y = int(center_y + (radius + r_offset) * np.sin(angle))
        centers.append((x, y))

    # --- generování elips ---
    def generate_wavy_ellipse(center, size, flattening, angle, num_points, waviness, fade_factor):
        points = []
        base_radius = size
        for i in range(num_points + 1):
            t = (1 - np.cos(i / num_points * np.pi * fade_factor)) / 2
            theta = t * 2 * np.pi
            r_offset = random.randint(-waviness, waviness)
            x = int(center[0] + (base_radius + r_offset) * np.cos(theta))
            y = int(center[1] + (base_radius * flattening + r_offset) * np.sin(theta))
            points.append([x, y])

        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        ones = np.ones((len(points), 1))
        points = np.hstack([np.array(points), ones])
        points = rot_mat @ points.T
        return points[:2].T.astype(int)

    for center in centers:
        size = random.randint(*ellipse_size_range_scaled)
        flattening = random.uniform(*flattening_range)
        angle = random.uniform(-rotation_range, rotation_range)

        wavy_ellipse = generate_wavy_ellipse(center, size, flattening, angle, num_points, waviness, fade_factor)

        cv2.fillPoly(img, [wavy_ellipse], color=1.0)
        cv2.fillPoly(mask, [wavy_ellipse], color=1.0)

    # --- noise ---
    noise_small = np.random.normal(0, brightness_variation / 255.0,
                                   (image_size // variation_scale, image_size // variation_scale))
    noise = cv2.resize(noise_small, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    img += noise * mask

    # --- blur ---
    for _ in range(blur_repeat):
        img = cv2.GaussianBlur(img, (blur_strength * 2 + 1, blur_strength * 2 + 1), 0)

    # --- normalizace ---
    img = np.clip(img, 0.0, 1.0)
    if np.max(img) > 0:
        img /= np.max(img)

    return img


