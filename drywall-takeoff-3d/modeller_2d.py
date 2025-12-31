from copy import deepcopy
import cv2
import numpy as np
import math
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from skimage.morphology import skeletonize

from floor_plan import FloorPlan

__all__ = ["FloorPlan2D"]


class FloorPlan2D(FloorPlan):

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)

        self._hyperparameters = hyperparameters
        self._width_in_feet = self._hyperparameters["modelling"]["width_in_feet"]
        self._height_in_feet = self._hyperparameters["modelling"]["height_in_feet"]
        self._walls_2d = list()

    def _close_jagged_openings(
        self,
        wall_lines,
        tolerance_distance=5,
        n_steps=2500,
    ):
        if not wall_lines:
            return wall_lines
        try:
            wall_lines = wall_lines.tolist()
        except:
            ...

        for _ in range(n_steps):
            wall_lines_new = deepcopy(wall_lines)
            reference_line = wall_lines[np.random.randint(len(wall_lines))]
            X1, Y1, X2, Y2 = reference_line[0]
            reference_line_type = self.classify_line(X1, Y1, X2, Y2)
            if reference_line_type not in ["vertical", "horizontal"]:
                continue
            for target_line in wall_lines:
                target_X1, target_Y1, target_X2, target_Y2 = target_line[0]
                target_line_type = self.classify_line(target_X1, target_Y1, target_X2, target_Y2)
                if target_line_type not in ["vertical", "horizontal"]:
                    continue
                if reference_line_type == target_line_type:
                    euclidean_distance_1 = np.hypot(target_X1 - X2, target_Y1 - Y2)
                    euclidean_distance_2 = np.hypot(target_X2 - X1, target_Y2 - Y1)
                    if min(euclidean_distance_1, euclidean_distance_2) <= tolerance_distance:
                        if euclidean_distance_1 <= euclidean_distance_2:
                            new_line_horizontal = [[X2, Y2, target_X1, Y2]]
                            new_line_vertical = [[target_X1, Y2, target_X1, target_Y1]]
                            wall_lines_new.extend([new_line_horizontal, new_line_vertical])
                            break
                        if euclidean_distance_2 <= euclidean_distance_1:
                            new_line_horizontal = [[X1, Y1, target_X2, Y1]]
                            new_line_vertical = [[target_X2, Y1, target_X2, target_Y2]]
                            wall_lines_new.extend([new_line_horizontal, new_line_vertical])
                            break
                else:
                    if reference_line_type == "horizontal":
                        chebyshev_distance_1 = max(abs(target_X1 - X1), min(abs(target_Y1 - Y1), abs(target_Y2 - Y1)))
                        chebyshev_distance_2 = max(abs(target_X1 - X2), min(abs(target_Y1 - Y2), abs(target_Y2 - Y2)))
                        if min(chebyshev_distance_1, chebyshev_distance_2) <= tolerance_distance:
                            if chebyshev_distance_1 <= chebyshev_distance_2:
                                if Y1 >= min(target_Y1, target_Y2) and Y1 <= max(target_Y1, target_Y2):
                                    new_line = [[X1, Y1, target_X1, Y1]]
                                    wall_lines_new.append(new_line)
                                else:
                                    new_line_horizontal = [[X1, Y1, target_X1, Y1]]
                                    new_line_vertical = [[target_X1, Y1, target_X1, target_Y1 if abs(target_Y1 - Y1) < abs(target_Y2 - Y1) else target_Y2]]
                                    wall_lines_new.extend([new_line_horizontal, new_line_vertical])
                            else:
                                if Y2 >= min(target_Y1, target_Y2) and Y2 <= max(target_Y1, target_Y2):
                                    new_line = [[X2, Y2, target_X1, Y2]]
                                    wall_lines_new.append(new_line)
                                else:
                                    new_line_horizontal = [[X2, Y2, target_X1, Y2]]
                                    new_line_vertical = [[target_X1, Y2, target_X1, target_Y1 if abs(target_Y1 - Y2) < abs(target_Y2 - Y2) else target_Y2]]
                                    wall_lines_new.extend([new_line_horizontal, new_line_vertical])
                            break
                    else:
                        chebyshev_distance_1 = max(abs(target_Y1 - Y1), min(abs(target_X1 - X1), abs(target_X2 - X1)))
                        chebyshev_distance_2 = max(abs(target_Y1 - Y2), min(abs(target_X1 - X2), abs(target_X2 - X2)))
                        if min(chebyshev_distance_1, chebyshev_distance_2) <= tolerance_distance:
                            if chebyshev_distance_1 <= chebyshev_distance_2:
                                if X1 >= min(target_X1, target_X2) and X1 <= max(target_X1, target_X2):
                                    new_line = [[X1, Y1, X1, target_Y1]]
                                    wall_lines_new.append(new_line)
                                else:
                                    new_line_vertical = [[X1, Y1, X1, target_Y1]]
                                    new_line_horizontal = [[X1, target_Y1, target_X1 if abs(target_X1 - X1) < abs(target_X2 - X1) else target_X2, target_Y1]]
                                    wall_lines_new.extend([new_line_horizontal, new_line_vertical])
                            else:
                                if X2 >= min(target_X1, target_X2) and X2 <= max(target_X1, target_X2):
                                    new_line = [[X2, Y2, X2, target_Y1]]
                                    wall_lines_new.append(new_line)
                                else:
                                    new_line_vertical = [[X2, Y2, X2, target_Y1]]
                                    new_line_horizontal = [[X2, target_Y1, target_X1 if abs(target_X1 - X1) < abs(target_X2 - X1) else target_X2, target_Y1]]
                                    wall_lines_new.extend([new_line_horizontal, new_line_vertical])
                            break
            wall_lines = wall_lines_new

        return wall_lines

    def _sniff_and_split_orthogonal(
        self,
        wall_lines,
        tolerance_distance=5,
    ):
        if not wall_lines:
            return wall_lines
        try:
            wall_lines = wall_lines.tolist()
        except:
            ...

        horizontal_wall_lines = list()
        vertical_wall_lines = list()
        wall_lines_splitted = list()
        for wall_line in wall_lines:
            X1, Y1, X2, Y2 = wall_line[0]
            wall_line_type = self.classify_line(X1, Y1, X2, Y2)
            if wall_line_type == "horizontal":
                horizontal_wall_lines.append(wall_line)
            elif wall_line_type == "vertical":
                vertical_wall_lines.append(wall_line)
            else:
                wall_lines_splitted.append(wall_line)

        for horizontal_wall_line in horizontal_wall_lines:
            X1, Y1, X2, Y2 = horizontal_wall_line[0]
            split_found = False
            for target_line in vertical_wall_lines:
                target_X1, target_Y1, target_X2, target_Y2 = target_line[0]
                if target_X1 > min(X1, X2) and target_X1 < max(X1, X2):
                    if min(target_Y1, target_Y2) - tolerance_distance <= Y1 <= max(target_Y1, target_Y2) + tolerance_distance:
                        wall_lines_splitted.append([[X1, Y1, target_X1, Y1]])
                        wall_lines_splitted.append([[target_X1, Y1, X2, Y2]])
                        split_found = True
                        break
            if not split_found:
                wall_lines_splitted.append(horizontal_wall_line)

        for vertical_wall_line in vertical_wall_lines:
            X1, Y1, X2, Y2 = vertical_wall_line[0]
            split_found = False
            for target_line in horizontal_wall_lines:
                target_X1, target_Y1, target_X2, target_Y2 = target_line[0]
                if target_Y1 > min(Y1, Y2) and target_Y1 < max(Y1, Y2):
                    if min(target_X1, target_X2) - tolerance_distance <= X1 <= max(target_X1, target_X2) + tolerance_distance:
                        wall_lines_splitted.append([[X1, Y1, X1, target_Y1]])
                        wall_lines_splitted.append([[X1, target_Y1, X2, Y2]])
                        split_found  = True
                        break
            if not split_found:
                wall_lines_splitted.append(vertical_wall_line)

        return wall_lines_splitted

    def _close_wall_openings_deterministic(
        self,
        wall_lines,
        tolerance_distance=25
    ):
        try:
            wall_lines = wall_lines.tolist()
        except:
            ...
    
        new_lines = set()

        for i, reference_line in enumerate(wall_lines):
            X1 = min(reference_line[0][0], reference_line[0][2])
            X2 = max(reference_line[0][0], reference_line[0][2])
            Y1 = min(reference_line[0][1], reference_line[0][3])
            Y2 = max(reference_line[0][1], reference_line[0][3])

            reference_line_type = self.classify_line(X1, Y1, X2, Y2)
            if reference_line_type not in ["vertical", "horizontal"]:
                continue

            for j, target_line in enumerate(wall_lines):
                if i == j:
                    continue

                t_X1 = min(target_line[0][0], target_line[0][2])
                t_X2 = max(target_line[0][0], target_line[0][2])
                t_Y1 = min(target_line[0][1], target_line[0][3])
                t_Y2 = max(target_line[0][1], target_line[0][3])

                target_line_type = self.classify_line(t_X1, t_Y1, t_X2, t_Y2)
                if target_line_type not in ["vertical", "horizontal"]:
                    continue

                if reference_line_type == target_line_type == "horizontal":
                    if abs(t_Y1 - Y1) <= tolerance_distance:
                        if abs(t_X1 - X1) <= tolerance_distance:
                            new_lines.add((X1, Y1, X1, t_Y1))
                        if abs(t_X2 - X2) <= tolerance_distance:
                            new_lines.add((X2, Y2, X2, t_Y1))

                elif reference_line_type == target_line_type == "vertical":
                    if abs(t_X1 - X1) <= tolerance_distance:
                        if abs(t_Y1 - Y1) <= tolerance_distance:
                            new_lines.add((X1, Y1, t_X1, Y1))
                        if abs(t_Y2 - Y2) <= tolerance_distance:
                            new_lines.add((X2, Y2, t_X1, Y2))

        for x1, y1, x2, y2 in new_lines:
            wall_lines.append([[x1, y1, x2, y2]])

        return wall_lines

    def _group_lines(self, wall_lines, tolerance_distance=5):
        """Cluster lines into groups of nearby, same-orientation lines."""
        n = len(wall_lines)
        adjacency = defaultdict(list)

        for i in range(n):
            x1, y1, x2, y2 = wall_lines[i][0]
            type_i = self.classify_line(x1, y1, x2, y2)
            if not type_i:
                continue
            for j in range(i + 1, n):
                x3, y3, x4, y4 = wall_lines[j][0]
                type_j = self.classify_line(x3, y3, x4, y4)
                if type_i != type_j:
                    continue
                dists = [
                    np.hypot(x1 - x3, y1 - y3),
                    np.hypot(x1 - x4, y1 - y4),
                    np.hypot(x2 - x3, y2 - y3),
                    np.hypot(x2 - x4, y2 - y4),
                ]
                if min(dists) <= tolerance_distance:
                    adjacency[i].append(j)
                    adjacency[j].append(i)

        visited = set()
        clusters = []
        for i in range(n):
            if i not in visited:
                stack = [i]
                cluster = []
                while stack:
                    node = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        cluster.append(node)
                        stack.extend(adjacency[node])
                clusters.append(cluster)

        return clusters

    def _merge_cluster(self, cluster, wall_lines):
        """Merge a cluster of lines into one smooth line."""
        points = list()
        for idx in cluster:
            x1, y1, x2, y2 = wall_lines[idx][0]
            points.extend([(x1, y1), (x2, y2)])

        x1, y1, x2, y2 = wall_lines[cluster[0]][0]
        line_type = self.classify_line(x1, y1, x2, y2)

        if line_type == "horizontal":
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            y_mean = int(round(np.mean(ys)))
            return [[min(xs), y_mean, max(xs), y_mean]]

        elif line_type == "vertical":
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x_mean = int(round(np.mean(xs)))
            return [[x_mean, min(ys), x_mean, max(ys)]]

        elif line_type == "inclined":
            xs = np.array([p[0] for p in points])
            ys = np.array([p[1] for p in points])
            m, c = np.polyfit(xs, ys, 1)
            x_min, x_max = xs.min(), xs.max()
            return [[int(x_min), int(m*x_min + c), int(x_max), int(m*x_max + c)]]

        return None

    def _jagged_to_smooth_lines_deterministic(
        self,
        wall_lines,
        tolerance_distance=15,
    ):
        """Convert jagged wall lines into smoother merged lines deterministically."""
        try:
            wall_lines = wall_lines.tolist()
        except:
            wall_lines = list(wall_lines)

        clusters = self._group_lines(wall_lines, tolerance_distance)
        merged_lines = list()

        for cluster in clusters:
            merged = self._merge_cluster(cluster, wall_lines)
            if merged:
                merged_lines.append(merged)

        return merged_lines

    def _remove_orthogonal_overlap(self, reference_line, target_line, reference_line_type="horizontal", tolerance_distance=5):
        x1_reference, y1_reference, x2_reference, y2_reference = reference_line[0]
        x1_target, y1_target, x2_target, y2_target = target_line[0]
        if reference_line_type == "horizontal":
            x_target = int(np.median([x1_target, x2_target]))
            y_reference = int(np.median([y1_reference, y2_reference]))
            if (x_target >= x1_reference and x_target <= x2_reference) and (y_reference >= y1_target and y_reference <= y2_target):
                if abs(y_reference - y1_target) < abs(y_reference - y2_target) and abs(y_reference - y1_target) <= tolerance_distance:
                    return [[x1_target, y_reference, x2_target, y2_target]]
                if abs(y_reference - y2_target) < abs(y_reference - y1_target) and abs(y_reference - y2_target) <= tolerance_distance:
                    return [[x1_target, y1_target, x2_target, y_reference]]
                return [[x1_target, y1_target, x2_target, y2_target]]
            return [[x1_target, y1_target, x2_target, y2_target]]
        if reference_line_type == "vertical":
            y_target = int(np.median([y1_target, y2_target]))
            x_reference = int(np.median([x1_reference, x2_reference]))
            if (y_target >= y1_reference and y_target <= y2_reference) and (x_reference >= x1_target and x_reference <= x2_target):
                if abs(x_reference - x1_target) < abs(x_reference - x2_target) and abs(x_reference - x1_target) <= tolerance_distance:
                    return [[x_reference, y1_target, x2_target, y2_target]]
                if abs(x_reference - x2_target) < abs(x_reference - x1_target) and abs(x_reference - x2_target) <= tolerance_distance:
                    return [[x1_target, y1_target, x_reference, y1_target]]
                return [[x1_target, y1_target, x2_target, y2_target]]
            return [[x1_target, y1_target, x2_target, y2_target]]

    def _no_orthogonal_overlap(
        self,
        wall_lines,
        tolerance_disance=100,
        n_steps=5000
    ):
        if wall_lines is None:
            return wall_lines
        for _ in range(n_steps):
            wall_lines_new = deepcopy(wall_lines)
            reference_wall_line = wall_lines[np.random.randint(len(wall_lines))]
            x1_reference, y1_reference, x2_reference, y2_reference = reference_wall_line[0]
            reference_line_type = self.classify_line(x1_reference, y1_reference, x2_reference, y2_reference)
            if reference_line_type not in ["vertical", "horizontal"]:
                continue
            for target_wall_line in wall_lines:
                x1_target, y1_target, x2_target, y2_target = target_wall_line[0]
                target_line_type = self.classify_line(x1_target, y1_target, x2_target, y2_target)
                if target_line_type not in ["vertical", "horizontal"]:
                    continue
                if reference_line_type == "horizontal" and target_line_type == "vertical":
                    target_wall_line_new = self._remove_orthogonal_overlap(reference_wall_line, target_wall_line, reference_line_type="horizontal", tolerance_distance=tolerance_disance)
                    wall_lines_new.remove(target_wall_line)
                    wall_lines_new.append(target_wall_line_new)
                if reference_line_type == "vertical" and target_line_type == "horizontal":
                    target_wall_line_new = self._remove_orthogonal_overlap(reference_wall_line, target_wall_line, reference_line_type="vertical", tolerance_distance=tolerance_disance)
                    wall_lines_new.remove(target_wall_line)
                    wall_lines_new.append(target_wall_line_new)
            wall_lines = wall_lines_new

        return wall_lines

    def _thin_edges(self, binary_image):
        """
        Convert thick walls/edges into 1-pixel wide lines.
        Input: binary image (edges in white, background black)
        Output: thinned skeleton image
        """
        _, binary = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)

        skeleton = np.zeros(binary.shape, np.uint8)

        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        done = False
        img = binary.copy()

        while not done:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            img = eroded.copy()

            done = (cv2.countNonZero(img) == 0)

        return skeleton

    def _load_topology(self, floor_plan_wall_segmented_binary: np.ndarray):
        def _binary_to_bool(img: np.ndarray) -> np.ndarray:
            return img > 0

        def _bool_to_binary(img_bool: np.ndarray) -> np.ndarray:
            return ((img_bool == False).astype(np.uint8) * 255)

        floor_plan_wall_segmented_bool = _binary_to_bool(floor_plan_wall_segmented_binary)
        floor_plan_topology_bool = skeletonize(floor_plan_wall_segmented_bool)
        floor_plan_topology_binary = _bool_to_binary(floor_plan_topology_bool)

        return floor_plan_topology_binary

    def _preprocessing(self, image_BGR, max_split=5, output_path=None):
        _, thresh = cv2.threshold(image_BGR, 50, 255, cv2.THRESH_BINARY_INV)

        edges_thinned = self._thin_edges(thresh)
        edges = cv2.Canny(edges_thinned, 50, 100, apertureSize=3)

        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)

        floor_plan_topology_binary = self._load_topology(edges)
        lines = self.detect_lines(edges)
        lines = self._normalize(lines)
        if lines is not None:
            lines = self._jagged_to_smooth_lines_deterministic(lines)
            lines = self._close_jagged_openings(lines)
            lines = self._close_wall_openings_deterministic(lines)
            for _ in range(3):
                lines = self._topology_guided_extend_and_conquer(lines, floor_plan_topology_binary)
                lines = self._topology_guided_closure_open_lines(lines, floor_plan_topology_binary)
            for _ in range(max_split):
                lines = self._sniff_and_split_orthogonal(lines)
                lines = self._deduplicate_lines(lines)
            lines = self._remove_invalid(lines)

            lines = self._topology_guided_closure_open_lines_dead_end(lines, maximum_length=250)
            lines = self._sniff_and_split_orthogonal(lines)
            lines = self._deduplicate_lines(lines)
            lines = self._remove_invalid(lines)

        if output_path:
            canvas = np.ones(image_BGR.shape, dtype=np.uint8) * 255
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 0), 1)
            cv2.imwrite(output_path, canvas)

        return lines

    def _topology_guided_closure_open_lines_dead_end(self, wall_lines, maximum_length=1000, tolerance=5):
        wall_lines_closed_dead_end = list()
        canvas = np.ones((1080, 1920), dtype=np.uint8) * 255
        for wall_line in wall_lines:
            X1, Y1, X2, Y2 = wall_line[0]
            cv2.line(canvas, (X1, Y1), (X2, Y2), 0, 1)
        for wall_line in wall_lines:
            X1, Y1, X2, Y2 = wall_line[0]
            open_ends = self.is_open(wall_line, wall_lines)
            if open_ends:
                orientation = self.classify_line(X1, Y1, X2, Y2)
                if orientation == "horizontal":
                    Y = int(np.median([Y1, Y2]))
                    if 'A' in open_ends:
                        target_X1 = X1 - np.argmin(canvas[Y - tolerance: Y + tolerance, : X1].mean(axis=0)[::-1])
                        pixel_value = canvas[Y - tolerance: Y + tolerance, : X1].mean(axis=0)[::-1][np.argmin(canvas[Y - tolerance: Y + tolerance, : X1].mean(axis=0)[::-1])]
                        if target_X1 > 0 and pixel_value != 255 and abs(X1 - target_X1) <= maximum_length:
                            wall_lines_closed_dead_end.append([[target_X1, Y, X1, Y]])
                    if 'B' in open_ends:
                        target_X2 = X2 + np.argmin(canvas[Y - tolerance: Y + tolerance, X2+1:].mean(axis=0))
                        pixel_value = canvas[Y - tolerance: Y + tolerance, X2+1:].mean(axis=0)[np.argmin(canvas[Y - tolerance: Y + tolerance, X2+1:].mean(axis=0))]
                        if target_X2 < 1920 and pixel_value != 255 and abs(target_X2 - X2) <= maximum_length:
                            wall_lines_closed_dead_end.append([[X2, Y, target_X2, Y]])
                if orientation == "vertical":
                    X = int(np.median([X1, X2]))
                    if 'A' in open_ends:
                        target_Y1 = Y1 - np.argmin(canvas[: Y1, X - tolerance: X + tolerance].mean(axis=1)[::-1])
                        pixel_value = canvas[: Y1, X - tolerance: X + tolerance].mean(axis=1)[::-1][np.argmin(canvas[: Y1, X - tolerance: X + tolerance].mean(axis=1)[::-1])]
                        if target_Y1 > 0  and pixel_value != 255 and abs(Y1 - target_Y1) <= maximum_length:
                            wall_lines_closed_dead_end.append([[X, target_Y1, X, Y1]])
                    if 'B' in open_ends:
                        target_Y2 = Y2 + np.argmin(canvas[Y2+1:, X - tolerance: X + tolerance].mean(axis=1))
                        pixel_value = canvas[Y2+1:, X - tolerance: X + tolerance].mean(axis=1)[np.argmin(canvas[Y2+1:, X - tolerance: X + tolerance].mean(axis=1))]
                        if target_Y2 < 1080  and pixel_value != 255 and abs(target_Y2 - Y2) <= maximum_length:
                            wall_lines_closed_dead_end.append([[X, Y2, X, target_Y2]])
                wall_lines_closed_dead_end.append(wall_line)
            else:
                wall_lines_closed_dead_end.append(wall_line)

        return wall_lines_closed_dead_end

    def _topology_guided_extend_and_conquer(self, lines, floor_plan_topology_binary, extension_maximum=10, tolerance=2):
        extended_lines = list()
        for line in lines:
            X1, Y1, X2, Y2 = line[0]
            orientation = self.classify_line(X1, Y1, X2, Y2)
            if orientation == "horizontal":
                Y = int(np.median([Y1, Y2]))
                for n_pixels in range(extension_maximum):
                    X1_new = max(0, X1 - (n_pixels + 1))
                    if np.any(floor_plan_topology_binary[Y - tolerance: Y + tolerance, X1_new]==0):
                        X1 = X1_new
                for n_pixels in range(extension_maximum):
                    X2_new = min(1920 - 1, X2 + (n_pixels + 1))
                    if np.any(floor_plan_topology_binary[Y - tolerance: Y + tolerance, X2_new]==0):
                        X2 = X2_new
                extended_lines.append([[X1, Y1, X2, Y2]])
            elif orientation == "vertical":
                X = int(np.median([X1, X2]))
                for n_pixels in range(extension_maximum):
                    Y1_new = max(0, Y1 - (n_pixels + 1))
                    if np.any(floor_plan_topology_binary[Y1_new, X - tolerance: X + tolerance]==0):
                        Y1 = Y1_new
                for n_pixels in range(extension_maximum):
                    Y2_new = min(1080 - 1, Y2 + (n_pixels + 1))
                    if np.any(floor_plan_topology_binary[Y2_new, X - tolerance: X + tolerance]==0):
                        Y2 = Y2_new
                extended_lines.append([[X1, Y1, X2, Y2]])
            else:
                extended_lines.append(line)

        return extended_lines

    def _topology_guided_closure_open_lines(self, wall_lines, floor_plan_topology_binary, extension_maximum=20, tolerance=10):
        if not wall_lines:
            return wall_lines
        try:
            wall_lines = wall_lines.tolist()
        except:
            wall_lines = list(wall_lines)

        def extend_open_line(wall_line, wall_line_type, open_end_type):
            extended_lines = list()
            X1, Y1, X2, Y2 = wall_line[0]
            if wall_line_type == "horizontal" and open_end_type == 'A':
                Y = int(np.median([Y1, Y2]))
                target_X1, target_Y1, target_X2, target_Y2 = X1, np.inf, X1, np.inf
                for n_pixels in range(extension_maximum):
                    Y_new = max(0, Y - (n_pixels + 1))
                    if np.any(floor_plan_topology_binary[Y_new, X1 - tolerance: X1 + tolerance]==0):
                        if target_Y2 == np.inf:
                            target_Y2 = Y_new
                        else:
                            target_Y1 = Y_new
                if target_Y1 != np.inf and target_Y2 != np.inf and math.hypot(0, target_Y1 - target_Y2) > tolerance:
                    extended_lines.append([[target_X1, target_Y1, target_X2, target_Y2]])
                target_X1, target_Y1, target_X2, target_Y2 = X1, np.inf, X1, np.inf
                for n_pixels in range(extension_maximum):
                    Y_new = min(1080 - 1, Y + (n_pixels + 1))
                    if np.any(floor_plan_topology_binary[Y_new, X1 - tolerance: X1 + tolerance]==0):
                        if target_Y1 == np.inf:
                            target_Y1 = Y_new
                        else:
                            target_Y2 = Y_new
                if target_Y1 != np.inf and target_Y2 != np.inf and math.hypot(0, target_Y1 - target_Y2) > tolerance:
                    extended_lines.append([[target_X1, target_Y1, target_X2, target_Y2]])

            if wall_line_type == "horizontal" and open_end_type == 'B':
                Y = int(np.median([Y1, Y2]))
                target_X1, target_Y1, target_X2, target_Y2 = X2, np.inf, X2, np.inf
                for n_pixels in range(extension_maximum):
                    Y_new = max(0, Y - (n_pixels + 1))
                    if np.any(floor_plan_topology_binary[Y_new, X2 - tolerance: X2 + tolerance]==0):
                        if target_Y2 == np.inf:
                            target_Y2 = Y_new
                        else:
                            target_Y1 = Y_new
                if target_Y1 != np.inf and target_Y2 != np.inf and math.hypot(0, target_Y1 - target_Y2) > tolerance:
                    extended_lines.append([[target_X1, target_Y1, target_X2, target_Y2]])
                target_X1, target_Y1, target_X2, target_Y2 = X2, np.inf, X2, np.inf
                for n_pixels in range(extension_maximum):
                    Y_new = min(1080 - 1, Y + (n_pixels + 1))
                    if np.any(floor_plan_topology_binary[Y_new, X2 - tolerance: X2 + tolerance]==0):
                        if target_Y1 == np.inf:
                            target_Y1 = Y_new
                        else:
                            target_Y2 = Y_new
                if target_Y1 != np.inf and target_Y2 != np.inf and math.hypot(0, target_Y1 - target_Y2) > tolerance:
                    extended_lines.append([[target_X1, target_Y1, target_X2, target_Y2]])

            if wall_line_type == "vertical" and open_end_type == 'A':
                X = int(np.median([X1, X2]))
                target_X1, target_Y1, target_X2, target_Y2 = np.inf, Y1, np.inf, Y1
                for n_pixels in range(extension_maximum):
                    X_new = max(0, X - (n_pixels + 1))
                    if np.any(floor_plan_topology_binary[Y1 - tolerance: Y1 + tolerance, X_new]==0):
                        if target_X2 == np.inf:
                            target_X2 = X_new
                        else:
                            target_X1 = X_new
                if target_X1 != np.inf and target_X2 != np.inf and math.hypot(target_X1 - target_X2, 0) > tolerance:
                    extended_lines.append([[target_X1, target_Y1, target_X2, target_Y2]])
                target_X1, target_Y1, target_X2, target_Y2 = np.inf, Y1, np.inf, Y1
                for n_pixels in range(extension_maximum):
                    X_new = min(1920 - 1, X + (n_pixels + 1))
                    if np.any(floor_plan_topology_binary[Y1 - tolerance: Y1 + tolerance, X_new]==0):
                        if target_X1 == np.inf:
                            target_X1 = X_new
                        else:
                            target_X2 = X_new
                if target_X1 != np.inf and target_X2 != np.inf and math.hypot(target_X1 - target_X2, 0) > tolerance:
                    extended_lines.append([[target_X1, target_Y1, target_X2, target_Y2]])

            if wall_line_type == "vertical" and open_end_type == 'B':
                X = int(np.median([X1, X2]))
                target_X1, target_Y1, target_X2, target_Y2 = np.inf, Y2, np.inf, Y2
                for n_pixels in range(extension_maximum):
                    X_new = max(0, X - (n_pixels + 1))
                    if np.any(floor_plan_topology_binary[Y2 - tolerance: Y2 + tolerance, X_new]==0):
                        if target_X2 == np.inf:
                            target_X2 = X_new
                        else:
                            target_X1 = X_new
                if target_X1 != np.inf and target_X2 != np.inf and math.hypot(target_X1 - target_X2, 0) > tolerance:
                    extended_lines.append([[target_X1, target_Y1, target_X2, target_Y2]])
                target_X1, target_Y1, target_X2, target_Y2 = np.inf, Y2, np.inf, Y2
                for n_pixels in range(extension_maximum):
                    X_new = min(1920 - 1, X + (n_pixels + 1))
                    if np.any(floor_plan_topology_binary[Y2 - tolerance: Y2 + tolerance, X_new]==0):
                        if target_X1 == np.inf:
                            target_X1 = X_new
                        else:
                            target_X2 = X_new
                if target_X1 != np.inf and target_X2 != np.inf and math.hypot(target_X1 - target_X2, 0) > tolerance:
                    extended_lines.append([[target_X1, target_Y1, target_X2, target_Y2]])

            return extended_lines

        horizontal_wall_lines = list()
        vertical_wall_lines = list()
        wall_lines_closed = list()

        for wall_line in wall_lines:
            X1, Y1, X2, Y2 = wall_line[0]
            line_type = self.classify_line(X1, Y1, X2, Y2)

            if line_type == "horizontal":
                horizontal_wall_lines.append(wall_line)
            elif line_type == "vertical":
                vertical_wall_lines.append(wall_line)
            else:
                wall_lines_closed.append(wall_line)

        for horizontal_wall_line in horizontal_wall_lines:
            open_ends = self.is_open(horizontal_wall_line, wall_lines, tolerance=tolerance)
            if open_ends and 'A' in open_ends:
                extended_lines = extend_open_line(horizontal_wall_line, "horizontal", 'A')
                if extended_lines:
                    wall_lines_closed.extend(extended_lines)
            if open_ends and 'B' in open_ends:
                extended_lines = extend_open_line(horizontal_wall_line, "horizontal", 'B')
                if extended_lines:
                    wall_lines_closed.extend(extended_lines)
            wall_lines_closed.append(horizontal_wall_line)
        for vertical_wall_line in vertical_wall_lines:
            open_ends = self.is_open(vertical_wall_line, wall_lines, tolerance=tolerance)
            if open_ends and 'A' in open_ends:
                extended_lines = extend_open_line(vertical_wall_line, "vertical", 'A')
                if extended_lines:
                    wall_lines_closed.extend(extended_lines)
            if open_ends and 'B' in open_ends:
                extended_lines = extend_open_line(vertical_wall_line, "vertical", 'B')
                if extended_lines:
                    wall_lines_closed.extend(extended_lines)
            wall_lines_closed.append(vertical_wall_line)

        return wall_lines_closed

    def _remove_invalid(self, wall_lines, open_tolerance_threshold=2, length_tolerance_threshold=10):
        valid_wall_lines = list()
        for wall_line in wall_lines:
            X1, Y1, X2, Y2 = wall_line[0]
            if math.hypot(X1 - X2, Y1 - Y2) <= length_tolerance_threshold:
                continue
            open_ends = self.is_open(wall_line, wall_lines, tolerance=open_tolerance_threshold)
            if open_ends and 'A' in open_ends and 'B' in open_ends:
                continue
            valid_wall_lines.append(wall_line)

        return valid_wall_lines

    def _normalize(self, lines):
        if lines is None:
            return lines
        normalized_lines = list()
        for line in lines:
            x0, y0, x1, y1 = line[0]
            distance_coord_0 = np.hypot(x0 - 0, y0 - 0)
            distance_coord_1 = np.hypot(x1 - 0, y1 - 0)
            if distance_coord_0 <= distance_coord_1 and [[x0, y0, x1, y1]] not in normalized_lines:
                normalized_lines.append([[x0, y0, x1, y1]]) 
            elif distance_coord_0 > distance_coord_1 and [[x1, y1, x0, y0]] not in normalized_lines:
                normalized_lines.append([[x1, y1, x0, y0]])
        return normalized_lines

    def _deduplicate_lines(self, wall_lines, tolerance=10):
        if wall_lines is None:
            return wall_lines
        unique = list()
        for l in wall_lines:
            x1, y1, x2, y2 = l[0]
            duplicate = False
            for u in unique:
                ux1, uy1, ux2, uy2 = u[0]
                if (abs(x1-ux1) <= tolerance and abs(y1-uy1) <= tolerance and 
                    abs(x2-ux2) <= tolerance and abs(y2-uy2) <= tolerance):
                    duplicate = True
                    break
            if not duplicate:
                unique.append(l)
        horizontal_wall_lines = list()
        vertical_wall_lines = list()
        deduplicated = list()

        for wall_line in unique:
            X1, Y1, X2, Y2 = wall_line[0]
            line_type = self.classify_line(X1, Y1, X2, Y2)

            if line_type == "horizontal":
                horizontal_wall_lines.append(wall_line)
            elif line_type == "vertical":
                vertical_wall_lines.append(wall_line)
            else:
                deduplicated.append(wall_line)

        for horizontal_wall_line in horizontal_wall_lines:
            X1, Y1, X2, Y2 = horizontal_wall_line[0]
            target_horizontal_wall_lines = deepcopy(horizontal_wall_lines)
            target_horizontal_wall_lines.remove(horizontal_wall_line)
            is_duplicate = False
            for target_horizontal_wall_line in target_horizontal_wall_lines:
                target_X1, target_Y1, target_X2, target_Y2 = target_horizontal_wall_line[0]
                if X1 >= target_X1 and X2 <= target_X2 and abs(np.median([Y1, Y2]) - np.median([target_Y1, target_Y2])) <= tolerance:
                    is_duplicate = True
                    break
            if not is_duplicate:
                deduplicated.append(horizontal_wall_line)

        for vertical_wall_line in vertical_wall_lines:
            X1, Y1, X2, Y2 = vertical_wall_line[0]
            target_vertical_wall_lines = deepcopy(vertical_wall_lines)
            target_vertical_wall_lines.remove(vertical_wall_line)
            is_duplicate = False
            for target_vertical_wall_line in target_vertical_wall_lines:
                target_X1, target_Y1, target_X2, target_Y2 = target_vertical_wall_line[0]
                if Y1 >= target_Y1 and Y2 <= target_Y2 and abs(np.median([X1, X2]) - np.median([target_X1, target_X2])) <= tolerance:
                    is_duplicate = True
                    break
            if not is_duplicate:
                deduplicated.append(vertical_wall_line)

        return deduplicated

    def _draw_line(self, wall_line, canvas):
        x1, y1, x2, y2 = wall_line[0]['x'], wall_line[0]['y'], wall_line[1]['x'], wall_line[1]['y']
        cv2.line(canvas, (x1, y1), (x2, y2), (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)), 1)

        return canvas

    def _patch_to_line(self, patch_GRAY, output_path):
        lines = self._preprocessing(
            patch_GRAY,
            output_path=Path(output_path).with_suffix(".tmp" + Path(output_path).suffix)
        )
        if lines is None:
            return

        lines = self._deduplicate_lines(lines)
        perimeter_lines, outer_drywall_surfaces = self.perimeter_lines(lines)

        return lines, perimeter_lines, outer_drywall_surfaces

    def _add_wall(self, wall_line, polygons, index):
        X1, Y1, X2, Y2 = wall_line[0]
        wall = dict(
            id=index,
            wall_line=[
                dict(x=int(X1), y=int(Y1)),
                dict(x=int(X2), y=int(Y2))
            ],
            thickness=self._width_in_feet,
            height=self._height_in_feet,
            length=math.hypot((X1 - X2) * self._hyperparameters["modelling"]["pixel_aspect_ratio"]["horizontal"], (Y1 - Y2) * self._hyperparameters["modelling"]["pixel_aspect_ratio"]["vertical"]),
            polygons_drywall=list()
        )
        for polygon in polygons:
            wall["polygons_drywall"].append(
                dict(
                    polygon=polygon["coordinates"],
                    type="",
                    enabled=polygon["enabled"],
                    room_name=''
                )
            )
        self._walls_2d.append(wall)

    def _extrude_drywall(self, line, outer_drywall_surface=None):
        polygons = list()
        X1, Y1, X2, Y2 = line[0]
        if self.classify_line(X1, Y1, X2, Y2) == "horizontal":
            polygon_a = [
                dict(x=int(X1+5), y=int(Y1-5)),
                dict(x=int(X2-5), y=int(Y2-5)),
                dict(x=int(X2-10), y=int(Y2-10)),
                dict(x=int(X1+10), y=int(Y1-10))
            ]
            if outer_drywall_surface == "UP":
                polygons.append(dict(coordinates=polygon_a, enabled=False))
            else:
                polygons.append(dict(coordinates=polygon_a, enabled=True))
            polygon_b = [
                dict(x=int(X1+5), y=int(Y1+5)),
                dict(x=int(X2-5), y=int(Y2+5)),
                dict(x=int(X2-10), y=int(Y2+10)),
                dict(x=int(X1+10), y=int(Y1+10))
            ]
            if outer_drywall_surface == "DOWN":
                polygons.append(dict(coordinates=polygon_b, enabled=False))
            else:
                polygons.append(dict(coordinates=polygon_b, enabled=True))

        if self.classify_line(X1, Y1, X2, Y2) == "vertical":
            polygon_a = [
                dict(x=int(X1-5), y=int(Y1+5)),
                dict(x=int(X2-5), y=int(Y2-5)),
                dict(x=int(X2-10), y=int(Y2-10)),
                dict(x=int(X1-10), y=int(Y1+10))
            ]
            if outer_drywall_surface == "LEFT":
                polygons.append(dict(coordinates=polygon_a, enabled=False))
            else:
                polygons.append(dict(coordinates=polygon_a, enabled=True))
            polygon_b = [
                dict(x=int(X1+5), y=int(Y1+5)),
                dict(x=int(X2+5), y=int(Y2-5)),
                dict(x=int(X2+10), y=int(Y2-10)),
                dict(x=int(X1+10), y=int(Y1+10))
            ]
            if outer_drywall_surface == "RIGHT":
                polygons.append(dict(coordinates=polygon_b, enabled=False))
            else:
                polygons.append(dict(coordinates=polygon_b, enabled=True))

        return polygons

    def save_plot_2d(self, model_2d_path, floor_plan_path="/tmp/floor_plan.png", overlay_enabled=False):
        with open(model_2d_path, 'r') as f:
            data = json.load(f)

        if overlay_enabled:
            canvas = cv2.imread(floor_plan_path)
            canvas = cv2.resize(canvas, (1920, 1080))
        else:
            canvas = np.ones((1080, 1920), dtype=np.uint8) * 255
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        for wall in data:
            self._draw_line(wall["wall_line"], canvas)
            for drywall in wall["polygons_drywall"]:
                pts = np.array([
                    [drywall["polygon"][0]['x'], drywall["polygon"][0]['y']],
                    [drywall["polygon"][1]['x'], drywall["polygon"][1]['y']],
                    [drywall["polygon"][2]['x'], drywall["polygon"][2]['y']],
                    [drywall["polygon"][3]['x'], drywall["polygon"][3]['y']]
                ], np.int32)
                pts = pts.reshape((-1, 1, 2))
                if drywall["enabled"]:
                    cv2.fillPoly(canvas, pts=[pts], color=(0, 255, 0))
                else:
                    cv2.fillPoly(canvas, pts=[pts], color=(0, 0, 255))

        if overlay_enabled:
            image_path = "/tmp/blueprint_model_2d_overlay_enabled.png"
        else:
            image_path = "/tmp/blueprint_model_2d.png"
        cv2.imwrite(image_path, canvas)
        return Path(image_path)

    def model(
        self,
        image_path="/tmp/floor_plan_wall_segmented.png",
        model_2d_path="/tmp/walls_2d.json",
        output_path="/tmp/blueprint_model_2d.png"
    ):
        image_GRAY = self.read_floor_plan(image_path)
        output_path = Path(output_path)
        wall_lines, perimeter_lines, outer_drywall_surfaces = self._patch_to_line(image_GRAY, output_path=output_path)
        for index, (perimeter_line, outer_drywall_surface) in enumerate(zip(perimeter_lines, outer_drywall_surfaces)):
            polygons = self._extrude_drywall(perimeter_line, outer_drywall_surface=outer_drywall_surface)
            self._add_wall(perimeter_line, polygons, index)
        for wall_line in wall_lines:
            index += 1
            if wall_line in perimeter_lines:
                continue
            polygons = self._extrude_drywall(wall_line)
            self._add_wall(wall_line, polygons, index)

        if model_2d_path:
            with open(model_2d_path, 'w') as f:
                json.dump(self._walls_2d, f, indent=2)
        return self._walls_2d, model_2d_path
