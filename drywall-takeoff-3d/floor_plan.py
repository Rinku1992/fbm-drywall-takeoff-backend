from copy import deepcopy

import math
import numpy as np
import cv2

__all__ = ["FloorPlan"]


class FloorPlan:

    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters
        self.tolerance_vertical = self.hyperparameters["modelling"]["tolerance_vertical"]
        self.tolerance_horizontal = self.hyperparameters["modelling"]["tolerance_horizontal"]

    def read_floor_plan(self, image_path, resize=None):
        image = cv2.imread(image_path).copy()
        if resize:
            image = cv2.resize(image, resize)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def detect_lines(self, image_BGR):
        lines = cv2.HoughLinesP(
            image_BGR,
            **self.hyperparameters["modelling"]["HoughLinesTransformation"]
        )
        return lines

    def image_to_patches(self, image):
        patches = list()
        kernel_parameters = self.hyperparameters["modelling"]["kernel"]

        n_horizontal_strides = (image.shape[1] // kernel_parameters["stride"]) + 1
        n_vertical_strides = (image.shape[0] // kernel_parameters["stride"]) + 1

        for v_stride_index in range(n_vertical_strides):
            for h_stride_index in range(n_horizontal_strides):
                X1 = h_stride_index * 100
                X2 = X1 + 1000
                Y1 = v_stride_index * 100
                Y2 = Y1 + 1000

                cropped_image = image[Y1:Y2, X1:X2]
                patches.append(cropped_image)
        return patches

    def classify_line(self, x1, y1, x2, y2):
        """Classify a line as horizontal, vertical, or inclined."""
        if abs(x2 - x1) > self.tolerance_horizontal and abs(y2 - y1) <= self.tolerance_vertical:
            return "horizontal"
        elif abs(x2 - x1) <= self.tolerance_horizontal and abs(y2 - y1) > self.tolerance_vertical:
            return "vertical"
        elif abs(x2 - x1) > self.tolerance_horizontal and abs(y2 - y1) > self.tolerance_vertical:
            return "inclined"
        return "invalid"

    def perimeter_lines(self, lines, resolution=(1080, 1920)):
        canvas = np.ones(resolution, dtype=np.uint8) * 255
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 0), 1)

        perimeter_lines = list()
        outer_drywall_surfaces = list()
        for line in lines:
            X1, Y1, X2, Y2 = line[0]
            orientation = self.classify_line(X1, Y1, X2, Y2)
            outer_drywall_surface = ''
            if orientation == "horizontal":
                up = np.any(canvas[:int(np.median([Y1, Y2])), X1: X2]==0)
                down = np.any(canvas[int(np.median([Y1, Y2])) + 1:, X1: X2]==0)
                left = np.any(canvas[int(np.median([Y1, Y2])), : X1]==0)
                right = np.any(canvas[int(np.median([Y1, Y2])), X2 + 1:]==0)
                if not up:
                    outer_drywall_surface = "UP"
                elif not down:
                    outer_drywall_surface = "DOWN"
                elif not left:
                    upper_left = np.any(canvas[int(np.median([Y1, Y2])) - 5, : X1]==0)
                    lower_left = np.any(canvas[int(np.median([Y1, Y2])) + 5, : X1]==0)
                    if not upper_left:
                        outer_drywall_surface = "UP"
                    elif not lower_left:
                        outer_drywall_surface = "DOWN"
                elif not right:
                    upper_right = np.any(canvas[int(np.median([Y1, Y2])) - 5, X2 + 1:]==0)
                    lower_right = np.any(canvas[int(np.median([Y1, Y2])) + 5, X2 + 1:]==0)
                    if not upper_right:
                        outer_drywall_surface = "UP"
                    elif not lower_right:
                        outer_drywall_surface = "DOWN"
            if orientation == "vertical":
                up = np.any(canvas[: Y1, int(np.median([X1, X2]))]==0)
                down = np.any(canvas[Y2 + 1:, int(np.median([X1, X2]))]==0)
                left = np.any(canvas[Y1: Y2, : int(np.median([X1, X2]))]==0)
                right = np.any(canvas[Y1: Y2, int(np.median([X1, X2])) + 1:]==0)
                if not left:
                    outer_drywall_surface = "LEFT"
                elif not right:
                    outer_drywall_surface = "RIGHT"
                elif not up:
                    upper_left = np.any(canvas[: Y1, int(np.median([X1, X2])) - 5]==0)
                    upper_right = np.any(canvas[: Y1, int(np.median([X1, X2])) + 5]==0)
                    if not upper_left:
                        outer_drywall_surface = "LEFT"
                    elif not upper_right:
                        outer_drywall_surface = "RIGHT"
                elif not down:
                    lower_left = np.any(canvas[Y2 + 1:, int(np.median([X1, X2])) - 5]==0)
                    lower_right = np.any(canvas[Y2 + 1:, int(np.median([X1, X2])) + 5]==0)
                    if not lower_left:
                        outer_drywall_surface = "LEFT"
                    elif not lower_right:
                        outer_drywall_surface = "RIGHT"
            if orientation != "horizontal" and orientation != "vertical":
                continue
            if not up or not down or not left or not right:
                perimeter_lines.append(line)
                outer_drywall_surfaces.append(outer_drywall_surface)

        return perimeter_lines, outer_drywall_surfaces

    def normalize(self, lines):
        if lines is None:
            return lines
        normalized_lines = list()
        for line in lines:
            X1, Y1, X2, Y2 = line[0]
            distance_coord_0 = np.hypot(X1 - 0, Y1 - 0)
            distance_coord_1 = np.hypot(X2 - 0, Y2 - 0)
            if distance_coord_0 <= distance_coord_1 and [[X1, Y1, X2, Y2]] not in normalized_lines:
                normalized_lines.append([[X1, Y1, X2, Y2]]) 
            elif distance_coord_0 > distance_coord_1 and [[X2, Y2, X1, Y1]] not in normalized_lines:
                normalized_lines.append([[X2, Y2, X1, Y1]])
        return normalized_lines

    def is_open(self, reference_line, target_lines, tolerance=10):
        open_ends = ['A', 'B']
        X1, Y1, X2, Y2 = reference_line[0]
        target_wall_lines = deepcopy(target_lines)
        if reference_line in target_wall_lines:
            target_wall_lines.remove(reference_line)
        for target_wall_line in target_wall_lines:
            target_X1, target_Y1, target_X2, target_Y2 = target_wall_line[0]
            if (math.hypot(X1 - target_X1, Y1 - target_Y1) <= tolerance or math.hypot(X1 - target_X2, Y1 - target_Y2) <= tolerance) and 'A' in open_ends:
                open_ends.remove('A')
            if (math.hypot(X2 - target_X1, Y2 - target_Y1) <= tolerance or math.hypot(X2 - target_X2, Y2 - target_Y2) <= tolerance) and 'B' in open_ends:
                open_ends.remove('B')

        return open_ends

    def neighbors(self, reference_line, target_lines, tolerance=10):
        neighbor_lines = list()
        X1, Y1, X2, Y2 = reference_line[0]
        target_wall_lines = deepcopy(target_lines)
        if reference_line in target_wall_lines:
            target_wall_lines.remove(reference_line)
        for target_wall_line in target_wall_lines:
            target_X1, target_Y1, target_X2, target_Y2 = target_wall_line[0]
            if math.hypot(X1 - target_X1, Y1 - target_Y1) <= tolerance or math.hypot(X1 - target_X2, Y1 - target_Y2) <= tolerance:
                neighbor_lines.append(target_wall_line)
            if math.hypot(X2 - target_X1, Y2 - target_Y1) <= tolerance or math.hypot(X2 - target_X2, Y2 - target_Y2) <= tolerance:
                neighbor_lines.append(target_wall_line)

        return neighbor_lines

    def nearest_neighbor(self, reference_line, target_lines, tolerance=200):
        X1, Y1, X2, Y2 = reference_line[0]
        distance_nearest_neighbor = np.inf
        target_wall_lines = deepcopy(target_lines)
        if reference_line in target_wall_lines:
            target_wall_lines.remove(reference_line)
        id_to_line = {id(target_wall_line): target_wall_line for target_wall_line in target_wall_lines}
        id_to_distance = dict()
        for wall_line_index, target_wall_line in id_to_line.items():
            target_X1, target_Y1, target_X2, target_Y2 = target_wall_line[0]
            euclidean_distance_A_A = math.hypot(X1 - target_X1, Y1 - target_Y1)
            euclidean_distance_A_B = math.hypot(X1 - target_X2, Y1 - target_Y2)
            euclidean_distance_B_A = math.hypot(X2 - target_X1, Y2 - target_Y1)
            euclidean_distance_B_B = math.hypot(X2 - target_X2, Y2 - target_Y2)
            if min(euclidean_distance_A_A, euclidean_distance_A_B, euclidean_distance_B_A, euclidean_distance_B_B) < distance_nearest_neighbor:
                distance_nearest_neighbor = min(euclidean_distance_A_A, euclidean_distance_A_B, euclidean_distance_B_A, euclidean_distance_B_B)
                id_to_distance[wall_line_index] = distance_nearest_neighbor

        if min(id_to_distance.values()) <= tolerance:
            index_minimum_id_to_distance = list(id_to_distance.values()).index(min(id_to_distance.values()))
            nearest_neighbor = id_to_line[list(id_to_distance.keys())[index_minimum_id_to_distance]]
            return nearest_neighbor

    def disconnected_shapes(self, wall_lines, tolerance=10):
        disconnected_shapes = list()
        unvisited = deepcopy(wall_lines)

        while unvisited:
            start_line = unvisited.pop()
            disconnected_shape = list()
            stack = [start_line]

            while stack:
                current_line = stack.pop()

                if current_line not in unvisited and current_line != start_line:
                    continue

                if current_line in unvisited and current_line != start_line:
                    unvisited.remove(current_line)
                disconnected_shape.append(current_line)

                wall_lines_target = deepcopy(wall_lines)
                neighbor_lines = self.neighbors(
                    reference_line=current_line,
                    target_lines=wall_lines_target,
                    tolerance=tolerance
                )
                for neighbor_line in neighbor_lines:
                    if neighbor_line in unvisited:
                        stack.append(neighbor_line)

            disconnected_shapes.append(disconnected_shape)

        return disconnected_shapes
