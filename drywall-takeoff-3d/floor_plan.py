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

    def is_open(self, reference_line, target_lines, tolerance=10):
        open_ends = ['A', 'B']
        X1, Y1, X2, Y2 = reference_line[0]
        target_wall_lines = deepcopy(target_lines)
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
        target_wall_lines.remove(reference_line)
        for target_wall_line in target_wall_lines:
            target_X1, target_Y1, target_X2, target_Y2 = target_wall_line[0]
            if math.hypot(X1 - target_X1, Y1 - target_Y1) <= tolerance or math.hypot(X1 - target_X2, Y1 - target_Y2) <= tolerance:
                neighbor_lines.append(target_wall_line)
            if math.hypot(X2 - target_X1, Y2 - target_Y1) <= tolerance or math.hypot(X2 - target_X2, Y2 - target_Y2) <= tolerance:
                neighbor_lines.append(target_wall_line)

        return neighbor_lines

    def disconnected_shapes(self, wall_lines, tolerance=10):
        disconnected_shapes = list()
        unvisited = set(map(id, wall_lines))
        id_to_line = {id(line): line for line in wall_lines}

        while unvisited:
            start_id = unvisited.pop()
            start_line = id_to_line[start_id]

            disconnected_shape = list()
            stack = [start_line]

            while stack:
                current = stack.pop()
                current_id = id(current)

                if current_id not in unvisited and current != start_line:
                    continue

                unvisited.discard(current_id)
                disconnected_shape.append(current)

                neighbors = self.neighbors(
                    reference_line=current,
                    target_lines=wall_lines,
                    tolerance=tolerance
                )

                for neighbor in neighbors:
                    neighbor_id = id(neighbor)
                    if neighbor_id in unvisited:
                        stack.append(neighbor)

            disconnected_shapes.append(disconnected_shape)

        return disconnected_shapes
