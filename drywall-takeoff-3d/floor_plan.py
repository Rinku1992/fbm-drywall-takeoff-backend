from copy import deepcopy

from fractions import Fraction
import math
import numpy as np
import cv2
from scipy.spatial import cKDTree
from pdf2image import convert_from_path

__all__ = ["FloorPlan"]


class FloorPlan:

    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters
        self.tolerance_angle = self.hyperparameters["modelling"]["tolerance_angle"]
        self._lines_classified = dict()
        self._perimeter_lines = list()

    def read_floor_plan(self, image_path, resize=None):
        image = cv2.imread(image_path).copy()
        if resize:
            image = cv2.resize(image, resize, interpolation=cv2.INTER_LANCZOS4)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def compute_imperial_scale_from_DPI(self, architectural_scale):
        DPI = self.hyperparameters["modelling"]["scale_adoption"]["dpi"]
        scale_on_paper_length = float(Fraction(architectural_scale.split('=')[0].strip('`')))
        imperial = DPI * scale_on_paper_length
        return (1 / imperial, 1 / imperial)

    def compute_pixel_aspect_ratio(self, scale_new, pixel_aspect_ratio_standard):
        scale_new_unnormalized = f"{round(float(Fraction(scale_new.split('=')[0].strip('`'))), 2)}``:1`0``"
        scale_new_on_paper_length = float(scale_new_unnormalized.split(':')[0].strip('`"'))
        scale_new_real_world_length_in_feet_and_inches_ = scale_new_unnormalized.split(':')[1].strip('`"')
        if scale_new_real_world_length_in_feet_and_inches_.find('`') !=-1:
            scale_new_real_world_length_in_feet_and_inches = scale_new_real_world_length_in_feet_and_inches_.split('`')
        else:
            scale_new_real_world_length_in_feet_and_inches = scale_new_real_world_length_in_feet_and_inches_.split("'")
        scale_new_real_world_length_in_feet = float(scale_new_real_world_length_in_feet_and_inches[0])
        if len(scale_new_real_world_length_in_feet_and_inches) == 2:
            scale_new_real_world_length_in_feet += float(scale_new_real_world_length_in_feet_and_inches[-1]) / 12
        scale_new_real_world_length_in_feet_scale = 0.25 / scale_new_on_paper_length
        scale_new_real_world_length_in_feet_new = scale_new_real_world_length_in_feet * scale_new_real_world_length_in_feet_scale
        scale = scale_new_real_world_length_in_feet_new / 1
        pixel_aspect_ratio_new = dict()
        pixel_aspect_ratio_new["horizontal"] = scale * pixel_aspect_ratio_standard["horizontal"]
        pixel_aspect_ratio_new["vertical"] = scale * pixel_aspect_ratio_standard["vertical"]
        pixel_aspect_ratio_new["area"] = scale * pixel_aspect_ratio_standard["area"]

        return pixel_aspect_ratio_new

    def detect_lines(self, image_GRAY):
        lines = cv2.HoughLinesP(
            image_GRAY,
            **self.hyperparameters["modelling"]["HoughLinesTransformation"]
        )
        return lines

    def load_KD_tree(self, lines):
        points = list()
        line_lookup = list()

        for idx, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]

            points.append((x1, y1))
            line_lookup.append(idx)

            points.append((x2, y2))
            line_lookup.append(idx)

        tree = cKDTree(np.array(points))
        return tree, line_lookup

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

    def is_inside_polygon(self, coordinate, polygon_vertices, tolerance=1e-9):
        X, Y = coordinate
        inside = False

        n_polygon_vertices = len(polygon_vertices)

        for i in range(n_polygon_vertices):
            X1, Y1 = polygon_vertices[i]
            X2, Y2 = polygon_vertices[(i + 1) % n_polygon_vertices]

            if (
                abs((Y2 - Y1) * (X - X1) - (X2 - X1) * (Y - Y1)) < tolerance and
                min(X1, X2) - tolerance <= X <= max(X1, X2) + tolerance and
                min(Y1, Y2) - tolerance <= Y <= max(Y1, Y2) + tolerance
            ):
                return True

            intersects = ((Y1 > Y) != (Y2 > Y))
            if intersects:
                x_intersect = (X2 - X1) * (Y - Y1) / (Y2 - Y1 + tolerance) + X1
                if X < x_intersect:
                    inside = not inside

        return inside

    def classify_line(self, x1, y1, x2, y2):
        line_id = str([x1, y1, x2, y2])
        if line_id in self._lines_classified:
            return self._lines_classified[line_id]
        inclination = math.degrees(math.atan2(abs(y1 - y2), abs(x1 - x2)))
        if inclination <= self.tolerance_angle:
            orientation = "horizontal"
        elif inclination >= 90 - self.tolerance_angle:
            orientation = "vertical"
        else:
            orientation = "inclined"
        self._lines_classified[line_id] = orientation
        return orientation

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
        tree, line_lookup = self.load_KD_tree(target_lines)
        x1, y1, x2, y2 = reference_line[0]

        neighbor_indices = set()

        near_p1 = tree.query_ball_point(
            [x1, y1],
            tolerance
        )

        near_p2 = tree.query_ball_point(
            [x2, y2],
            tolerance
        )

        all_hits = near_p1 + near_p2

        for endpoint_idx in all_hits:
            line_idx = line_lookup[endpoint_idx]
            candidate_line = target_lines[line_idx]

            if candidate_line != reference_line:
                neighbor_indices.add(line_idx)

        return [
            target_lines[i]
            for i in neighbor_indices
        ]

    def nearest_neighbor(self, reference_line, end_type, target_lines, tolerance=500):
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
            if end_type == 'A':
                if min(euclidean_distance_A_A, euclidean_distance_A_B) < distance_nearest_neighbor:
                    distance_nearest_neighbor = min(euclidean_distance_A_A, euclidean_distance_A_B)
                    id_to_distance[wall_line_index] = distance_nearest_neighbor
            if end_type == 'B':
                if min(euclidean_distance_B_A, euclidean_distance_B_B) < distance_nearest_neighbor:
                    distance_nearest_neighbor = min(euclidean_distance_B_A, euclidean_distance_B_B)
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

    def load_perimeter(self, coordinates, wall_lines, tolerance=10, bound_capture=True, scale=None):
        if scale:
            scale_x, scale_y = scale
            tolerance *= np.mean([scale[0], scale[1]])
        perimeter_lines = list()
        for source_coordinate in coordinates:
            for target_coordinate in coordinates:
                perimeter_line_found = False
                perimeter_segments = list()
                X1, Y1, X2, Y2 = self.normalize([[[source_coordinate[0], source_coordinate[1], target_coordinate[0], target_coordinate[1]]]])[0][0]
                if scale:
                    orientation = self.classify_line(X1 / scale_x, Y1 / scale_y, X2 / scale_x, Y2 / scale_y)
                else:
                    orientation = self.classify_line(X1, Y1, X2, Y2)
                for wall_line in wall_lines:
                    target_X1, target_Y1, target_X2, target_Y2 = wall_line[0]
                    if scale:
                        orientation_target = self.classify_line(target_X1 / scale_x, target_Y1 / scale_y, target_X2 / scale_x, target_Y2 / scale_y)
                    else:
                        orientation_target = self.classify_line(target_X1, target_Y1, target_X2, target_Y2)
                    if orientation == "horizontal" and orientation_target == "horizontal":
                        if abs(np.median([Y1, Y2]) - np.median([target_Y1, target_Y2])) <= tolerance and target_X1 - X1 >= -tolerance and target_X2 - X2 <= tolerance:
                            perimeter_segments.append(wall_line)
                    if orientation == "vertical" and orientation_target == "vertical":
                        if abs(np.median([X1, X2]) - np.median([target_X1, target_X2])) <= tolerance and target_Y1 - Y1 >= -tolerance and target_Y2 - Y2 <= tolerance:
                            perimeter_segments.append(wall_line)

                    if abs(target_X1 - X1) <= tolerance and abs(target_Y1 - Y1) <= tolerance and abs(target_X2 - X2) <= tolerance and abs(target_Y2 - Y2) <= tolerance:
                        perimeter_line_found = True
                        perimeter_line = [[target_X1, target_Y1, target_X2, target_Y2]]
                        if perimeter_line not in perimeter_lines:
                            perimeter_lines.append([[target_X1, target_Y1, target_X2, target_Y2]])
                        break
                if not perimeter_line_found:
                    for perimeter_segment in perimeter_segments:
                        if perimeter_segment not in perimeter_lines:
                            perimeter_lines.append(perimeter_segment)

        if bound_capture:
            for wall_line in wall_lines:
                target_X1, target_Y1, target_X2, target_Y2 = wall_line[0]
                if self.is_inside_polygon((target_X1, target_Y1), coordinates) and self.is_inside_polygon((target_X2, target_Y2), coordinates):
                    if wall_line not in perimeter_lines:
                        perimeter_lines.append(wall_line)

        return perimeter_lines

    def perimeter_lines(self, lines, resolution=(1080, 1920)):
        canvas = np.ones(resolution, dtype=np.uint8) * 255
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 0), 1)

        perimeter_lines = list()
        outer_drywall_surfaces = list()
        for line in self._perimeter_lines:
            X1, Y1, X2, Y2 = line[0]
            if self._perimeter_lines.count(line) == 2:
                if line not in perimeter_lines:
                    perimeter_lines.append(line)
                    outer_drywall_surfaces.append("INVALID")
                continue
            orientation = self.classify_line(X1, Y1, X2, Y2)
            if orientation == "horizontal":
                up = np.any(canvas[:int(np.median([Y1, Y2])), X1: X2]==0)
                down = np.any(canvas[int(np.median([Y1, Y2])) + 1:, X1: X2]==0)
                left = np.any(canvas[int(np.median([Y1, Y2])), : X1]==0)
                right = np.any(canvas[int(np.median([Y1, Y2])), X2 + 1:]==0)
                outer_drywall_surface = ''
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
                if not outer_drywall_surface:
                    continue
                perimeter_lines.append(line)
                outer_drywall_surfaces.append(outer_drywall_surface)
            if orientation == "vertical":
                up = np.any(canvas[: Y1, int(np.median([X1, X2]))]==0)
                down = np.any(canvas[Y2 + 1:, int(np.median([X1, X2]))]==0)
                left = np.any(canvas[Y1: Y2, : int(np.median([X1, X2]))]==0)
                right = np.any(canvas[Y1: Y2, int(np.median([X1, X2])) + 1:]==0)
                outer_drywall_surface = ''
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
                if not outer_drywall_surface:
                    continue
                perimeter_lines.append(line)
                outer_drywall_surfaces.append(outer_drywall_surface)

        return perimeter_lines, outer_drywall_surfaces

    def _smoothen_polygon(self, coordinates, edge_minimum=10, tolerance=20, minimal_expected_polygon_sides=4):
        polygon_smoothened = [coordinates[0]]
        for coordinate in coordinates[1:]:
            X1, Y1 = polygon_smoothened[-1]
            X2, Y2 = coordinate
            if math.hypot(X1 - X2, Y1 - Y2) < edge_minimum:
                continue
            orientation = self.classify_line(X1, Y1, X2, Y2)
            if orientation != "horizontal" and orientation != "vertical" and math.hypot(X1 - X2, Y1 - Y2) <= tolerance:
                continue
            polygon_smoothened.append(coordinate)
        if len(polygon_smoothened) < minimal_expected_polygon_sides:
            if abs(X2 - X1) < abs(Y2 - Y1):
                polygon_smoothened.append((X1, Y2))
            else:
                polygon_smoothened.append((X2, Y1))

        return polygon_smoothened

    def polygonize(self, wall_lines):
        canvas = np.ones((1080, 1920), dtype=np.uint8) * 255
        for wall_line in wall_lines:
            X1, Y1, X2, Y2 = wall_line[0]
            cv2.line(canvas, (X1, Y1), (X2, Y2), (0, 0, 0), 1)
        _, canvas_binary = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        canvas_dilated = cv2.dilate(canvas_binary, kernel, iterations=2)
        canvas_eroded = cv2.erode(canvas_dilated, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(canvas_eroded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        polygonized = list()
        self._perimeter_lines = wall_lines * 2
        perimeter_lines_contours = list()
        for contour, component in zip(contours, hierarchy[0]):
            if component[3] == -1:
                continue
            area = cv2.contourArea(contour)
            if area < 250:
                continue

            epsilon = 0.01 * cv2.arcLength(contour, True)
            geometry_polygons = cv2.approxPolyDP(contour, epsilon, True)

            coordinates = [
                (round(coordinate[0][0]), round(coordinate[0][1])) for coordinate in geometry_polygons
            ]
            coordinates = self._smoothen_polygon(coordinates)
            perimeter_lines_contour = self.load_perimeter(coordinates, wall_lines)
            for perimeter_line_contour in perimeter_lines_contour:
                if perimeter_line_contour in self._perimeter_lines:
                    self._perimeter_lines.remove(perimeter_line_contour)
            perimeter_lines_contours.append(perimeter_lines_contour)
            polygonized.append((area, coordinates))

        contours, _ = cv2.findContours(canvas_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        external_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        external_contour_normalized = self._smoothen_polygon(external_contour.reshape(-1, 2).tolist())

        return polygonized, perimeter_lines_contours, external_contour_normalized

    def recompute_dimensions_walls_and_polygons(self, walls_JSON, polygons_JSON, pixel_aspect_ratio_new, floor_plan_pdf_path):
        width, height = convert_from_path(floor_plan_pdf_path, dpi=400)[0].size
        scale_x = 1920 / width
        scale_y = 1080 / height
        walls_JSON_updated, polygons_JSON_updated = list(), list()
        for wall in walls_JSON:
            X1, Y1, X2, Y2 = wall["wall_line"][0]['x'], wall["wall_line"][0]['y'], wall["wall_line"][1]['x'], wall["wall_line"][1]['y']
            X1_scaled, Y1_scaled, X2_scaled, Y2_scaled = scale_x * X1, scale_y * Y1, scale_x * X2, scale_y * Y2
            length_target = round(math.hypot(
                (X1_scaled - X2_scaled) * pixel_aspect_ratio_new["horizontal"],
                (Y1_scaled - Y2_scaled) * pixel_aspect_ratio_new["vertical"]
            ), 2)
            wall["length"] = length_target
            walls_JSON_updated.append(wall)
        for polygon in polygons_JSON:
            polygon_vertices_scaled = [(scale_x * vertex[0], scale_y * vertex[1]) for vertex in polygon["vertices"]]
            polygon_vertices_scaled = np.array(polygon_vertices_scaled, np.float32)
            area = cv2.contourArea(polygon_vertices_scaled)
            polygon["area"] = pixel_aspect_ratio_new["area"] * area
            polygons_JSON_updated.append(polygon)

        return walls_JSON_updated, polygons_JSON_updated

    def compute_sloped_area_polygon(self, area, slope):
        if slope is None or slope == 0:
            return area
        theta = math.radians(slope)
        return round(area / math.cos(theta), 2)
