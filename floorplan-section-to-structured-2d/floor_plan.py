from copy import deepcopy
from collections import defaultdict
import re

from fractions import Fraction
import math
import numpy as np
import cv2
from scipy.spatial import cKDTree
import xml.etree.ElementTree as ET
from scipy import stats
from PIL import Image
import subprocess
from pathlib import Path

__all__ = ["FloorPlan"]


class FloorPlan:

    scales_architectural = [
        "3/64``=1`0``",
        "1/32``=1`0``",
        "1/16``=1`0``",
        "3/32``=1`0``",
        "1/8``=1`0``",
        "3/16``=1`0``",
        "1/4``=1`0``",
        "3/8``=1`0``",
        "1/2``=1`0``",
        "3/4``=1`0``",
        "1``=1`0``"
    ]

    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters
        self.tolerance_angle = self.hyperparameters["modelling"]["tolerance_angle"]
        self._lines_classified = dict()
        self._perimeter_lines = list()

    def read_floor_plan(self, image_path, resize=None):
        image = cv2.imread(image_path).copy()
        if resize:
            image = cv2.resize(image, resize)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def compute_imperial_scale_from_DPI(self, architectural_scale_normalized):
        DPI = self.hyperparameters["modelling"]["scale_adoption"]["dpi"]
        scale_on_paper_length = float(architectural_scale_normalized.split(':')[0].strip('`'))
        imperial = DPI * scale_on_paper_length
        return (1 / imperial, 1 / imperial)

    @classmethod
    def is_none(cls, image_path):
        return cv2.imread(image_path) is None

    @classmethod
    def normalize_scale(cls, scale):
        scale_on_paper_length = round(float(scale.split(':')[0].strip('`')), 2)
        for scale_architecture in cls.scales_architectural:
            if round(float(Fraction(scale_architecture.split('=')[0].strip('`'))), 2) == scale_on_paper_length:
                return scale_architecture

    @classmethod
    def scale_canonical(cls, scale):
        if scale.find(':') != -1:
            on_paper, real_world = scale.split(':')
        if scale.find('=') != -1:
            on_paper, real_world = scale.split('=')
        return f"{round(float(Fraction(on_paper.strip('`"'))), 2)}``:{real_world.replace("'", '`').replace('"', "``")}"        

    @classmethod
    def scale_to(
        cls,
        floor_plan_path="/tmp/floor_plan.png",
        pdf_path="/tmp/scaled_floor_plan.pdf",
        svg_path="/tmp/scaled_floor_plan.svg",
        resolution=None
    ):
        canvas = Image.open(floor_plan_path)
        width_in_pixels, height_in_pixels = canvas.size
        if canvas.mode != "RGB":
            canvas = canvas.convert("RGB")

        if resolution:
            canvas = canvas.resize(resolution, Image.Resampling.LANCZOS)
            width_in_pixels, height_in_pixels = resolution

        canvas.save(pdf_path, save_all=True)

        subprocess.run(
            ["pdftocairo", "-svg", pdf_path, svg_path],
            check=True
        )
        tree = ET.parse(svg_path)
        root = tree.getroot()
        width_in_points = root.attrib.get("width")
        height_in_points = root.attrib.get("height")
        root.set("width", "100%")
        root.set("height", "100%")
        if not root.get("preserveAspectRatio"):
            root.set("preserveAspectRatio", "xMidYMid meet")
        tree.write(svg_path, encoding="utf-8", xml_declaration=True)

        return Path(svg_path), dict(
            height_in_pixels=height_in_pixels,
            width_in_pixels=width_in_pixels,
            height_in_points=height_in_points,
            width_in_points=width_in_points,
            size=Path(svg_path).stat().st_size
        )

    def compute_pixel_aspect_ratio(self, scale_new, pixel_aspect_ratio_standard):
        scale_new_on_paper_length = float(scale_new.split(':')[0].strip('`'))
        scale_new_real_world_length_in_feet_and_inches_ = scale_new.split(':')[1].strip('`"')
        if scale_new_real_world_length_in_feet_and_inches_.find('`') != -1:
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
        pixel_aspect_ratio_new["area"] = scale * scale * pixel_aspect_ratio_standard["area"]

        return pixel_aspect_ratio_new

    def detect_lines(self, image_GRAY, offset=None, scale=None, floor_plan_path=None):
        lines = cv2.HoughLinesP(
            image_GRAY,
            **self.hyperparameters["modelling"]["HoughLinesTransformation"]
        )
        lines = self.normalize(lines)
        print(lines)
        if offset and lines is not None:
            canvas = cv2.imread(floor_plan_path)
            height_in_pixels, width_in_pixels, _ = canvas.shape
            margin_X, margin_Y = 10 * round(width_in_pixels / 1920), 10 * round(height_in_pixels / 1080)
            (offset_top_left_X, offset_top_left_Y), (offset_bottom_right_X, offset_bottom_right_Y) = offset
            LEFT = round(offset_top_left_X * width_in_pixels)
            TOP = round(offset_top_left_Y * height_in_pixels)
            BOTTOM = round(offset_bottom_right_Y * height_in_pixels)
            RIGHT = round(offset_bottom_right_X * width_in_pixels)
            LEFT_x = max(0, LEFT - margin_X)
            TOP_y = max(0, TOP - margin_Y)
            RIGHT_x = min(width_in_pixels, RIGHT + margin_X)
            BOTTOM_y = min(height_in_pixels, BOTTOM + margin_Y)
            lines_offset_bound = list()
            scale_x, scale_y = scale
            for line in lines:
                X1, Y1, X2, Y2 = line[0]
                X1_normalized, Y1_normalized, X2_normalized, Y2_normalized = round(scale_x * X1), round(scale_y * Y1), round(scale_x * X2), round(scale_y * Y2)
                if X1_normalized >= LEFT_x and Y1_normalized >= TOP_y and X2_normalized <= RIGHT_x and Y2_normalized <= BOTTOM_y:
                    lines_offset_bound.append(line)
            return lines_offset_bound
        return lines

    def normalize_transcription(self, text_corpus):
        text_corpus = text_corpus.lower()

        text_corpus = text_corpus.replace("”", '"')
        text_corpus = text_corpus.replace("“", '"')
        text_corpus = text_corpus.replace("′", "'")
        text_corpus = text_corpus.replace("`", "'")

        text_corpus = text_corpus.replace("o\"", '0"')
        text_corpus = text_corpus.replace("o'", "0'")
        text_corpus = text_corpus.replace("i/", "1/")
        text_corpus = text_corpus.replace("l/", "1/")

        text_corpus = re.sub(r"\s+", " ", text_corpus)

        return text_corpus.strip()

    def load_scale_from_text(self, text_corpus):
        text_corpus_normalized = self.normalize_transcription(text_corpus)
        regex_scale = re.compile(self.hyperparameters["modelling"]["scale_regex"], re.IGNORECASE | re.VERBOSE)
        scale = regex_scale.search(text_corpus_normalized)
        if not scale:
            return
        scale_identified = scale.group(1)
        if scale_identified:
            if scale_identified.find(':') != -1:
                scale_on_paper_length = scale_identified.split(':')[0].strip("'`\"")
            elif scale_identified.find('=') != -1:
                scale_on_paper_length = scale_identified.split('=')[0].strip("'`\"")
            else:
                scale_on_paper_length = scale_identified.split()[0].strip("'`\"")
            for scale_architecture in self.scales_architectural:
                try:
                    if round(float(Fraction(scale_architecture.split('=')[0].strip('`'))), 2) == round(float(Fraction(scale_on_paper_length)), 2):
                        return scale_architecture
                except ValueError:
                    return

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

    def sample_wall_line_test_points(self, X1, Y1, X2, Y2, test_samples=5):
        sampled_points = list()

        for i in range(1, test_samples + 1):
            t = i / (test_samples + 1)

            x = X1 + t * (X2 - X1)
            y = Y1 + t * (Y2 - Y1)

            sampled_points.append((x, y))

        return sampled_points

    def classify_line(self, x1, y1, x2, y2):
        """Classify a line as horizontal, vertical, or inclined."""
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
            distance_origin_A = np.hypot(X1 - 0, Y1 - 0)
            distance_origin_B = np.hypot(X2 - 0, Y2 - 0)
            if distance_origin_A <= distance_origin_B and [[X1, Y1, X2, Y2]] not in normalized_lines:
                normalized_lines.append([[X1, Y1, X2, Y2]]) 
            elif distance_origin_A > distance_origin_B and [[X2, Y2, X1, Y1]] not in normalized_lines:
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

    def nearest_neighbor(self, reference_line, end_type, target_lines, tolerance=500, top_k=None):
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

        if top_k:
            id_to_distance = {wall_line_index: distance for wall_line_index, distance in id_to_distance.items() if distance <= tolerance}
            sorted_items = sorted(id_to_distance.items(), key=lambda x: x[1])
            nearest_neighbors = [
                id_to_line[item[0]] for item in sorted_items[:top_k]]

            return nearest_neighbors

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

    def vertex_intersects_segment(self, x, y, x1, y1, x2, y2, threshold):
        dx = x2 - x1
        dy = y2 - y1

        seg_len_sq = dx * dx + dy * dy

        if seg_len_sq == 0:
            return math.hypot(x - x1, y - y1) <= threshold

        t = ((x - x1) * dx + (y - y1) * dy) / seg_len_sq

        if 0 <= t <= 1:
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
        else:
            if t < 0:
                proj_x, proj_y = x1, y1
            else:
                proj_x, proj_y = x2, y2

        distance = math.hypot(x - proj_x, y - proj_y)

        return distance <= threshold

    def load_perimeter(self, coordinates, wall_lines, tolerance=10, bound_capture=True, scale=None):
        tolerance_x, tolerance_y = tolerance, tolerance
        if scale:
            scale_x, scale_y = scale
            tolerance *= np.mean([scale[0], scale[1]])
            tolerance_x *= scale_x
            tolerance_y *= scale_y
        perimeter_lines = list()
        perimeter_lines_unbound = list()
        for source_coordinate in coordinates:
            for target_coordinate in coordinates:
                if math.hypot(target_coordinate[0] - source_coordinate[0], target_coordinate[1] - source_coordinate[1]) == 0:
                    continue
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
                        if abs(np.median([Y1, Y2]) - np.median([target_Y1, target_Y2])) <= tolerance_y and target_X1 - X1 >= -tolerance_x and target_X2 - X2 <= tolerance_x:
                            perimeter_segments.append(wall_line)
                        if abs(np.median([Y1, Y2]) - np.median([target_Y1, target_Y2])) <= tolerance_y and X1 - target_X1 >= -tolerance_x and X2 - target_X2 <= tolerance_x:
                            perimeter_lines_unbound.append(wall_line)
                    if orientation == "vertical" and orientation_target == "vertical":
                        if abs(np.median([X1, X2]) - np.median([target_X1, target_X2])) <= tolerance_x and target_Y1 - Y1 >= -tolerance_y and target_Y2 - Y2 <= tolerance_y:
                            perimeter_segments.append(wall_line)
                        if abs(np.median([X1, X2]) - np.median([target_X1, target_X2])) <= tolerance_x and Y1 - target_Y1 >= -tolerance_y and Y2 - target_Y2 <= tolerance_y:
                            perimeter_lines_unbound.append(wall_line)
                    if orientation == "inclined" and orientation_target == "inclined":
                        dx = X2 - X1
                        dy = Y2 - Y1
                        length = math.hypot(dx, dy)

                        if length > 0:
                            ux = dx / length
                            uy = dy / length

                            target_dx = target_X2 - target_X1
                            target_dy = target_Y2 - target_Y1
                            target_length = math.hypot(target_dx, target_dy)

                            if target_length > 0:

                                cross = abs(
                                    ux * (target_dy / target_length)
                                    - uy * (target_dx / target_length)
                                )

                                if cross <= 0.1:

                                    def point_line_distance(px, py):
                                        return abs(
                                            (px - X1) * uy -
                                            (py - Y1) * ux
                                        )

                                    d1 = point_line_distance(target_X1, target_Y1)
                                    d2 = point_line_distance(target_X2, target_Y2)

                                    if d1 <= tolerance and d2 <= tolerance:

                                        ref_start = X1 * ux + Y1 * uy
                                        ref_end = X2 * ux + Y2 * uy

                                        tgt_start = target_X1 * ux + target_Y1 * uy
                                        tgt_end = target_X2 * ux + target_Y2 * uy

                                        ref_min = min(ref_start, ref_end)
                                        ref_max = max(ref_start, ref_end)

                                        tgt_min = min(tgt_start, tgt_end)
                                        tgt_max = max(tgt_start, tgt_end)

                                        if (
                                            tgt_min >= ref_min - tolerance
                                            and tgt_max <= ref_max + tolerance
                                        ):
                                            perimeter_segments.append(wall_line)
                                        if (
                                            ref_min >= tgt_min - tolerance
                                            and ref_max <= tgt_max + tolerance
                                        ):
                                            perimeter_lines_unbound.append(wall_line)

                    if abs(target_X1 - X1) <= tolerance_x and abs(target_Y1 - Y1) <= tolerance_y and abs(target_X2 - X2) <= tolerance_x and abs(target_Y2 - Y2) <= tolerance_y:
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

        return perimeter_lines if perimeter_lines else perimeter_lines_unbound

    def load_perimeter_(self, coordinates, wall_lines):
        perimeter_lines =list()
        for wall_line in wall_lines:
            X1, Y1, X2, Y2 = wall_line[0]
            if self.classify_line(X1, Y1, X2, Y2) == "horizontal":
                centroid_perimeter_line = (round((X1 + X2) / 2), round(np.median([Y1, Y2])))
                if self.is_inside_polygon((centroid_perimeter_line[0], centroid_perimeter_line[1] - 100), coordinates) or self.is_inside_polygon((centroid_perimeter_line[0], centroid_perimeter_line[1] + 100), coordinates):
                    perimeter_lines.append(wall_line)
            if self.classify_line(X1, Y1, X2, Y2) == "vertical":
                centroid_perimeter_line = (round(np.median([X1, X2])), round((Y1 + Y2) / 2))
                if self.is_inside_polygon((centroid_perimeter_line[0] - 100, centroid_perimeter_line[1]), coordinates) or self.is_inside_polygon((centroid_perimeter_line[0] + 100, centroid_perimeter_line[1]), coordinates):
                    perimeter_lines.append(wall_line)
            if self.classify_line(X1, Y1, X2, Y2) == "inclined":
                dx = X2 - X1
                dy = Y2 - Y1
                length = math.hypot(dx, dy)

                nx = -dy / length
                ny =  dx / length

                mx = (X1 + X2) / 2
                my = (Y1 + Y2) / 2

                if self.is_inside_polygon((round(mx + nx * 100), round(my + ny * 100)), coordinates) or self.is_inside_polygon((round(mx - nx * 100), round(my - ny * 100)), coordinates):
                    perimeter_lines.append(wall_line)

        return perimeter_lines

    def load_perimeter_from_smoothened_polygon(self, coordinates, wall_lines, tolerance=100):
        perimeter_lines = list()
        polygon_smoothened = set()
        for wall_line in wall_lines:
            X1, Y1, X2, Y2 = wall_line[0]
            for coordinate in coordinates:
                if self.vertex_intersects_segment(coordinate[0], coordinate[1], X1, Y1, X2, Y2, tolerance):
                    if wall_line not in perimeter_lines:
                        perimeter_lines.append(wall_line)
                    if all([math.hypot(X1 - polygon_coordinate[0], Y1 - polygon_coordinate[1]) > 10 for polygon_coordinate in list(polygon_smoothened)]):
                        polygon_smoothened.add((X1, Y1))
                    if all([math.hypot(X2 - polygon_coordinate[0], Y2 - polygon_coordinate[1]) > 10 for polygon_coordinate in list(polygon_smoothened)]):
                        polygon_smoothened.add((X2, Y2))

        polygon_smoothened = list(polygon_smoothened)
        centroid_x = sum(coordinate[0] for coordinate in polygon_smoothened) / len(polygon_smoothened)
        centroid_y = sum(coordinate[1] for coordinate in polygon_smoothened) / len(polygon_smoothened)
        polygon_smoothened = sorted(polygon_smoothened, key=lambda coordinate: math.atan2(coordinate[1] - centroid_y, coordinate[0] - centroid_x))

        return perimeter_lines, polygon_smoothened

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
        print(wall_lines)
        canvas = np.ones((1080, 1920), dtype=np.uint8) * 255
        for wall_line in wall_lines:
            X1, Y1, X2, Y2 = wall_line[0]
            cv2.line(canvas, (X1, Y1), (X2, Y2), (0, 0, 0), 1)
        _, canvas_binary = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        canvas_dilated = cv2.dilate(canvas_binary, kernel, iterations=1)
        canvas_eroded = cv2.erode(canvas_dilated, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(canvas_eroded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        polygonized = list()
        self._perimeter_lines = wall_lines * 2
        perimeter_lines_contours = list()
        for contour, component in zip(contours, hierarchy[0]):
            if component[3] == -1:
                continue
            area = cv2.contourArea(contour)

            epsilon = max(2, 0.005 * cv2.arcLength(contour, True))
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

        print(polygonized)
        if not polygonized:
            return polygonized, perimeter_lines_contours, list()
        coordinates_all = np.vstack([polygon[1] for polygon in polygonized])
        hull_external = cv2.convexHull(coordinates_all)
        epsilon = max(2, 0.005 * cv2.arcLength(hull_external, True))
        external_contour = cv2.approxPolyDP(hull_external, epsilon, True)
        external_contour_normalized = self._smoothen_polygon(external_contour.reshape(-1, 2).tolist())

        return polygonized, perimeter_lines_contours, external_contour_normalized

    def merge_polygons(self, polygon_master, polygons_to_intersect):
        coordinates_all = np.vstack([polygon_master] + polygons_to_intersect)
        hull_external = cv2.convexHull(coordinates_all)
        epsilon = max(2, 0.005 * cv2.arcLength(hull_external, True))
        external_contour = cv2.approxPolyDP(hull_external, epsilon, True)
        external_contour_normalized = self._smoothen_polygon(external_contour.reshape(-1, 2).tolist())
        return external_contour_normalized

    def topology_guided_endpoint_snapping(self, lines):
        def nearest_endpoint(reference_line, end_type, target_line):
            X1_reference, Y1_reference, X2_reference, Y2_reference = reference_line[0]
            X1_target, Y1_target, X2_target, Y2_target = target_line[0]
            if end_type == 'A':
                if math.hypot(X1_reference - X1_target, Y1_reference - Y1_target) < math.hypot(X1_reference - X2_target, Y1_reference - Y2_target):
                    return (X1_target, Y1_target)
                return (X2_target, Y2_target)
            if end_type == 'B':
                if math.hypot(X2_reference - X1_target, Y2_reference - Y1_target) < math.hypot(X2_reference - X2_target, Y2_reference - Y2_target):
                    return (X1_target, Y1_target)
                return (X2_target, Y2_target)

        lines_snapped = list()
        for line in lines:
            X1, Y1, X2, Y2 = line[0]
            orientation = self.classify_line(*line[0])
            for end_type in ['A', 'B']:
                nearest_neighbors = self.nearest_neighbor(line, end_type, lines, tolerance=20, top_k=3)
                if nearest_neighbors:
                    nearest_neighbor = nearest_neighbors[-1]
                    endpoint_X, endpoint_Y = nearest_endpoint(line, end_type, nearest_neighbor)
                    if orientation == "horizontal":
                        if end_type == 'A':
                            line_target = [[endpoint_X, Y1, X2, Y2]]
                            #lines.remove(line)
                            lines_snapped.append(line_target)
                            X1, Y1, X2, Y2 = line_target[0]
                        if end_type == 'B':
                            line_target = [[X1, Y1, endpoint_X, Y2]]
                            #lines.remove(line)
                            lines_snapped.append(line_target)
                            X1, Y1, X2, Y2 = line_target[0]
                    if orientation == "vertical":
                        if end_type == 'A':
                            line_target = [[X1, endpoint_Y, X2, Y2]]
                            #lines.remove(line)
                            lines_snapped.append(line_target)
                            #line = line_target
                            X1, Y1, X2, Y2 = line_target[0]
                        if end_type == 'B':
                            line_target = [[X1, Y1, X2, endpoint_Y]]
                            #lines.remove(line)
                            lines_snapped.append(line_target)
                            #line = line_target
                            X1, Y1, X2, Y2 = line_target[0]
                    if orientation == "inclined":
                        dx = X2 - X1
                        dy = Y2 - Y1
                        norm = math.hypot(dx, dy)

                        if norm == 0:
                            continue

                        ux = dx / norm
                        uy = dy / norm

                        if end_type == 'A':
                            t = (endpoint_X - X1) * ux + (endpoint_Y - Y1) * uy
                            new_X1 = int(round(X1 + t * ux))
                            new_Y1 = int(round(Y1 + t * uy))

                            line_target = [[new_X1, new_Y1, X2, Y2]]
                            #lines.remove(line)
                            lines_snapped.append(line_target)
                            #line = line_target
                            X1, Y1, X2, Y2 = line_target[0]

                        if end_type == 'B':
                            t = (endpoint_X - X2) * ux + (endpoint_Y - Y2) * uy
                            new_X2 = int(round(X2 + t * ux))
                            new_Y2 = int(round(Y2 + t * uy))

                            line_target = [[X1, Y1, new_X2, new_Y2]]
                            #lines.remove(line)
                            lines_snapped.append(line_target)
                            #line = line_target
                            X1, Y1, X2, Y2 = line_target[0]
                else:
                    if line not in lines_snapped:
                        lines_snapped.append(line)

        return lines_snapped

    def lines_to_topology(self, lines):
        def grid_key(p, tolerance=20):
            return (int(p[0] // tolerance), int(p[1] // tolerance))

        def cluster_nodes(nodes, tolerance=20):
            buckets = defaultdict(list)

            for node in nodes:
                buckets[grid_key(node)].append(node)

            clusters = list()
            visited = set()

            for bucket in buckets.values():
                for p in bucket:
                    if p in visited:
                        continue

                    cluster = [p]
                    visited.add(p)

                    for q in bucket:
                        if q not in visited:
                            if math.hypot(p[0]-q[0], p[1]-q[1]) <= tolerance:
                                cluster.append(q)
                                visited.add(q)

                    clusters.append(cluster)

            return clusters

        def dominant_direction(cluster):
            cluster = clean_cluster(cluster)

            if len(cluster) < 2:
                return np.array([1.0, 0.0])

            points = np.array(cluster, dtype=np.float64)

            if np.all(points == points[0]):
                return np.array([1.0, 0.0])

            cov = np.cov(points.T)

            if not np.all(np.isfinite(cov)):
                return np.array([1.0, 0.0])

            eigvals, eigvecs = np.linalg.eig(cov)

            return eigvecs[:, np.argmax(eigvals)]

        def clean_cluster(cluster):
            clean = list()
            for x, y in cluster:
                if np.isfinite(x) and np.isfinite(y):
                    clean.append((x, y))
            return clean

        def representative_point(cluster, representation_type="dominant"):
            cluster = clean_cluster(cluster)

            if len(cluster) == 0:
                return (0, 0)

            if len(cluster) == 1:
                return cluster[0]

            if representation_type == "mode":
                x = stats.mode([p[0] for p in cluster]).mode
                y = stats.mode([p[1] for p in cluster]).mode
                return (int(round(x)), int(round(y)))

            if representation_type == "median":
                x = np.median([p[0] for p in cluster])
                y = np.median([p[1] for p in cluster])
                return (int(round(x)), int(round(y)))

            direction = dominant_direction(cluster)

            points = np.array(cluster)
            projections = points @ direction
            idx = np.argmax(projections)

            return tuple(points[idx].astype(int))

        def build_node_mapping(clusters):
            mapping = dict()
            for cluster in clusters:
                representative = representative_point(cluster)
                for p in cluster:
                    mapping[p] = representative
            return mapping

        def remap_edges(edges, mapping):
            new_edges = set()

            for (p1, p2) in edges:
                np1 = mapping[p1]
                np2 = mapping[p2]

                if np1 != np2:
                    new_edges.add(tuple(sorted([np1, np2])))

            return list(new_edges)

        nodes = list()
        edges = list()
        for line in lines:
            X1, Y1, X2, Y2 = line[0]
            nodes.extend([(X1, Y1), (X2, Y2)])
            edges.append([(X1, Y1), (X2, Y2)])

        clusters = cluster_nodes(nodes)
        mapping = build_node_mapping(clusters)
        edges_clean = remap_edges(edges, mapping)
        lines_clean = list()
        for edge in edges_clean:
            X1, Y1, X2, Y2 = edge[0][0], edge[0][1], edge[1][0], edge[1][1]
            lines_clean.append([[X1, Y1, X2, Y2]])
        lines_clean = self.normalize(lines_clean)
        return lines_clean
