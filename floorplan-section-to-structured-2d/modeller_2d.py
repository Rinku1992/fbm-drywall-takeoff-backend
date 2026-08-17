from typing import Tuple
from copy import deepcopy
import cv2
import numpy as np
np.random.seed(0)
import math
import json
import logging
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from fractions import Fraction
from skimage.morphology import skeletonize
from vertexai.generative_models import Part, Content
from shapely.geometry import Polygon
import Levenshtein

from floor_plan import FloorPlan
from prompts import (
    POLYGON_DETECTOR_AND_DRYWALL_PREDICTOR,
    POLYGON_DETECTOR_AND_DRYWALL_PREDICTOR_CALIFORNIA_FEW_SHOT,
    POLYGON_DETECTOR_AND_DRYWALL_PREDICTOR_CUSTOM,
    DRYWALL_PREDICTOR,
    SCALE_AND_CEILING_HEIGHT_DETECTOR,
    SCALE_DETECTOR,
    CEILING_HEIGHT_DETECTOR,
    WALL_RECTIFIER,
    WALL_RECTIFIER_FEW_SHOT,
    SHAPE_RECTIFIER,
    SHAPE_RECTIFIER_FEW_SHOT,
    CEILING_CHOICES,
    WALL_CHOICES,
    OPENING_TYPE_CHOICES,
    PolygonDetectorAndDrywallPredictorResponse,
    ScaleAndCeilingHeightDetectorResponse,
    ScaleDetectorResponse,
    CeilingHeightDetectorResponse,
    WallRectifierResponse,
    ShapeRectifierResponse,
    DrywallAssemblyCeiling,
    CeilingModelAndPredict,
    DrywallAssemblyWall,
    WallParameterModelAndPredict,
    prune_model,
    load_schema_pydantic,
)
from helper import (
    load_vertex_ai_client,
    phoenix_call,
)
from rate_limiter import RateLimiter

__all__ = ["FloorPlan2D"]


class FloorPlan2D(FloorPlan):

    def __init__(self, credentials, hyperparameters, drywall_templates, project_location, section_name=None):
        super().__init__(hyperparameters)

        self._credentials = credentials
        self._hyperparameters = hyperparameters
        self._drywall_templates = drywall_templates
        self._project_location = project_location
        self._section_name = section_name
        self._width_in_feet = self._hyperparameters["modelling"]["width_in_feet"]
        self._height_in_feet = self._hyperparameters["modelling"]["height_in_feet"]
        self._scale = self._hyperparameters["modelling"]["scale"]
        self._is_scale_detected = False
        self._imperial_scales_sampled = dict(X=list(), Y=list(), A=list())
        self._walls_2d = list()
        self._polygons = list()
        self._llm_rate_limiter = RateLimiter(3)

    def reload(self, section_name):
        self._walls_2d = list()
        self._polygons = list()
        self._section_name = section_name
        self._is_scale_detected = False

    @property
    def is_scale_detected(self):
        return self._is_scale_detected

    @classmethod
    def load_vertex_ai_clients(cls, credentials, client_ip_address, drywall_templates, project_location):
        is_cached = dict()
        vertex_ai_client_polygon_detection_and_drywall_prediction, generation_config, cache_enabled = load_vertex_ai_client(
            credentials,
            client_ip_address,
            prompts=[POLYGON_DETECTOR_AND_DRYWALL_PREDICTOR.format(drywall_templates=drywall_templates, project_location=project_location)]
        )
        is_cached["POLYGON_DETECTOR_AND_DRYWALL_PREDICTOR"] = cache_enabled
        vertex_ai_client_polygon_detection_and_drywall_prediction_custom, generation_config, cache_enabled = load_vertex_ai_client(
            credentials,
            client_ip_address,
            prompts=[POLYGON_DETECTOR_AND_DRYWALL_PREDICTOR_CUSTOM.format(drywall_templates=drywall_templates, project_location=project_location)]
        )
        is_cached["POLYGON_DETECTOR_AND_DRYWALL_PREDICTOR_CUSTOM"] = cache_enabled
        vertex_ai_client_metadata_extraction, _, cache_enabled = load_vertex_ai_client(
            credentials,
            client_ip_address,
            prompts=[SCALE_AND_CEILING_HEIGHT_DETECTOR.format(supported_scales_architectural=cls.scales_architectural)]
        )
        is_cached["SCALE_AND_CEILING_HEIGHT_DETECTOR"] = cache_enabled
        vertex_ai_client_metadata_scale_extraction, _, cache_enabled = load_vertex_ai_client(
            credentials,
            client_ip_address,
            prompts=[SCALE_DETECTOR.format(supported_scales_architectural=cls.scales_architectural)]
        )
        is_cached["SCALE_DETECTOR"] = cache_enabled
        vertex_ai_client_metadata_ceiling_height_extraction, _, cache_enabled = load_vertex_ai_client(
            credentials,
            client_ip_address,
            prompts=[CEILING_HEIGHT_DETECTOR]
        )
        is_cached["CEILING_HEIGHT_DETECTOR"] = cache_enabled
        vertex_ai_client_drywall_prediction, generation_config, cache_enabled = load_vertex_ai_client(
            credentials,
            client_ip_address,
            prompts=[DRYWALL_PREDICTOR.format(drywall_templates=drywall_templates, project_location=project_location)]
        )
        is_cached["DRYWALL_PREDICTOR"] = cache_enabled
        vertex_ai_client_wall_rectification, _, cache_enabled = load_vertex_ai_client(
            credentials,
            client_ip_address,
            prompts=[WALL_RECTIFIER]
        )
        is_cached["WALL_RECTIFIER"] = cache_enabled
        vertex_ai_client_shape_rectification, _, cache_enabled = load_vertex_ai_client(
            credentials,
            client_ip_address,
            prompts=[SHAPE_RECTIFIER]
        )
        is_cached["SHAPE_RECTIFIER"] = cache_enabled
        vertex_ai_clients = (
            vertex_ai_client_polygon_detection_and_drywall_prediction,
            vertex_ai_client_polygon_detection_and_drywall_prediction_custom,
            vertex_ai_client_metadata_extraction,
            vertex_ai_client_metadata_scale_extraction,
            vertex_ai_client_metadata_ceiling_height_extraction,
            vertex_ai_client_drywall_prediction,
            vertex_ai_client_wall_rectification,
            vertex_ai_client_shape_rectification
        )
        return is_cached, vertex_ai_clients, generation_config

    def from_vertex_ai_clients(self, is_cached, vertex_ai_clients, generation_config):
        self._vertex_ai_generation_config = generation_config
        self._is_cached = is_cached
        self._vertex_ai_client_polygon_detection_and_drywall_prediction = vertex_ai_clients[0]
        self._vertex_ai_client_polygon_detection_and_drywall_prediction_custom = vertex_ai_clients[1]
        self._vertex_ai_client_metadata_extraction = vertex_ai_clients[2]
        self._vertex_ai_client_metadata_scale_extraction = vertex_ai_clients[3]
        self._vertex_ai_client_metadata_ceiling_height_extraction = vertex_ai_clients[4]
        self._vertex_ai_client_drywall_prediction = vertex_ai_clients[5]
        self._vertex_ai_client_wall_rectification = vertex_ai_clients[6]
        self._vertex_ai_client_shape_rectification = vertex_ai_clients[7]

    def _close_jagged_openings(
        self,
        wall_lines,
        tolerance_distance=5,
    ):
        def load_wall_line_index_random(n_wall_lines, wall_line_indices_prior):
            while True:
                wall_line_index = np.random.randint(n_wall_lines)
                if wall_line_index in wall_line_indices_prior:
                    continue
                break
            return wall_line_index

        if not wall_lines:
            return wall_lines
        try:
            wall_lines = wall_lines.tolist()
        except:
            ...

        wall_line_indices_prior = list()
        for n_step in range(len(wall_lines)):
            wall_lines_new = deepcopy(wall_lines)
            if n_step >= len(wall_lines):
                wall_line_indices_prior = list()
            wall_line_index = load_wall_line_index_random(len(wall_lines), wall_line_indices_prior)
            wall_line_indices_prior.append(wall_line_index)
            reference_line = wall_lines[wall_line_index]
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
        clusters = list()
        for i in range(n):
            if i not in visited:
                stack = [i]
                cluster = list()
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
    ):
        def load_wall_line_index_random(n_wall_lines, wall_line_indices_prior):
            while True:
                wall_line_index = np.random.randint(n_wall_lines)
                if wall_line_index in wall_line_indices_prior:
                    continue
                break
            return wall_line_index

        if wall_lines is None:
            return wall_lines
        wall_line_indices_prior = list()
        for n_step in range(len(wall_lines)):
            wall_lines_new = deepcopy(wall_lines)
            if n_step >= len(wall_lines):
                wall_line_indices_prior = list()
            wall_line_index = load_wall_line_index_random(len(wall_lines), wall_line_indices_prior)
            wall_line_indices_prior.append(wall_line_index)
            reference_wall_line = wall_lines[wall_line_index]
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

    def _apply_morphological_closing(self, image_GRAY):
        edges_thinned = self._thin_edges(image_GRAY)
        edges = cv2.Canny(edges_thinned, 50, 100, apertureSize=3)

        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        return edges

    def _preprocessing(self, image_BGR, floor_plan_path, offset, scale, max_split=5):
        _, thresh = cv2.threshold(image_BGR, 50, 255, cv2.THRESH_BINARY_INV)
        closed = self._apply_morphological_closing(thresh)

        floor_plan_topology_binary = self._load_topology(closed)
        lines = self.detect_lines(thresh, offset=offset, scale=scale, floor_plan_path=floor_plan_path)
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

            lines = self._topology_guided_closure_open_lines_dead_end(lines, maximum_length=150)
            lines = self._sniff_and_split_orthogonal(lines)
            lines = self._deduplicate_lines(lines)
            lines = self._remove_invalid(lines)
            shapes = self.disconnected_shapes(lines)

            is_valid_futures = list()
            with ThreadPoolExecutor(max_workers=10) as executor:
                for shape in shapes:
                    is_valid_futures.append(executor.submit(
                        self._is_shape_valid,
                        shape,
                        scale,
                        floor_plan_path,
                    ))
                is_shape_valid = [future.result() for future in is_valid_futures]
            lines = list()
            for shape, is_valid in zip(shapes, is_shape_valid):
                if is_valid and len(shape) > 4:
                    lines.extend(self._merge_nearest_neighbor(shape))

        return lines

    def _is_shape_valid(self, lines, scale, floor_plan_path):
        scale_x, scale_y = scale
        wall_lines_structured = list()
        canvas = cv2.imread(floor_plan_path)
        for line in lines:
            X1, Y1, X2, Y2 = line[0]
            polygons_drywall = self._extrude_polygon_perimeter(line, scale, outer_drywall_surface="INVALID")
            for drywall in polygons_drywall:
                pts = np.array([
                    [drywall["coordinates"][0]['x'], drywall["coordinates"][0]['y']],
                    [drywall["coordinates"][1]['x'], drywall["coordinates"][1]['y']],
                    [drywall["coordinates"][2]['x'], drywall["coordinates"][2]['y']],
                    [drywall["coordinates"][3]['x'], drywall["coordinates"][3]['y']]
                ], np.int32)
                pts = pts.reshape((-1, 1, 2))
                canvas_to_overlay = canvas.copy()
                cv2.fillPoly(canvas_to_overlay, pts=[pts], color=(0, 0, 255))
                canvas = cv2.addWeighted(canvas_to_overlay, 0.7, canvas, 0.3, 0)
            canvas_to_overlay = canvas.copy()
            cv2.line(canvas_to_overlay, (round(scale_x * X1), round(scale_y * Y1)), (round(scale_x * X2), round(scale_y * Y2)), (0, 0, 255), 2)
            canvas = cv2.addWeighted(canvas_to_overlay, 0.7, canvas, 0.3, 0)
            wall_lines_structured.append(dict(wall=[{'x': round(scale_x * X1), 'y': round(scale_y * Y1)}, {'x': round(scale_x * X2), 'y': round(scale_y * Y2)}]))

        _, canvas_buffer_array = cv2.imencode(".png", canvas)
        bytes_canvas = canvas_buffer_array.tobytes()
        query = Content(role="user", parts=[
            Part.from_text(json.dumps(wall_lines_structured)),
            Part.from_data(data=bytes_canvas, mime_type="image/png")
        ])
        try:
            if self._is_cached["SHAPE_RECTIFIER"]:
                _, is_valid = phoenix_call(
                    lambda feedback_prompt, temperature: self._vertex_ai_client_shape_rectification.generate_content(
                        contents=SHAPE_RECTIFIER_FEW_SHOT+[feedback_prompt, query] if feedback_prompt else SHAPE_RECTIFIER_FEW_SHOT+[query],
                        generation_config={**self._vertex_ai_generation_config, "temperature": temperature},
                    ),
                    self._llm_rate_limiter,
                    max_retry=self._credentials["VertexAI"]["llm"]["max_retry"],
                    pydantic_model=ShapeRectifierResponse,
                )
            else:
                _, is_valid = phoenix_call(
                    lambda feedback_prompt, temperature: self._vertex_ai_client_shape_rectification(SHAPE_RECTIFIER).generate_content(
                        contents=SHAPE_RECTIFIER_FEW_SHOT+[feedback_prompt, query] if feedback_prompt else SHAPE_RECTIFIER_FEW_SHOT+[query],
                        generation_config={**self._vertex_ai_generation_config, "temperature": temperature},
                    ),
                    self._llm_rate_limiter,
                    max_retry=self._credentials["VertexAI"]["llm"]["max_retry"],
                    pydantic_model=ShapeRectifierResponse,
                )
            if is_valid["confidence"] > 0.9:
                return is_valid["is_valid"]
            return True
        except Exception as e:
            logging.warning(f"SYSTEM: Section: {self._section_name}, Wall validator failed with error: {e}")
            return True

    def _merge_nearest_neighbor(self, wall_lines, tolerance=500):
        wall_lines_closed_dead_end = deepcopy(wall_lines)
        for wall_line in wall_lines:
            X1, Y1, X2, Y2 = wall_line[0]
            open_ends = self.is_open(wall_line, wall_lines)
            if 'A' in open_ends:
                nearest_neighbor = self.nearest_neighbor(wall_line, 'A', wall_lines, tolerance=tolerance)
                if nearest_neighbor:
                    X1_nearest, Y1_nearest, X2_nearest, Y2_nearest = nearest_neighbor[0]
                    X1_new, Y1_new = X1, Y1
                    if math.hypot(X1_new - X1_nearest, Y1_new - Y1_nearest) < math.hypot(X1_new - X2_nearest, Y1_new - Y2_nearest):
                        wall_lines_closed_dead_end.append([[X1_new, Y1_new, X1_nearest, Y1_nearest]])
                    else:
                        wall_lines_closed_dead_end.append([[X1_new, Y1_new, X2_nearest, Y2_nearest]])
            if 'B' in open_ends:
                nearest_neighbor = self.nearest_neighbor(wall_line, 'B', wall_lines, tolerance=tolerance)
                if nearest_neighbor:
                    X1_nearest, Y1_nearest, X2_nearest, Y2_nearest = nearest_neighbor[0]
                    X1_new, Y1_new = X2, Y2
                    if math.hypot(X1_new - X1_nearest, Y1_new - Y1_nearest) < math.hypot(X1_new - X2_nearest, Y1_new - Y2_nearest):
                        wall_lines_closed_dead_end.append([[X1_new, Y1_new, X1_nearest, Y1_nearest]])
                    else:
                        wall_lines_closed_dead_end.append([[X1_new, Y1_new, X2_nearest, Y2_nearest]])

        return wall_lines_closed_dead_end

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
                dx = X2 - X1
                dy = Y2 - Y1
 
                norm = np.hypot(dx, dy)
                if norm == 0:
                    extended_lines.append([[X1, Y1, X2, Y2]])
                    continue
 
                ux = dx / norm
                uy = dy / norm

                for n_pixels in range(extension_maximum):
                    step = n_pixels + 1
 
                    X1_new = int(round(X1 - ux * step))
                    Y1_new = int(round(Y1 - uy * step))
 
                    if not (0 <= X1_new < 1920 and 0 <= Y1_new < 1080):
                        break
 
                    perp_dx = -uy
                    perp_dy = ux
 
                    valid = False
                    for t in range(-tolerance, tolerance + 1):
                        xs = int(round(X1_new + perp_dx * t))
                        ys = int(round(Y1_new + perp_dy * t))
 
                        if 0 <= xs < 1920 and 0 <= ys < 1080:
                            if floor_plan_topology_binary[ys, xs] == 0:
                                valid = True
                                break
 
                    if valid:
                        X1, Y1 = X1_new, Y1_new
                    else:
                        break
 
                for n_pixels in range(extension_maximum):
                    step = n_pixels + 1
 
                    X2_new = int(round(X2 + ux * step))
                    Y2_new = int(round(Y2 + uy * step))
 
                    if not (0 <= X2_new < 1920 and 0 <= Y2_new < 1080):
                        break
 
                    perp_dx = -uy
                    perp_dy = ux
 
                    valid = False
                    for t in range(-tolerance, tolerance + 1):
                        xs = int(round(X2_new + perp_dx * t))
                        ys = int(round(Y2_new + perp_dy * t))
 
                        if 0 <= xs < 1920 and 0 <= ys < 1080:
                            if floor_plan_topology_binary[ys, xs] == 0:
                                valid = True
                                break
 
                    if valid:
                        X2, Y2 = X2_new, Y2_new
                    else:
                        break
 
                extended_lines.append([[X1, Y1, X2, Y2]])

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
        canvas = np.ones((1080, 1920), dtype=np.uint8) * 255
        for wall_line in wall_lines:
            X1, Y1, X2, Y2 = wall_line[0]
            cv2.line(canvas, (X1, Y1), (X2, Y2), 0, 1)
        for wall_line in wall_lines:
            X1, Y1, X2, Y2 = wall_line[0]
            if math.hypot(X1 - X2, Y1 - Y2) <= length_tolerance_threshold:
                continue
            open_ends = self.is_open(wall_line, wall_lines, tolerance=open_tolerance_threshold)
            if open_ends and 'A' in open_ends and 'B' in open_ends:
                orientation = self.classify_line(X1, Y1, X2, Y2)
                if orientation == "horizontal":
                    is_extended = False
                    pixel_distance_UP_A = np.argmin(canvas[: Y1, X1 - open_tolerance_threshold: X1 + open_tolerance_threshold].mean(axis=1)[::-1])
                    pixel_value_UP_A = canvas[: Y1, X1 - open_tolerance_threshold: X1 + open_tolerance_threshold].mean(axis=1)[::-1][np.argmin(canvas[: Y1, X1 - open_tolerance_threshold: X1 + open_tolerance_threshold].mean(axis=1)[::-1])]
                    pixel_distance_UP_A = pixel_distance_UP_A if pixel_value_UP_A != 255 else np.inf
                    pixel_distance_DOWN_A = np.argmin(canvas[Y1 + 1:, X1 - open_tolerance_threshold: X1 + open_tolerance_threshold].mean(axis=1))
                    pixel_value_DOWN_A = canvas[Y1 + 1:, X1 - open_tolerance_threshold: X1 + open_tolerance_threshold].mean(axis=1)[np.argmin(canvas[Y1 + 1:, X1 - open_tolerance_threshold: X1 + open_tolerance_threshold].mean(axis=1))]
                    pixel_distance_DOWN_A = pixel_distance_DOWN_A if pixel_value_DOWN_A != 255 else np.inf
                    pixel_distance_LEFT_A = np.argmin(canvas[round(np.median([Y1, Y2])) - open_tolerance_threshold: round(np.median([Y1, Y2])) + open_tolerance_threshold, : X1].mean(axis=0)[::-1])
                    pixel_value_LEFT_A = canvas[round(np.median([Y1, Y2])) - open_tolerance_threshold: round(np.median([Y1, Y2])) + open_tolerance_threshold, : X1].mean(axis=0)[::-1][np.argmin(canvas[round(np.median([Y1, Y2])) - open_tolerance_threshold: round(np.median([Y1, Y2])) + open_tolerance_threshold, : X1].mean(axis=0)[::-1])]
                    pixel_distance_LEFT_A = pixel_distance_LEFT_A if pixel_value_LEFT_A != 255 else np.inf
                    if min(pixel_distance_UP_A, pixel_distance_DOWN_A, pixel_distance_LEFT_A) <= 500:
                        if np.argmin([pixel_distance_UP_A, pixel_distance_DOWN_A, pixel_distance_LEFT_A]) == 0 and pixel_value_UP_A != 255:
                            valid_wall_lines.append([[X1, Y1 - pixel_distance_UP_A, X1, Y1]])
                            is_extended = True
                        if np.argmin([pixel_distance_UP_A, pixel_distance_DOWN_A, pixel_distance_LEFT_A]) == 1 and pixel_value_DOWN_A != 255:
                            valid_wall_lines.append([[X1, Y1, X1, Y1 + pixel_distance_DOWN_A]])
                            is_extended = True
                        if np.argmin([pixel_distance_UP_A, pixel_distance_DOWN_A, pixel_distance_LEFT_A]) == 2 and pixel_value_LEFT_A != 255:
                            valid_wall_lines.append([[X1 - pixel_distance_LEFT_A, Y1, X1, Y1]])
                            is_extended = True
                    pixel_distance_UP_B = np.argmin(canvas[: Y2, X2 - open_tolerance_threshold: X2 + open_tolerance_threshold].mean(axis=1)[::-1])
                    pixel_value_UP_B = canvas[: Y2, X2 - open_tolerance_threshold: X2 + open_tolerance_threshold].mean(axis=1)[::-1][np.argmin(canvas[: Y2, X2 - open_tolerance_threshold: X2 + open_tolerance_threshold].mean(axis=1)[::-1])]
                    pixel_distance_UP_B = pixel_distance_UP_B if pixel_value_UP_B != 255 else np.inf
                    pixel_distance_DOWN_B = np.argmin(canvas[Y2 + 1:, X2 - open_tolerance_threshold: X2 + open_tolerance_threshold].mean(axis=1))
                    pixel_value_DOWN_B = canvas[Y2 + 1:, X2 - open_tolerance_threshold: X2 + open_tolerance_threshold].mean(axis=1)[np.argmin(canvas[Y2 + 1:, X2 - open_tolerance_threshold: X2 + open_tolerance_threshold].mean(axis=1))]
                    pixel_distance_DOWN_B = pixel_distance_DOWN_B if pixel_value_DOWN_B != 255 else np.inf
                    pixel_distance_RIGHT_B = np.argmin(canvas[round(np.median([Y1, Y2])) - open_tolerance_threshold: round(np.median([Y1, Y2])) + open_tolerance_threshold, X2 + 1:].mean(axis=0)[::-1])
                    pixel_value_RIGHT_B = canvas[round(np.median([Y1, Y2])) - open_tolerance_threshold: round(np.median([Y1, Y2])) + open_tolerance_threshold, X2 + 1:].mean(axis=0)[::-1][np.argmin(canvas[round(np.median([Y1, Y2])) - open_tolerance_threshold: round(np.median([Y1, Y2])) + open_tolerance_threshold, X2 + 1:].mean(axis=0)[::-1])]
                    pixel_distance_RIGHT_B = pixel_distance_RIGHT_B if pixel_value_RIGHT_B != 255 else np.inf
                    if min(pixel_distance_UP_B, pixel_distance_DOWN_B, pixel_distance_RIGHT_B) <= 500:
                        if np.argmin([pixel_distance_UP_B, pixel_distance_DOWN_B, pixel_distance_RIGHT_B]) == 0 and pixel_value_UP_B != 255:
                            valid_wall_lines.append([[X2, Y2 - pixel_distance_UP_B, X2, Y2]])
                            is_extended = True
                        if np.argmin([pixel_distance_UP_B, pixel_distance_DOWN_B, pixel_distance_RIGHT_B]) == 1 and pixel_value_DOWN_B != 255:
                            valid_wall_lines.append([[X2, Y2, X2, Y2 + pixel_distance_DOWN_B]])
                            is_extended = True
                        if np.argmin([pixel_distance_UP_B, pixel_distance_DOWN_B, pixel_distance_RIGHT_B]) == 2 and pixel_value_RIGHT_B != 255:
                            valid_wall_lines.append([[X2, Y2, X2 + pixel_distance_RIGHT_B, Y2]])
                            is_extended = True
                    if is_extended:
                        valid_wall_lines.append(wall_line)
                elif orientation == "vertical":
                    is_extended = False
                    pixel_distance_LEFT_A = np.argmin(canvas[Y1 - open_tolerance_threshold: Y1 + open_tolerance_threshold, :X1].mean(axis=0)[::-1])
                    pixel_value_LEFT_A = canvas[Y1 - open_tolerance_threshold: Y1 + open_tolerance_threshold, :X1].mean(axis=0)[::-1][np.argmin(canvas[Y1 - open_tolerance_threshold: Y1 + open_tolerance_threshold, :X1].mean(axis=0)[::-1])]
                    pixel_distance_LEFT_A = pixel_distance_LEFT_A if pixel_value_LEFT_A != 255 else np.inf
                    pixel_distance_RIGHT_A = np.argmin(canvas[Y1 - open_tolerance_threshold:Y1 + open_tolerance_threshold, X1 + 1:].mean(axis=0))
                    pixel_value_RIGHT_A = canvas[Y1 - open_tolerance_threshold:Y1 + open_tolerance_threshold, X1 + 1:].mean(axis=0)[np.argmin(canvas[Y1 - open_tolerance_threshold:Y1 + open_tolerance_threshold, X1 + 1:].mean(axis=0))]
                    pixel_distance_RIGHT_A = pixel_distance_RIGHT_A if pixel_value_RIGHT_A != 255 else np.inf
                    pixel_distance_UP_A = np.argmin(canvas[:Y1, X1 - open_tolerance_threshold:X1 + open_tolerance_threshold].mean(axis=1)[::-1])
                    pixel_value_UP_A = canvas[:Y1, X1 - open_tolerance_threshold:X1 + open_tolerance_threshold].mean(axis=1)[::-1][np.argmin(canvas[:Y1, X1 - open_tolerance_threshold:X1 + open_tolerance_threshold].mean(axis=1)[::-1])]
                    pixel_distance_UP_A = pixel_distance_UP_A if pixel_value_UP_A != 255 else np.inf
                    if min(pixel_distance_LEFT_A, pixel_distance_RIGHT_A, pixel_distance_UP_A) <= 500:
                        if np.argmin([pixel_distance_LEFT_A, pixel_distance_RIGHT_A, pixel_distance_UP_A]) == 0 and pixel_value_LEFT_A != 255:
                            valid_wall_lines.append([[X1 - pixel_distance_LEFT_A, Y1, X1, Y1]])
                            is_extended = True
                        if np.argmin([pixel_distance_LEFT_A, pixel_distance_RIGHT_A, pixel_distance_UP_A]) == 1 and pixel_value_RIGHT_A != 255:
                            valid_wall_lines.append([[X1, Y1, X1 + pixel_distance_RIGHT_A, Y1]])
                            is_extended = True
                        if np.argmin([pixel_distance_LEFT_A, pixel_distance_RIGHT_A, pixel_distance_UP_A]) == 2 and pixel_value_UP_A != 255:
                            valid_wall_lines.append([[X1, Y1 - pixel_distance_UP_A, X1, Y1]])
                            is_extended = True
                    pixel_distance_LEFT_B = np.argmin(canvas[Y2 - open_tolerance_threshold:Y2 + open_tolerance_threshold, :X2].mean(axis=0)[::-1])
                    pixel_value_LEFT_B = canvas[Y2 - open_tolerance_threshold:Y2 + open_tolerance_threshold, :X2].mean(axis=0)[::-1][np.argmin(canvas[Y2 - open_tolerance_threshold:Y2 + open_tolerance_threshold, :X2].mean(axis=0)[::-1])]
                    pixel_distance_LEFT_B = pixel_distance_LEFT_B if pixel_value_LEFT_B != 255 else np.inf
                    pixel_distance_RIGHT_B = np.argmin(canvas[Y2 - open_tolerance_threshold:Y2 + open_tolerance_threshold, X2 + 1:].mean(axis=0))
                    pixel_value_RIGHT_B = canvas[Y2 - open_tolerance_threshold:Y2 + open_tolerance_threshold, X2 + 1:].mean(axis=0)[np.argmin(canvas[Y2 - open_tolerance_threshold:Y2 + open_tolerance_threshold, X2 + 1:].mean(axis=0))]
                    pixel_distance_RIGHT_B = pixel_distance_RIGHT_B if pixel_value_RIGHT_B != 255 else np.inf
                    pixel_distance_DOWN_B = np.argmin(canvas[Y2 + 1:, X2 - open_tolerance_threshold:X2 + open_tolerance_threshold].mean(axis=1))
                    pixel_value_DOWN_B = canvas[Y2 + 1:, X2 - open_tolerance_threshold:X2 + open_tolerance_threshold].mean(axis=1)[np.argmin(canvas[Y2 + 1:, X2 - open_tolerance_threshold:X2 + open_tolerance_threshold].mean(axis=1))]
                    pixel_distance_DOWN_B = pixel_distance_DOWN_B if pixel_value_DOWN_B != 255 else np.inf
                    if min(pixel_distance_LEFT_B, pixel_distance_RIGHT_B, pixel_distance_DOWN_B) <= 500:
                        if np.argmin([pixel_distance_LEFT_B, pixel_distance_RIGHT_B, pixel_distance_DOWN_B]) == 0 and pixel_value_LEFT_B != 255:
                            valid_wall_lines.append([[X2 - pixel_distance_LEFT_B, Y2, X2, Y2]])
                            is_extended = True
                        if np.argmin([pixel_distance_LEFT_B, pixel_distance_RIGHT_B, pixel_distance_DOWN_B]) == 1 and pixel_value_RIGHT_B != 255:
                            valid_wall_lines.append([[X2, Y2, X2 + pixel_distance_RIGHT_B, Y2]])
                            is_extended = True
                        if np.argmin([pixel_distance_LEFT_B, pixel_distance_RIGHT_B, pixel_distance_DOWN_B]) == 2 and pixel_value_DOWN_B != 255:
                            valid_wall_lines.append([[X2, Y2, X2, Y2 + pixel_distance_DOWN_B]])
                            is_extended = True
                    if is_extended:
                        valid_wall_lines.append(wall_line)
                else:
                    valid_wall_lines.append(wall_line)

                continue
            valid_wall_lines.append(wall_line)

        return valid_wall_lines

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

    def _draw_line(self, wall_line, canvas, overlay_enabled=False):
        x1, y1, x2, y2 = wall_line[0]['x'], wall_line[0]['y'], wall_line[1]['x'], wall_line[1]['y']
        line_width = 1
        if overlay_enabled:
            line_width = 3
        cv2.line(
            canvas,
            (x1, y1),
            (x2, y2),
            (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)),
            line_width
        )

        return canvas

    def _patch_to_line(self, patch_GRAY, floor_plan_path, offset, scale):
        lines = self._preprocessing(
            patch_GRAY,
            floor_plan_path,
            offset,
            scale,
        )
        if lines is None:
            return

        lines = self._deduplicate_lines(lines)

        return lines

    @property
    def scale(self):
        return self._scale

    def _load_ceiling_height_and_scale(
        self,
        offset,
        plan_BGR,
        transcription_block_with_centroids,
        architectural_scale=None,
        standard_ceiling_height=None,
        trust_scale=True,
    ):
        def load_least_scale(scales):
            if not scales:
                return
            scales_on_paper_length = list()
            scales_normalized = list()
            for scale in scales:
                scales_normalized.append(self.scale_canonical(scale))
                scales_on_paper_length.append(float(Fraction(scale.split('=')[0].strip('`'))))
            least_scale_index = scales_on_paper_length.index(min(scales_on_paper_length))
            return scales_normalized[least_scale_index]

        def load_scale_from_OCR(offset_LEFT, offset_RIGHT, offset_TOP, offset_BOTTOM):
            scales_possible = list()
            nearest_transcription_blocks = self._load_nearest_transcription_blocks(((offset_LEFT + offset_RIGHT) / 2, (offset_TOP + offset_BOTTOM) / 2), transcription_block_with_centroids)
            for transcription, _ in nearest_transcription_blocks.items():
                scale_possible = self.load_scale_from_text(transcription)
                if scale_possible:
                    scales_possible.append(scale_possible)
            if not scales_possible:
                for transcription, _ in transcription_block_with_centroids.items():
                    scale_possible = self.load_scale_from_text(transcription)
                    if scale_possible:
                        scales_possible.append(scale_possible)
            scale = load_least_scale(scales_possible)
            return scale

        height_in_pixels, width_in_pixels, _ = plan_BGR.shape
        (offset_top_left_X, offset_top_left_Y), (offset_bottom_right_X, offset_bottom_right_Y) = offset
        margin_X, margin_Y = 10 * round(width_in_pixels / 1920), 10 * round(height_in_pixels / 1080)
        LEFT = round(offset_top_left_X * width_in_pixels)
        TOP = round(offset_top_left_Y * height_in_pixels)
        BOTTOM = round(offset_bottom_right_Y * height_in_pixels)
        RIGHT = round(offset_bottom_right_X * width_in_pixels)
        canvas = cv2.rectangle(
            plan_BGR,
            (max(0, LEFT - margin_X), max(0, TOP - margin_Y)),
            (min(width_in_pixels, RIGHT + margin_X), min(height_in_pixels, BOTTOM + margin_Y)),
            (0, 255, 0),
            10
        )
        _, canvas_buffer_array = cv2.imencode(".png", canvas)
        bytes_canvas = canvas_buffer_array.tobytes()
        query = Content(role="user", parts=[Part.from_data(data=bytes_canvas, mime_type="image/png")])
        ceiling_height_and_scale = dict(ceiling_height=self._height_in_feet, scale=self._scale)

        scale, ceiling_height = None, None
        architectural_scale_fallback = None
        if not trust_scale:
            architectural_scale_fallback = architectural_scale
            architectural_scale = None
        if architectural_scale and standard_ceiling_height:
            scale = self.scale_canonical(architectural_scale)
            ceiling_height = standard_ceiling_height
        elif not architectural_scale and standard_ceiling_height:
            try:
                if self._is_cached["SCALE_DETECTOR"]:
                    response, _ = phoenix_call(
                        lambda feedback_prompt, temperature: self._vertex_ai_client_metadata_scale_extraction.generate_content(
                            contents=[feedback_prompt, query] if feedback_prompt else [query],
                            generation_config={**self._vertex_ai_generation_config, "temperature": temperature},
                        ),
                        self._llm_rate_limiter,
                        max_retry=self._credentials["VertexAI"]["llm"]["max_retry"],
                        pydantic_model=ScaleDetectorResponse,
                    )
                else:
                    response, _ = phoenix_call(
                        lambda feedback_prompt, temperature: self._vertex_ai_client_metadata_scale_extraction(SCALE_DETECTOR.format(supported_scales_architectural=self.scales_architectural)).generate_content(
                            contents=[feedback_prompt, query] if feedback_prompt else [query],
                            generation_config={**self._vertex_ai_generation_config, "temperature": temperature},
                        ),
                        self._llm_rate_limiter,
                        max_retry=self._credentials["VertexAI"]["llm"]["max_retry"],
                        pydantic_model=ScaleDetectorResponse,
                    )
                if response.scale and response.scale.upper() == "NULL":
                    scale = None
                if response.scale and response.scale.upper() != "NULL":
                    scale = self.scale_canonical(response.scale)
                if response.scale_confidence < 0.95:
                    scale = load_scale_from_OCR(LEFT, RIGHT, TOP, BOTTOM)
                ceiling_height = standard_ceiling_height
                logging.info(f"SYSTEM: Section: {self._section_name}, Vertex AI Gemini: Scale: {response.scale}, Confidence: {response.scale_confidence}")
            except Exception as e:
                logging.warning(f"SYSTEM: Section: {self._section_name}, Standard Scale detection failed with error: {e}")
        elif architectural_scale and not standard_ceiling_height:
            try:
                if self._is_cached["CEILING_HEIGHT_DETECTOR"]:
                    response, _ = phoenix_call(
                        lambda feedback_prompt, temperature: self._vertex_ai_client_metadata_ceiling_height_extraction.generate_content(
                            contents=[feedback_prompt, query] if feedback_prompt else [query],
                            generation_config={**self._vertex_ai_generation_config, "temperature": temperature},
                        ),
                        self._llm_rate_limiter,
                        max_retry=self._credentials["VertexAI"]["llm"]["max_retry"],
                        pydantic_model=CeilingHeightDetectorResponse,
                    )
                else:
                    response, _ = phoenix_call(
                        lambda feedback_prompt, temperature: self._vertex_ai_client_metadata_ceiling_height_extraction(CEILING_HEIGHT_DETECTOR).generate_content(
                            contents=[feedback_prompt, query] if feedback_prompt else [query],
                            generation_config={**self._vertex_ai_generation_config, "temperature": temperature},
                        ),
                        self._llm_rate_limiter,
                        max_retry=self._credentials["VertexAI"]["llm"]["max_retry"],
                        pydantic_model=CeilingHeightDetectorResponse,
                    )
                scale = self.scale_canonical(architectural_scale)
                ceiling_height = response.ceiling_height
                logging.info(f"SYSTEM: Section: {self._section_name}, Vertex AI Gemini: Ceiling Height: {response.ceiling_height}")
            except Exception as e:
                logging.warning(f"SYSTEM: Section: {self._section_name}, Standard Ceiling Height detection failed with error: {e}")
        else:
            try:
                if self._is_cached["SCALE_AND_CEILING_HEIGHT_DETECTOR"]:
                    response, _ = phoenix_call(
                        lambda feedback_prompt, temperature: self._vertex_ai_client_metadata_extraction.generate_content(
                            contents=[feedback_prompt, query] if feedback_prompt else [query],
                            generation_config={**self._vertex_ai_generation_config, "temperature": temperature},
                        ),
                        self._llm_rate_limiter,
                        max_retry=self._credentials["VertexAI"]["llm"]["max_retry"],
                        pydantic_model=ScaleAndCeilingHeightDetectorResponse,
                    )
                else:
                    response, _ = phoenix_call(
                        lambda feedback_prompt, temperature: self._vertex_ai_client_metadata_extraction(SCALE_AND_CEILING_HEIGHT_DETECTOR.format(supported_scales_architectural=self.scales_architectural)).generate_content(
                            contents=[feedback_prompt, query] if feedback_prompt else [query],
                            generation_config={**self._vertex_ai_generation_config, "temperature": temperature},
                        ),
                        self._llm_rate_limiter,
                        max_retry=self._credentials["VertexAI"]["llm"]["max_retry"],
                        pydantic_model=ScaleAndCeilingHeightDetectorResponse,
                    )
                ceiling_height = response.ceiling_height
                if response.scale and response.scale.upper() == "NULL":
                    scale = None
                if response.scale and response.scale.upper() != "NULL":
                    scale = self.scale_canonical(response.scale)
                if response.scale_confidence < 0.95:
                    scale = load_scale_from_OCR(LEFT, RIGHT, TOP, BOTTOM)
                logging.info(f"SYSTEM: Section: {self._section_name}, Vertex AI Gemini: Scale: {response.scale}, Confidence: {response.scale_confidence}")
                logging.info(f"SYSTEM: Section: {self._section_name}, Vertex AI Gemini: Ceiling Height: {response.ceiling_height}")
            except Exception as e:
                logging.warning(f"SYSTEM: Section: {self._section_name}, Standard Scale and Ceiling Height detection failed with error: {e}")

        if scale:
            self._scale = scale
            ceiling_height_and_scale["scale"] = scale
            self._is_scale_detected = True
        if not scale and architectural_scale_fallback:
            self._scale = self.scale_canonical(architectural_scale_fallback)
            ceiling_height_and_scale["scale"] = self._scale
            self._is_scale_detected = True
        if ceiling_height:
            ceiling_height_and_scale["ceiling_height"] = ceiling_height

        new_pixel_aspect_ratio_to_feet = self.compute_pixel_aspect_ratio(ceiling_height_and_scale["scale"], self._hyperparameters["pixel_aspect_ratio_to_feet"])
        self._hyperparameters["pixel_aspect_ratio_to_feet"] = new_pixel_aspect_ratio_to_feet
        self._hyperparameters["modelling"]["pixel_aspect_ratio"] = new_pixel_aspect_ratio_to_feet
        self._hyperparameters["modelling"]["height_in_feet"] = round(ceiling_height_and_scale["ceiling_height"], 3)
        return ceiling_height_and_scale

    def _is_wall_valid(
        self,
        wall_line,
        drywall_polygons,
        floor_plan_path,
        offset,
    ):
        canvas = cv2.imread(floor_plan_path)
        height_in_pixels, width_in_pixels, _ = canvas.shape
        (offset_top_left_X, offset_top_left_Y), (offset_bottom_right_X, offset_bottom_right_Y) = offset
        margin_X, margin_Y = 10 * round(width_in_pixels / 1920), 10 * round(height_in_pixels / 1080)
        LEFT = round(offset_top_left_X * width_in_pixels)
        TOP = round(offset_top_left_Y * height_in_pixels)
        BOTTOM = round(offset_bottom_right_Y * height_in_pixels)
        RIGHT = round(offset_bottom_right_X * width_in_pixels)
        canvas = cv2.rectangle(
            canvas,
            (max(0, LEFT - margin_X), max(0, TOP - margin_Y)),
            (min(width_in_pixels, RIGHT + margin_X), min(height_in_pixels, BOTTOM + margin_Y)),
            (0, 255, 0),
            10
        )
        canvas[:max(0, TOP - margin_Y), :] = 255
        canvas[:, :max(0, LEFT - margin_X)] = 255
        canvas[:, min(width_in_pixels, RIGHT + margin_X):] = 255
        canvas[min(height_in_pixels, BOTTOM + margin_Y):, :] = 255
        for drywall_polygon in drywall_polygons:
            canvas_to_overlay = canvas.copy()
            cv2.fillPoly(canvas_to_overlay, pts=[drywall_polygon], color=(0, 0, 255))
            canvas = cv2.addWeighted(canvas_to_overlay, 0.7, canvas, 0.3, 0)
        X1, Y1, X2, Y2 = wall_line[0]
        canvas_to_overlay = canvas.copy()
        cv2.line(canvas_to_overlay, (X1, Y1), (X2, Y2), (0, 0, 255), 2)
        canvas = cv2.addWeighted(canvas_to_overlay, 0.7, canvas, 0.3, 0)

        #from google.cloud.storage import Client as CloudStorageClient
        #client = CloudStorageClient()
        #plan_path = f"/tmp/wall_valid_{np.random.randint(999)}.png"
        #cv2.imwrite(plan_path, canvas)
        #bucket = client.bucket(self._credentials["CloudStorage"]["bucket_name"])
        #blob_path = f"tmp/wall/{Path(plan_path).name}"
        #blob = bucket.blob(blob_path)
        #blob.upload_from_filename(plan_path)

        _, canvas_buffer_array = cv2.imencode(".png", canvas)
        bytes_canvas = canvas_buffer_array.tobytes()
        wall_line_structured = dict(wall=dict(X1=int(X1), Y1=int(Y1), X2=int(X2), Y2=int(Y2)))
        query = Content(role="user", parts=[
            Part.from_text(json.dumps(wall_line_structured)),
            Part.from_data(data=bytes_canvas, mime_type="image/png")
        ])
        try:
            if self._is_cached["WALL_RECTIFIER"]:
                _, is_valid = phoenix_call(
                    lambda feedback_prompt, temperature: self._vertex_ai_client_wall_rectification.generate_content(
                        contents=WALL_RECTIFIER_FEW_SHOT+[feedback_prompt, query] if feedback_prompt else WALL_RECTIFIER_FEW_SHOT+[query],
                        generation_config={**self._vertex_ai_generation_config, "temperature": temperature},
                    ),
                    self._llm_rate_limiter,
                    max_retry=self._credentials["VertexAI"]["llm"]["max_retry"],
                    pydantic_model=WallRectifierResponse,
                )
            else:
                _, is_valid = phoenix_call(
                    lambda feedback_prompt, temperature: self._vertex_ai_client_wall_rectification(WALL_RECTIFIER).generate_content(
                        contents=WALL_RECTIFIER_FEW_SHOT+[feedback_prompt, query] if feedback_prompt else WALL_RECTIFIER_FEW_SHOT+[query],
                        generation_config={**self._vertex_ai_generation_config, "temperature": temperature},
                    ),
                    self._llm_rate_limiter,
                    max_retry=self._credentials["VertexAI"]["llm"]["max_retry"],
                    pydantic_model=WallRectifierResponse,
                )
            if is_valid["confidence"] > 0.9:
                return is_valid["is_valid"]
            return True
        except Exception as e:
            logging.warning(f"SYSTEM: Section: {self._section_name}, Wall validator failed with error: {e}")
            return True

    def _model_polygon(
        self,
        vertices,
        walls,
        area_target,
        polygons_pts,
        floor_plan_path,
        elevation_paths,
        transcription_block_with_centroids,
        walls_unnormalized,
        offset,
        height_default=9.125,
    ):
        def verify_tolerance_length(dimension_wall, wall_unnormalized, confidence_score, wall_normalized):
            if dimension_wall["length"] and confidence_score >= 0.9:
                dimension_wall["length"] = round(dimension_wall["length"], 3)
                X1, Y1, X2, Y2 = wall_normalized[0]
                orientation = self.classify_line(X1, Y1, X2, Y2)
                if orientation == "horizontal":
                    self._imperial_scales_sampled['X'].append(dimension_wall["length"] / (X2 - X1))
                if orientation == "vertical":
                    self._imperial_scales_sampled['Y'].append(dimension_wall["length"] / (Y2 - Y1))
            else:
                X1, Y1, X2, Y2 = wall_unnormalized[0]
                length_target = round(math.hypot(
                    (X1 - X2) * self._hyperparameters["modelling"]["pixel_aspect_ratio"]["horizontal"],
                    (Y1 - Y2) * self._hyperparameters["modelling"]["pixel_aspect_ratio"]["vertical"]
                ), 3)
                dimension_wall["length"] = length_target
            if not dimension_wall["width"]:
                dimension_wall["width"] = self._width_in_feet
            else:
                dimension_wall["width"] = round(dimension_wall["width"], 3)

            return dimension_wall

        def verify_tolerance_area(area_polygon_predicted, area_polygon_target, confidence_score, vertices_normalized):
            if area_polygon_predicted and confidence_score >= 0.9:
                area_pixels = cv2.contourArea(np.array(vertices_normalized, dtype=np.float32))
                self._imperial_scales_sampled['A'].append(area_polygon_predicted / area_pixels)
                return round(area_polygon_predicted, 3)
            return round(area_polygon_target, 3)

        def verify_tolerance_height(height_predicted, confidence_score):
            if height_predicted and height_predicted != -1 and confidence_score >= 0.9:
                return round(height_predicted, 3)

            return round(height_default, 3)

        canvas = cv2.imread(floor_plan_path)
        height_in_pixels, width_in_pixels, _ = canvas.shape
        (offset_top_left_X, offset_top_left_Y), (offset_bottom_right_X, offset_bottom_right_Y) = offset
        margin_X, margin_Y = 10 * round(width_in_pixels / 1920), 10 * round(height_in_pixels / 1080)
        LEFT = round(offset_top_left_X * width_in_pixels)
        TOP = round(offset_top_left_Y * height_in_pixels)
        BOTTOM = round(offset_bottom_right_Y * height_in_pixels)
        RIGHT = round(offset_bottom_right_X * width_in_pixels)
        canvas = cv2.rectangle(
            canvas,
            (max(0, LEFT - margin_X), max(0, TOP - margin_Y)),
            (min(width_in_pixels, RIGHT + margin_X), min(height_in_pixels, BOTTOM + margin_Y)),
            (0, 255, 0),
            10
        )
        canvas[:max(0, TOP - margin_Y), :] = 255
        canvas[:, :max(0, LEFT - margin_X)] = 255
        canvas[:, min(width_in_pixels, RIGHT + margin_X):] = 255
        canvas[min(height_in_pixels, BOTTOM + margin_Y):, :] = 255
        vertices = np.array(vertices)
        canvas_to_overlay = canvas.copy()
        cv2.fillPoly(canvas_to_overlay, pts=[vertices], color=(0, 0, 255))
        canvas = cv2.addWeighted(canvas_to_overlay, 0.3, canvas, 0.7, 0)
        for wall in walls:
            X1, Y1, X2, Y2 = wall[0]
            bounding_box_top_left = (X1 - 10, Y1 - 10)
            bounding_box_bottom_right = (X2 + 10, Y2 + 10)
            canvas = cv2.rectangle(canvas, bounding_box_top_left, bounding_box_bottom_right, (255, 0, 0), 3)
        for polygon_pts in polygons_pts:
            canvas_to_overlay = canvas.copy()
            cv2.fillPoly(canvas_to_overlay, pts=[polygon_pts], color=(0, 255, 0))
            canvas = cv2.addWeighted(canvas_to_overlay, 0.5, canvas, 0.5, 0)
        transcription_entries = list()
        for transcription, centroid in transcription_block_with_centroids.items():
            transcription_entries.append(dict(text=transcription, centroid=dict(X=centroid[0], Y=centroid[1])))
        _, canvas_buffer_array = cv2.imencode(".png", canvas)
        bytes_canvas = canvas_buffer_array.tobytes()
        perimeter_lines = list()
        for wall in walls:
            X1, Y1, X2, Y2 = wall[0]
            perimeter_line = dict(wall=dict(X1=int(X1), Y1=int(Y1), X2=int(X2), Y2=int(Y2)))
            if perimeter_line not in perimeter_lines:
                perimeter_lines.append(perimeter_line)
        polygon = dict(vertices=vertices.tolist(), perimeter_wall_lines=list(perimeter_lines), transcription_entries=transcription_entries)
        parts_elevations = list()
        for elevation_index, elevation_path in enumerate(elevation_paths):
            elevation_canvas = cv2.imread(elevation_path)
            _, elevation_canvas_buffer_array = cv2.imencode(".png", elevation_canvas)
            bytes_elevation_canvas = elevation_canvas_buffer_array.tobytes()
            parts_elevations.append(Part.from_text(f"ELEVATION PLAN: {elevation_index + 1}")),
            parts_elevations.append(Part.from_data(data=bytes_elevation_canvas, mime_type="image/png"))
        query = Content(role="user", parts=[
            Part.from_text(json.dumps(polygon)),
            Part.from_data(data=bytes_canvas, mime_type="image/png")
        ]+parts_elevations)

        try:
            if self._is_cached["POLYGON_DETECTOR_AND_DRYWALL_PREDICTOR"]:
                _, model_polygon = phoenix_call(
                    lambda feedback_prompt, temperature: self._vertex_ai_client_polygon_detection_and_drywall_prediction.generate_content(
                        contents=POLYGON_DETECTOR_AND_DRYWALL_PREDICTOR_CALIFORNIA_FEW_SHOT+[feedback_prompt, query] if feedback_prompt else POLYGON_DETECTOR_AND_DRYWALL_PREDICTOR_CALIFORNIA_FEW_SHOT+[query],
                        generation_config={**self._vertex_ai_generation_config, "temperature": temperature},
                    ),
                    self._llm_rate_limiter,
                    max_retry=self._credentials["VertexAI"]["llm"]["max_retry"],
                    pydantic_model=PolygonDetectorAndDrywallPredictorResponse,
                    verify_field_counts=dict(wall_parameters=len(perimeter_lines)),
                )
            else:
                _, model_polygon = phoenix_call(
                    lambda feedback_prompt, temperature: self._vertex_ai_client_polygon_detection_and_drywall_prediction(POLYGON_DETECTOR_AND_DRYWALL_PREDICTOR.format(drywall_templates=self._drywall_templates, project_location=self._project_location)).generate_content(
                        contents=POLYGON_DETECTOR_AND_DRYWALL_PREDICTOR_CALIFORNIA_FEW_SHOT+[feedback_prompt, query] if feedback_prompt else POLYGON_DETECTOR_AND_DRYWALL_PREDICTOR_CALIFORNIA_FEW_SHOT+[query],
                        generation_config={**self._vertex_ai_generation_config, "temperature": temperature},
                    ),
                    self._llm_rate_limiter,
                    max_retry=self._credentials["VertexAI"]["llm"]["max_retry"],
                    pydantic_model=PolygonDetectorAndDrywallPredictorResponse,
                    verify_field_counts=dict(wall_parameters=len(perimeter_lines)),
                )
            model_polygon["ceiling"]["area"] = round(area_target, 3)
            model_polygon["ceiling"]["height"] = verify_tolerance_height(model_polygon["ceiling"]["height"], model_polygon["ceiling"]["confidence_height"])
            for index, (dimension_wall_predicted, wall_unnormalized, wall_normalized) in enumerate(zip(model_polygon["wall_parameters"], walls_unnormalized, walls)):
                dimension_wall_rectified = dimension_wall_predicted
                dimension_wall_rectified = verify_tolerance_length(dimension_wall_predicted, wall_unnormalized, dimension_wall_predicted["confidence_length"], wall_normalized)
                dimension_wall_rectified["drywall_assembly"]["height"] = verify_tolerance_height(dimension_wall_predicted["drywall_assembly"]["height"], dimension_wall_predicted["drywall_assembly"]["confidence_height"])

                model_polygon["wall_parameters"][index] = dimension_wall_rectified
            logging.info(f"SYSTEM: Section: {self._section_name}, POLYGON DETECTED: {json.dumps(model_polygon)}")
        except Exception as e:
            logging.warning(f"SYSTEM: Section: {self._section_name}, Drywall prediction for polygon: {json.dumps(polygon)} failed with error: {e}")
            model_polygon = {
                "ceiling": {
                    "room_name": '',
                    "area": area_target,
                    "ceiling_type": "Flat",
                    "height": height_default,
                    "pitch": {
                        "rise": 0.0,
                        "run": 0.0
                    },
                    "slope_enabled": False,
                    "tilt_axis": '',
                    "drywall_assembly": {
                        "material": "D12C - 1/2\" DW INTERIOR CEILING",
                        "color_code": [10, 78, 69],
                        "materials_vertically_stacked": [],
                        "color_codes_stacked": [],
                        "thickness": 0.04,
                        "layers": 1,
                        "fire_rating": 0,
                        "waste_factor": "8-12%",
                    },
                    "code_references": list(),
                    "recommendation": "FP - Verify the drywall assignment"
                }
            }
            wall_parameters = list()
            for wall in walls:
                X1, Y1, X2, Y2 = wall[0]
                length = round(math.hypot(
                    (X1 - X2) * self._hyperparameters["modelling"]["pixel_aspect_ratio"]["horizontal"],
                    (Y1 - Y2) * self._hyperparameters["modelling"]["pixel_aspect_ratio"]["vertical"]
                ), 2)
                wall_parameters.append(
                    {
                        "room_name": '',
                        "length": length,
                        "width": self._hyperparameters["modelling"]["width_in_feet"],
                        "wall_type": '',
                        "openings": [dict(opening_type="NULL", count=0, length=0, height=0)],
                        "drywall_assembly": {
                            "material": "D12L - 1/2\" DW LITE-WEIGHT",
                            "height": height_default,
                            "color_code": [71, 239, 143],
                            "materials_vertically_stacked": [],
                            "color_codes_stacked": [],
                            "heights_stacked": [],
                            "thickness": 0.04,
                            "layers": 1,
                            "fire_rating": 0,
                            "waste_factor": "8-12%"
                        },
                        "code_references": list(),
                        "recommendation": "FP - Verify the drywall assignment"
                    }
                )
            model_polygon["wall_parameters"] = wall_parameters

        return model_polygon

    def _predict_polygon(
        self,
        vertices,
        walls,
        area_target,
        polygons_pts,
        floor_plan_path,
        elevation_paths,
        output_schema_custom,
        polygon_detector_and_drywall_predictor_custom_response,
        transcription_block_with_centroids,
        walls_unnormalized,
        offset,
        height_default=9.125,
    ):
        def verify_tolerance_length(dimension_wall, wall_unnormalized, confidence_score, wall_normalized):
            if dimension_wall["length"] and confidence_score >= 0.9:
                dimension_wall["length"] = round(dimension_wall["length"], 3)
                X1, Y1, X2, Y2 = wall_normalized[0]
                orientation = self.classify_line(X1, Y1, X2, Y2)
                if orientation == "horizontal":
                    self._imperial_scales_sampled['X'].append(dimension_wall["length"] / (X2 - X1))
                if orientation == "vertical":
                    self._imperial_scales_sampled['Y'].append(dimension_wall["length"] / (Y2 - Y1))
            else:
                X1, Y1, X2, Y2 = wall_unnormalized[0]
                length_target = round(math.hypot(
                    (X1 - X2) * self._hyperparameters["modelling"]["pixel_aspect_ratio"]["horizontal"],
                    (Y1 - Y2) * self._hyperparameters["modelling"]["pixel_aspect_ratio"]["vertical"]
                ), 3)
                dimension_wall["length"] = length_target
            if not dimension_wall["width"]:
                dimension_wall["width"] = self._width_in_feet
            else:
                dimension_wall["width"] = round(dimension_wall["width"], 3)

            return dimension_wall

        def verify_tolerance_height(height_predicted, confidence_score):
            if height_predicted and height_predicted != -1 and confidence_score >= 0.9:
                return round(height_predicted, 3)

            return round(height_default, 3)

        canvas = cv2.imread(floor_plan_path)
        height_in_pixels, width_in_pixels, _ = canvas.shape
        (offset_top_left_X, offset_top_left_Y), (offset_bottom_right_X, offset_bottom_right_Y) = offset
        margin_X, margin_Y = 10 * round(width_in_pixels / 1920), 10 * round(height_in_pixels / 1080)
        LEFT = round(offset_top_left_X * width_in_pixels)
        TOP = round(offset_top_left_Y * height_in_pixels)
        BOTTOM = round(offset_bottom_right_Y * height_in_pixels)
        RIGHT = round(offset_bottom_right_X * width_in_pixels)
        canvas = cv2.rectangle(
            canvas,
            (max(0, LEFT - margin_X), max(0, TOP - margin_Y)),
            (min(width_in_pixels, RIGHT + margin_X), min(height_in_pixels, BOTTOM + margin_Y)),
            (0, 255, 0),
            10
        )
        canvas[:max(0, TOP - margin_Y), :] = 255
        canvas[:, :max(0, LEFT - margin_X)] = 255
        canvas[:, min(width_in_pixels, RIGHT + margin_X):] = 255
        canvas[min(height_in_pixels, BOTTOM + margin_Y):, :] = 255
        vertices = np.array(vertices)
        canvas_to_overlay = canvas.copy()
        cv2.fillPoly(canvas_to_overlay, pts=[vertices], color=(0, 0, 255))
        canvas = cv2.addWeighted(canvas_to_overlay, 0.3, canvas, 0.7, 0)
        for wall in walls:
            X1, Y1, X2, Y2 = wall[0]
            bounding_box_top_left = (X1 - 10, Y1 - 10)
            bounding_box_bottom_right = (X2 + 10, Y2 + 10)
            canvas = cv2.rectangle(canvas, bounding_box_top_left, bounding_box_bottom_right, (255, 0, 0), 3)
        for polygon_pts in polygons_pts:
            canvas_to_overlay = canvas.copy()
            cv2.fillPoly(canvas_to_overlay, pts=[polygon_pts], color=(0, 255, 0))
            canvas = cv2.addWeighted(canvas_to_overlay, 0.5, canvas, 0.5, 0)
        transcription_entries = list()
        for transcription, centroid in transcription_block_with_centroids.items():
            transcription_entries.append(dict(text=transcription, centroid=dict(X=centroid[0], Y=centroid[1])))
        _, canvas_buffer_array = cv2.imencode(".png", canvas)
        bytes_canvas = canvas_buffer_array.tobytes()
        perimeter_lines = list()
        for wall in walls:
            X1, Y1, X2, Y2 = wall[0]
            perimeter_line = dict(wall=dict(X1=int(X1), Y1=int(Y1), X2=int(X2), Y2=int(Y2)))
            if perimeter_line not in perimeter_lines:
                perimeter_lines.append(perimeter_line)
        polygon = dict(vertices=vertices.tolist(), perimeter_wall_lines=list(perimeter_lines), transcription_entries=transcription_entries)
        parts_elevations = list()
        for elevation_index, elevation_path in enumerate(elevation_paths):
            elevation_canvas = cv2.imread(elevation_path)
            _, elevation_canvas_buffer_array = cv2.imencode(".png", elevation_canvas)
            bytes_elevation_canvas = elevation_canvas_buffer_array.tobytes()
            parts_elevations.append(Part.from_text(f"ELEVATION PLAN: {elevation_index + 1}")),
            parts_elevations.append(Part.from_data(data=bytes_elevation_canvas, mime_type="image/png"))
        query = Content(role="user", parts=[
            Part.from_text(json.dumps(polygon)),
            Part.from_data(data=bytes_canvas, mime_type="image/png"),
        ]+parts_elevations+[Part.from_text(json.dumps(dict(output_schema=output_schema_custom)))])

        try:
            if self._is_cached["POLYGON_DETECTOR_AND_DRYWALL_PREDICTOR_CUSTOM"]:
                _, predict_polygon = phoenix_call(
                    lambda feedback_prompt, temperature: self._vertex_ai_client_polygon_detection_and_drywall_prediction_custom.generate_content(
                        contents=POLYGON_DETECTOR_AND_DRYWALL_PREDICTOR_CALIFORNIA_FEW_SHOT+[feedback_prompt, query] if feedback_prompt else POLYGON_DETECTOR_AND_DRYWALL_PREDICTOR_CALIFORNIA_FEW_SHOT+[query],
                        generation_config={**self._vertex_ai_generation_config, "temperature": temperature},
                    ),
                    self._llm_rate_limiter,
                    max_retry=self._credentials["VertexAI"]["llm"]["max_retry"],
                    pydantic_model=polygon_detector_and_drywall_predictor_custom_response,
                    verify_field_counts=dict(wall_parameters=len(perimeter_lines)),
                )
            else:
                _, predict_polygon = phoenix_call(
                    lambda feedback_prompt, temperature: self._vertex_ai_client_polygon_detection_and_drywall_prediction_custom(POLYGON_DETECTOR_AND_DRYWALL_PREDICTOR_CUSTOM.format(drywall_templates=self._drywall_templates, project_location=self._project_location)).generate_content(
                        contents=POLYGON_DETECTOR_AND_DRYWALL_PREDICTOR_CALIFORNIA_FEW_SHOT+[feedback_prompt, query] if feedback_prompt else POLYGON_DETECTOR_AND_DRYWALL_PREDICTOR_CALIFORNIA_FEW_SHOT+[query],
                        generation_config={**self._vertex_ai_generation_config, "temperature": temperature},
                    ),
                    self._llm_rate_limiter,
                    max_retry=self._credentials["VertexAI"]["llm"]["max_retry"],
                    pydantic_model=polygon_detector_and_drywall_predictor_custom_response,
                    verify_field_counts=dict(wall_parameters=len(perimeter_lines)),
                )
            if predict_polygon.get("ceiling") and predict_polygon["ceiling"].get("area"): 
                predict_polygon["ceiling"]["area"] = round(area_target, 3)
            if predict_polygon.get("ceiling") and predict_polygon["ceiling"].get("height"): 
                predict_polygon["ceiling"]["height"] = verify_tolerance_height(predict_polygon["ceiling"]["height"], predict_polygon["ceiling"]["confidence_height"])
            if predict_polygon.get("wall_parameters"): 
                for index, (dimension_wall_predicted, wall_unnormalized, wall_normalized) in enumerate(zip(predict_polygon["wall_parameters"], walls_unnormalized, walls)):
                    dimension_wall_rectified = dimension_wall_predicted
                    if dimension_wall_predicted.get("length"):
                        dimension_wall_rectified = verify_tolerance_length(dimension_wall_predicted, wall_unnormalized, dimension_wall_predicted["confidence_length"], wall_normalized)
                    if dimension_wall_predicted.get("drywall_assembly") and dimension_wall_predicted.get("drywall_assembly").get("height"):
                        dimension_wall_rectified["drywall_assembly"]["height"] = verify_tolerance_height(dimension_wall_predicted["drywall_assembly"]["height"], dimension_wall_predicted["drywall_assembly"]["confidence_height"])

                    predict_polygon["wall_parameters"][index] = dimension_wall_rectified
            logging.info(f"SYSTEM: Section: {self._section_name}, POLYGON DETECTED: {json.dumps(predict_polygon)}")
        except Exception as e:
            logging.warning(f"SYSTEM: Section: {self._section_name}, Drywall prediction for polygon: {json.dumps(polygon)} failed with error: {e}")
            predict_polygon = {
                "ceiling": {
                    "room_name": '',
                    "area": area_target,
                    "ceiling_type": "Flat",
                    "height": height_default,
                    "pitch": {
                        "rise": 0.0,
                        "run": 0.0
                    },
                    "slope_enabled": False,
                    "tilt_axis": '',
                    "drywall_assembly": {
                        "material": "D12C - 1/2\" DW INTERIOR CEILING",
                        "color_code": [10, 78, 69],
                        "materials_vertically_stacked": [],
                        "color_codes_stacked": [],
                        "thickness": 0.04,
                        "layers": 1,
                        "fire_rating": 0,
                        "waste_factor": "8-12%",
                    },
                    "code_references": list(),
                    "recommendation": "FP - Verify the drywall assignment"
                }
            }
            wall_parameters = list()
            for wall in walls:
                X1, Y1, X2, Y2 = wall[0]
                length = round(math.hypot(
                    (X1 - X2) * self._hyperparameters["modelling"]["pixel_aspect_ratio"]["horizontal"],
                    (Y1 - Y2) * self._hyperparameters["modelling"]["pixel_aspect_ratio"]["vertical"]
                ), 2)
                wall_parameters.append(
                    {
                        "room_name": '',
                        "length": length,
                        "width": self._hyperparameters["modelling"]["width_in_feet"],
                        "wall_type": '',
                        "openings": [dict(opening_type="NULL", count=0, length=0, height=0)],
                        "drywall_assembly": {
                            "material": "D12L - 1/2\" DW LITE-WEIGHT",
                            "height": height_default,
                            "color_code": [71, 239, 143],
                            "materials_vertically_stacked": [],
                            "color_codes_stacked": [],
                            "heights_stacked": [],
                            "thickness": 0.04,
                            "layers": 1,
                            "fire_rating": 0,
                            "waste_factor": "8-12%"
                        },
                        "code_references": list(),
                        "recommendation": "FP - Verify the drywall assignment"
                    }
                )
            predict_polygon["wall_parameters"] = wall_parameters

        return predict_polygon

    @classmethod
    def model_to_sketch(
        cls,
        walls_2d,
        polygons,
    ):
        walls_2d_layout = deepcopy(walls_2d)
        if walls_2d_layout:
            for wall_2d_layout in walls_2d_layout:
                wall_2d_layout["thickness"] = -1
                wall_2d_layout["length"] = -1
                wall_2d_layout["type"] = ''
                wall_2d_layout["openings"] = list()
                for wall_2d_polygon_drywall in wall_2d_layout["polygons_drywall"]:
                    wall_2d_polygon_drywall["room_name"] = ''
                    wall_2d_polygon_drywall["type"] = "--"
                    wall_2d_polygon_drywall["height"] = -1
                    wall_2d_polygon_drywall["color"] = [0, 0, 0]
                    wall_2d_polygon_drywall["type_stacked"] = list()
                    wall_2d_polygon_drywall["color_stacked"] = list()
                    wall_2d_polygon_drywall["height_stacked"] = list()
                    wall_2d_polygon_drywall["thickness"] = -1
                    wall_2d_polygon_drywall["layers"] = -1
                    wall_2d_polygon_drywall["fire_rating"] = -1
                    wall_2d_polygon_drywall["recommendation"] = ''
                    wall_2d_polygon_drywall["waste_factor"] = ''

        polygons_layout = deepcopy(polygons)
        if polygons_layout:
            for polygon_layout in polygons_layout:
                polygon_layout["type"] = "--"
                polygon_layout["height"] = -1
                polygon_layout["pitch"] = dict(rise=-1, run=-1)
                polygon_layout["tilt_axis"] = ''
                polygon_layout["room_name"] = ''
                polygon_layout["polygon_drywall"]["type"] = "--"
                polygon_layout["polygon_drywall"]["color"] = [25, 25, 25]
                polygon_layout["polygon_drywall"]["type_stacked"] = list()
                polygon_layout["polygon_drywall"]["color_stacked"] = list()
                polygon_layout["polygon_drywall"]["thickness"] = -1
                polygon_layout["polygon_drywall"]["layers"] = -1
                polygon_layout["polygon_drywall"]["fire_rating"] = -1
                polygon_layout["polygon_drywall"]["recommendation"] = ''
                polygon_layout["polygon_drywall"]["waste_factor"] = -1

        return walls_2d_layout, polygons_layout

    def _add_walls_polygon(
        self,
        vertices,
        area,
        perimeter_walls,
        polygons,
        scale,
        height_default,
        floor_plan_path,
        elevation_paths,
        transcription_block_with_centroids,
        index,
        offset,
    ):
        def load_wall_payload(wall_line):
            X1, Y1, X2, Y2 = wall_line[0]
            wall_line_structured = [
                dict(x=int(X1), y=int(Y1)),
                dict(x=int(X2), y=int(Y2))
            ]
            for wall_2d in self._walls_2d:
                if wall_2d["wall_line"] == wall_line_structured:
                    return wall_2d

        scale_x, scale_y = scale
        perimeter_walls_unnormalized = list()
        for perimeter_wall in perimeter_walls:
            X1, Y1, X2, Y2 = perimeter_wall[0]
            perimeter_wall_unnormalized = [[round(X1 / scale_x), round(Y1 / scale_y), round(X2 / scale_x), round(Y2 / scale_y)]]
            perimeter_walls_unnormalized.append(perimeter_wall_unnormalized)
        polygons_pts_normalized = list()
        for polygon in polygons:
            if isinstance(polygon, list):
                polygon = polygon[0]
            pts_normalized = np.array([
                [polygon["coordinates"][0]['x'], polygon["coordinates"][0]['y']],
                [polygon["coordinates"][1]['x'], polygon["coordinates"][1]['y']],
                [polygon["coordinates"][2]['x'], polygon["coordinates"][2]['y']],
                [polygon["coordinates"][3]['x'], polygon["coordinates"][3]['y']]
            ], np.int32)
            polygons_pts_normalized.append(pts_normalized)
        model_polygon = self._model_polygon(
            vertices,
            perimeter_walls,
            area,
            polygons_pts_normalized,
            floor_plan_path,
            elevation_paths,
            transcription_block_with_centroids,
            perimeter_walls_unnormalized,
            offset,
            height_default=height_default,
        )

        polygon_ids_drywall_interior = list()
        for wall_line, wall_parameter, polygon in zip(perimeter_walls, model_polygon["wall_parameters"], polygons):
            wall_payload = load_wall_payload(wall_line)
            if wall_payload:
                try:
                    thickness = round(float(wall_parameter["drywall_assembly"]["thickness"]), 2)
                except ValueError:
                    thickness = wall_parameter["drywall_assembly"]["thickness"]
                except (KeyError, TypeError):
                    wall_parameter["drywall_assembly"] = dict(
                        material="DISABLED",
                        height=height_default,
                        color_code=[0, 0, 255],
                        materials_vertically_stacked=[],
                        color_codes_stacked=[],
                        heights_stacked=[],
                        thickness=-1,
                        layers=0,
                        fire_rating=0,
                        waste_factor="NA"
                    )
                    wall_parameter["recommendation"] = "NA"
                    wall_parameter["room_name"] = ''
                    thickness=-1
                if len(wall_payload["polygons_drywall"]) == 2:
                    continue
                wall_payload["polygons_drywall"].append(
                    dict(
                        id=f"{wall_payload["id"]}.b",
                        room_name=wall_parameter["room_name"],
                        polygon=polygon["coordinates"] if isinstance(polygon, dict) else polygon[0]["coordinates"],
                        type=wall_parameter["drywall_assembly"]["material"],
                        height=wall_parameter["drywall_assembly"]["height"],
                        color=list(wall_parameter["drywall_assembly"]["color_code"]),
                        type_stacked=wall_parameter["drywall_assembly"]["materials_vertically_stacked"],
                        color_stacked=list(wall_parameter["drywall_assembly"]["color_codes_stacked"]),
                        height_stacked=wall_parameter["drywall_assembly"].get("heights_stacked", list()),
                        thickness=thickness,
                        layers=wall_parameter["drywall_assembly"]["layers"],
                        fire_rating=wall_parameter["drywall_assembly"]["fire_rating"],
                        recommendation=wall_parameter["recommendation"],
                        waste_factor=wall_parameter["drywall_assembly"]["waste_factor"],
                        enabled=True
                    )
                )
                polygon_ids_drywall_interior.append(f"{wall_payload["id"]}.b")
            else:
                X1, Y1, X2, Y2 = wall_line[0]
                wall = dict(
                    id=len(self._walls_2d),
                    wall_line=[
                        dict(x=int(X1), y=int(Y1)),
                        dict(x=int(X2), y=int(Y2))
                    ],
                    thickness=wall_parameter["width"],
                    length=wall_parameter.get("length", -1),
                    type=wall_parameter["wall_type"],
                    openings=wall_parameter["openings"],
                    polygons_drywall=list()
                )
                try:
                    thickness = round(float(wall_parameter["drywall_assembly"]["thickness"]), 2)
                except ValueError:
                    thickness = wall_parameter["drywall_assembly"]["thickness"]
                except (KeyError, TypeError):
                    wall_parameter["drywall_assembly"] = dict(
                        material="DISABLED",
                        height=height_default,
                        color_code=[0, 0, 255],
                        materials_vertically_stacked=[],
                        color_codes_stacked=[],
                        heights_stacked=[],
                        thickness=-1,
                        layers=0,
                        fire_rating=0,
                        waste_factor="NA"
                    )
                    wall_parameter["recommendation"] = "NA"
                    wall_parameter["room_name"] = ''
                    thickness=-1
                wall["polygons_drywall"].append(
                    dict(
                        id=f"{len(self._walls_2d)}.a",
                        room_name=wall_parameter["room_name"],
                        polygon=polygon["coordinates"] if isinstance(polygon, dict) else polygon[0]["coordinates"],
                        type=wall_parameter["drywall_assembly"]["material"],
                        height=wall_parameter["drywall_assembly"]["height"],
                        color=list(wall_parameter["drywall_assembly"]["color_code"]),
                        type_stacked=wall_parameter["drywall_assembly"]["materials_vertically_stacked"],
                        color_stacked=list(wall_parameter["drywall_assembly"]["color_codes_stacked"]),
                        height_stacked=wall_parameter["drywall_assembly"].get("heights_stacked", list()),
                        thickness=thickness,
                        layers=wall_parameter["drywall_assembly"]["layers"],
                        fire_rating=wall_parameter["drywall_assembly"]["fire_rating"],
                        recommendation=wall_parameter["recommendation"],
                        waste_factor=wall_parameter["drywall_assembly"]["waste_factor"],
                        enabled=True,
                    )
                )
                polygon_ids_drywall_interior.append(f"{len(self._walls_2d)}.a")
                if isinstance(polygon, list):
                    wall["polygons_drywall"].append(
                        dict(
                            id=f"{len(self._walls_2d)}.b",
                            room_name=wall_parameter["room_name"],
                            polygon=polygon[1]["coordinates"],
                            type=wall_parameter["drywall_assembly"]["material"],
                            height=wall_parameter["drywall_assembly"]["height"],
                            color=list(wall_parameter["drywall_assembly"]["color_code"]),
                            type_stacked=wall_parameter["drywall_assembly"]["materials_vertically_stacked"],
                            color_stacked=list(wall_parameter["drywall_assembly"]["color_codes_stacked"]),
                            height_stacked=wall_parameter["drywall_assembly"].get("heights_stacked", list()),
                            thickness=thickness,
                            layers=wall_parameter["drywall_assembly"]["layers"],
                            fire_rating=wall_parameter["drywall_assembly"]["fire_rating"],
                            recommendation=wall_parameter["recommendation"],
                            waste_factor=wall_parameter["drywall_assembly"]["waste_factor"],
                            enabled=True,
                        )
                    )
                    polygon_ids_drywall_interior.append(f"{len(self._walls_2d)}.b")
                self._walls_2d.append(wall)

        polygon_ids_drywall_interior_filtered = list()
        interior_wall_ids = set()
        for polygon_id_drywall_interior in polygon_ids_drywall_interior:
            wall_id = polygon_id_drywall_interior.split('.')[0]
            if wall_id in interior_wall_ids:
                continue
            polygon_ids_drywall_interior_filtered.append(polygon_id_drywall_interior)
            interior_wall_ids.add(wall_id)

        polygon = dict(
            id=index,
            area=model_polygon["ceiling"]["area"],
            vertices=vertices,
            type=model_polygon["ceiling"]["ceiling_type"],
            height=model_polygon["ceiling"]["height"] if model_polygon["ceiling"]["height"] else height_default,
            pitch=model_polygon["ceiling"]["pitch"],
            slope_enabled=model_polygon["ceiling"]["slope_enabled"],
            tilt_axis=model_polygon["ceiling"]["tilt_axis"],
            room_name=model_polygon["ceiling"]["room_name"],
            polygon_ids_drywall_interior=polygon_ids_drywall_interior_filtered,
            polygon_drywall=dict(
                type=model_polygon["ceiling"]["drywall_assembly"]["material"],
                color=tuple(model_polygon["ceiling"]["drywall_assembly"]["color_code"]),
                type_stacked=model_polygon["ceiling"]["drywall_assembly"]["materials_vertically_stacked"],
                color_stacked=[],
                thickness=model_polygon["ceiling"]["drywall_assembly"]["thickness"],
                layers=model_polygon["ceiling"]["drywall_assembly"]["layers"],
                fire_rating=model_polygon["ceiling"]["drywall_assembly"]["fire_rating"],
                recommendation=model_polygon["ceiling"]["recommendation"],
                waste_factor=model_polygon["ceiling"]["drywall_assembly"]["waste_factor"],
                enabled=True,
            )
        )
        self._polygons.append(polygon)

    def _sketch_walls_polygon(
        self,
        vertices,
        area,
        perimeter_walls,
        polygons,
        index,
    ):
        def load_wall_payload(wall_line):
            X1, Y1, X2, Y2 = wall_line[0]
            wall_line_structured = [
                dict(x=int(X1), y=int(Y1)),
                dict(x=int(X2), y=int(Y2))
            ]
            for wall_2d in self._walls_2d:
                if wall_2d["wall_line"] == wall_line_structured:
                    return wall_2d

        polygon_ids_drywall_interior = list()
        for wall_line, polygon in zip(perimeter_walls, polygons):
            wall_payload = load_wall_payload(wall_line)
            if wall_payload:
                if len(wall_payload["polygons_drywall"]) == 2:
                    continue
                wall_payload["polygons_drywall"].append(
                    dict(
                        id=f"{wall_payload["id"]}.b",
                        room_name='',
                        polygon=polygon["coordinates"] if isinstance(polygon, dict) else polygon[0]["coordinates"],
                        type="--",
                        height=-1,
                        color=[0, 0, 0],
                        type_stacked=list(),
                        color_stacked=list(),
                        height_stacked=list(),
                        thickness=-1,
                        layers=-1,
                        fire_rating=-1,
                        recommendation='',
                        waste_factor='',
                        enabled=True
                    )
                )
                polygon_ids_drywall_interior.append(f"{wall_payload["id"]}.b")
            else:
                X1, Y1, X2, Y2 = wall_line[0]
                wall = dict(
                    id=len(self._walls_2d),
                    wall_line=[
                        dict(x=int(X1), y=int(Y1)),
                        dict(x=int(X2), y=int(Y2))
                    ],
                    thickness=-1,
                    length=-1,
                    type='',
                    openings=list(),
                    polygons_drywall=list()
                )
                wall["polygons_drywall"].append(
                    dict(
                        id=f"{len(self._walls_2d)}.a",
                        room_name='',
                        polygon=polygon["coordinates"] if isinstance(polygon, dict) else polygon[0]["coordinates"],
                        type="--",
                        height=-1,
                        color=[0, 0, 0],
                        type_stacked=list(),
                        color_stacked=list(),
                        height_stacked=list(),
                        thickness=-1,
                        layers=-1,
                        fire_rating=-1,
                        recommendation='',
                        waste_factor='',
                        enabled=True,
                    )
                )
                polygon_ids_drywall_interior.append(f"{len(self._walls_2d)}.a")
                if isinstance(polygon, list):
                    wall["polygons_drywall"].append(
                        dict(
                            id=f"{len(self._walls_2d)}.b",
                            room_name='',
                            polygon=polygon[1]["coordinates"],
                            type="--",
                            height=-1,
                            color=[0, 0, 0],
                            type_stacked=list(),
                            color_stacked=list(),
                            height_stacked=list(),
                            thickness=-1,
                            layers=-1,
                            fire_rating=-1,
                            recommendation='',
                            waste_factor='',
                            enabled=True,
                        )
                    )
                    polygon_ids_drywall_interior.append(f"{len(self._walls_2d)}.b")
                self._walls_2d.append(wall)

        polygon_ids_drywall_interior_filtered = list()
        interior_wall_ids = set()
        for polygon_id_drywall_interior in polygon_ids_drywall_interior:
            wall_id = polygon_id_drywall_interior.split('.')[0]
            if wall_id in interior_wall_ids:
                continue
            polygon_ids_drywall_interior_filtered.append(polygon_id_drywall_interior)
            interior_wall_ids.add(wall_id)

        polygon = dict(
            id=index,
            area=area,
            vertices=vertices,
            type="--",
            height=-1,
            pitch=dict(rise=-1, run=-1),
            slope_enabled=False,
            tilt_axis='',
            room_name='',
            polygon_ids_drywall_interior=polygon_ids_drywall_interior_filtered,
            polygon_drywall=dict(
                type="--",
                color=[25, 25, 25],
                type_stacked=list(),
                color_stacked=list(),
                thickness=-1,
                layers=-1,
                fire_rating=-1,
                recommendation='',
                waste_factor=-1,
                enabled=True,
            )
        )
        self._polygons.append(polygon)

    def _add_walls_polygon_given_preselection(
        self,
        polygon_preselected,
        payload_wall_parameters_preselected,
        payload_wall_drywalls_preselected,
        output_schema_custom,
        polygon_detector_and_drywall_predictor_custom_response,
        scale,
        height_default,
        floor_plan_path,
        elevation_paths,
        transcription_block_with_centroids,
        offset,
    ):
        def load_wall_payload(wall_line):
            X1, Y1, X2, Y2 = wall_line[0]
            wall_line_structured = [
                dict(x=int(X1), y=int(Y1)),
                dict(x=int(X2), y=int(Y2))
            ]
            for wall_2d in self._walls_2d:
                if wall_2d["wall_line"] == wall_line_structured:
                    return wall_2d

        scale_x, scale_y = scale
        perimeter_walls = list()
        perimeter_walls_unnormalized = list()
        for perimeter_wall in payload_wall_parameters_preselected:
            X1, Y1, X2, Y2 = perimeter_wall["wall_line"][0]['x'], perimeter_wall["wall_line"][0]['y'], perimeter_wall["wall_line"][1]['x'], perimeter_wall["wall_line"][1]['y']
            perimeter_walls.append([[X1, Y1, X2, Y2]])
            perimeter_wall_unnormalized = [[round(X1 / scale_x), round(Y1 / scale_y), round(X2 / scale_x), round(Y2 / scale_y)]]
            perimeter_walls_unnormalized.append(perimeter_wall_unnormalized)
        polygons_pts_normalized = list()
        for polygon_wall_drywall in payload_wall_drywalls_preselected:
            polygon = polygon_wall_drywall["polygon"]
            pts_normalized = np.array([
                [polygon[0]['x'], polygon[0]['y']],
                [polygon[1]['x'], polygon[1]['y']],
                [polygon[2]['x'], polygon[2]['y']],
                [polygon[3]['x'], polygon[3]['y']]
            ], np.int32)
            polygons_pts_normalized.append(pts_normalized)
        predict_polygon = self._predict_polygon(
            polygon_preselected["vertices"],
            perimeter_walls,
            polygon_preselected["area"],
            polygons_pts_normalized,
            floor_plan_path,
            elevation_paths,
            output_schema_custom,
            polygon_detector_and_drywall_predictor_custom_response,
            transcription_block_with_centroids,
            perimeter_walls_unnormalized,
            offset,
            height_default=height_default,
        )

        polygon_ids_drywall_interior = list()
        for wall_line, wall_parameter_predicted, wall_parameter_preselected, wall_drywall_preselected in zip(perimeter_walls, predict_polygon["wall_parameters"], payload_wall_parameters_preselected, payload_wall_drywalls_preselected):
            wall_payload = load_wall_payload(wall_line)
            wall_payload["thickness"] = wall_parameter_predicted.get("width") if wall_parameter_preselected["thickness"] == -1 else wall_parameter_preselected["thickness"]
            wall_payload["length"] = wall_parameter_predicted.get("length") if wall_parameter_preselected["length"] == -1 else wall_parameter_preselected["length"]
            wall_payload["type"] = wall_parameter_predicted.get("wall_type") if wall_parameter_preselected["wall_type"] == '' else wall_parameter_preselected["wall_type"]
            wall_payload["openings"] = wall_parameter_predicted.get("openings") if wall_parameter_preselected["openings"] == [] else wall_parameter_preselected["openings"]
            polygon_drywall = list(filter(lambda polygon_drywall: polygon_drywall["id"] == wall_drywall_preselected["id"], wall_payload["polygons_drywall"]))[0]
            polygon_drywall["room_name"] = wall_parameter_predicted.get("room_name") if wall_parameter_preselected["room_name"] == '' else wall_parameter_preselected["room_name"]
            polygon_drywall["type"] = wall_parameter_predicted["drywall_assembly"]["material"] if wall_drywall_preselected["type"] == "--" else wall_drywall_preselected["type"]
            polygon_drywall["height"] = wall_parameter_predicted["drywall_assembly"]["height"] if wall_drywall_preselected["height"] == -1 else wall_drywall_preselected["height"]
            polygon_drywall["color"] = list(wall_parameter_predicted["drywall_assembly"]["color_code"]) if wall_drywall_preselected["color"] == [0, 0, 0] else wall_drywall_preselected["color"],
            polygon_drywall["type_stacked"] = wall_parameter_predicted["drywall_assembly"]["materials_vertically_stacked"] if wall_drywall_preselected["type_stacked"] == [] else wall_drywall_preselected["type_stacked"]
            polygon_drywall["color_stacked"] = list(wall_parameter_predicted["drywall_assembly"]["color_codes_stacked"]) if wall_drywall_preselected["color_stacked"] == [] else wall_drywall_preselected["color_stacked"]
            polygon_drywall["height_stacked"] = wall_parameter_predicted["drywall_assembly"]["heights_stacked"] if wall_drywall_preselected["height_stacked"] == [] else wall_drywall_preselected["height_stacked"]
            polygon_drywall["thickness"] = wall_parameter_predicted["drywall_assembly"]["thickness"] if wall_drywall_preselected["thickness"] == -1 else wall_drywall_preselected["thickness"]
            polygon_drywall["layers"] = wall_parameter_predicted["drywall_assembly"]["layers"] if wall_drywall_preselected["layers"] == -1 else wall_drywall_preselected["layers"]
            polygon_drywall["fire_rating"] = wall_parameter_predicted["drywall_assembly"]["fire_rating"] if wall_drywall_preselected["fire_rating"] == -1 else wall_drywall_preselected["fire_rating"]
            polygon_drywall["recommendation"] = wall_parameter_predicted["drywall_assembly"]["recommendation"] if wall_drywall_preselected["recommendation"] == '' else wall_drywall_preselected["recommendation"]
            polygon_drywall["waste_factor"] = wall_parameter_predicted["drywall_assembly"]["waste_factor"] if wall_drywall_preselected["waste_factor"] == '' else wall_drywall_preselected["waste_factor"]
            polygon_drywall["enabled"] = True

        polygon_ids_drywall_interior_filtered = list()
        interior_wall_ids = set()
        for polygon_id_drywall_interior in polygon_ids_drywall_interior:
            wall_id = polygon_id_drywall_interior.split('.')[0]
            if wall_id in interior_wall_ids:
                continue
            polygon_ids_drywall_interior_filtered.append(polygon_id_drywall_interior)
            interior_wall_ids.add(wall_id)

        polygon_preselected["type"] = predict_polygon["ceiling"]["ceiling_type"] if polygon_preselected["type"] == "--" else polygon_preselected["type"]
        polygon_preselected["height"] = predict_polygon["ceiling"]["height"] or height_default if polygon_preselected["height"] == -1 else polygon_preselected["height"]
        polygon_preselected["pitch"] = predict_polygon["ceiling"]["pitch"] if polygon_preselected["pitch"] == {"rise": -1, "run": -1} else polygon_preselected["pitch"]
        polygon_preselected["slope_enabled"] = predict_polygon["ceiling"]["slope_enabled"] if not polygon_preselected["slope_enabled"] else polygon_preselected["slope_enabled"]
        polygon_preselected["tilt_axis"] = predict_polygon["ceiling"]["tilt_axis"] if polygon_preselected["tilt_axis"] == '' else polygon_preselected["tilt_axis"]
        polygon_preselected["room_name"] = predict_polygon["ceiling"]["room_name"] if polygon_preselected["room_name"] == '' else polygon_preselected["room_name"]
        polygon_preselected["polygon_drywall"]["type"] = predict_polygon["ceiling"]["drywall_assembly"]["material"] if polygon_preselected["polygon_drywall"]["type"] == "--" else polygon_preselected["polygon_drywall"]["type"]
        polygon_preselected["polygon_drywall"]["color"] = tuple(predict_polygon["ceiling"]["drywall_assembly"]["color_code"]) if polygon_preselected["polygon_drywall"]["color"] == [25, 25, 25] else polygon_preselected["polygon_drywall"]["color"]
        polygon_preselected["polygon_drywall"]["type_stacked"] = predict_polygon["ceiling"]["drywall_assembly"]["materials_vertically_stacked"] if polygon_preselected["polygon_drywall"]["type_stacked"] == [] else polygon_preselected["polygon_drywall"]["type_stacked"]
        polygon_preselected["polygon_drywall"]["color_stacked"] = predict_polygon["ceiling"]["drywall_assembly"]["color_stacked"] if polygon_preselected["polygon_drywall"]["color_stacked"] == [] else polygon_preselected["polygon_drywall"]["color_stacked"]
        polygon_preselected["polygon_drywall"]["thickness"] = predict_polygon["ceiling"]["drywall_assembly"]["thickness"] if polygon_preselected["polygon_drywall"]["thickness"] == -1 else polygon_preselected["polygon_drywall"]["thickness"]
        polygon_preselected["polygon_drywall"]["layers"] = predict_polygon["ceiling"]["drywall_assembly"]["layers"] if polygon_preselected["polygon_drywall"]["layers"] == -1 else polygon_preselected["polygon_drywall"]["layers"]
        polygon_preselected["polygon_drywall"]["fire_rating"] = predict_polygon["ceiling"]["drywall_assembly"]["fire_rating"] if polygon_preselected["polygon_drywall"]["fire_rating"] == -1 else polygon_preselected["polygon_drywall"]["fire_rating"]
        polygon_preselected["polygon_drywall"]["recommendation"] = predict_polygon["ceiling"]["drywall_assembly"]["recommendation"] if polygon_preselected["polygon_drywall"]["recommendation"] == '' else polygon_preselected["polygon_drywall"]["recommendation"]
        polygon_preselected["polygon_drywall"]["waste_factor"] = predict_polygon["ceiling"]["waste_factor"] if polygon_preselected["polygon_drywall"]["waste_factor"] == -1 else polygon_preselected["polygon_drywall"]["waste_factor"] 
        polygon_preselected["polygon_drywall"]["enabled"] = True

    def _add_wall_perimeter(
        self,
        wall_line,
        polygons,
        height_default,
        scale,
        thickness_default=0.29,
    ):
        X1, Y1, X2, Y2 = wall_line[0]
        wall_length_expected = round(math.hypot(
            (X1 - X2) * self._hyperparameters["modelling"]["pixel_aspect_ratio"]["horizontal"],
            (Y1 - Y2) * self._hyperparameters["modelling"]["pixel_aspect_ratio"]["vertical"]
        ), 2)
        scale_x, scale_y = scale
        wall_line_structured = [
            dict(x=round(scale_x * X1), y=round(scale_y * Y1)),
            dict(x=round(scale_x * X2), y=round(scale_y * Y2))
        ]
        wall_payload = None
        for wall_2d in self._walls_2d:
            if wall_2d["wall_line"] == wall_line_structured:
                wall_payload = wall_2d

        if wall_payload:
            wall_payload["polygons_drywall"].append(
                dict(
                    id=f"{wall_payload["id"]}.b",
                    polygon=polygons[0]["coordinates"],
                    type="DISABLED",
                    height=height_default,
                    color=[0, 0, 255],
                    type_stacked=[],
                    color_stacked=[],
                    height_stacked=[],
                    thickness=-1,
                    layers=0,
                    fire_rating=0,
                    recommendation="NA",
                    waste_factor="NA",
                    enabled=False,
                    room_name='',
                )
            )
        else:
            wall = dict(
                id=len(self._walls_2d),
                wall_line=[
                    dict(x=round(scale_x * X1), y=round(scale_y * Y1)),
                    dict(x=round(scale_x * X2), y=round(scale_y * Y2))
                ],
                thickness=thickness_default,
                length=wall_length_expected,
                polygons_drywall=list(),
                type='',
                openings=[dict(opening_type="NULL", count=0, length=0, height=0)]
            )
            for polygon, polygon_index in zip(polygons, ['a', 'b']):
                wall["polygons_drywall"].append(
                    dict(
                        id=f"{len(self._walls_2d)}.{polygon_index}",
                        polygon=polygon["coordinates"],
                        type="DISABLED",
                        height=height_default,
                        color=[0, 0, 255],
                        type_stacked=[],
                        color_stacked=[],
                        height_stacked=[],
                        thickness=-1,
                        layers=0,
                        fire_rating=0,
                        recommendation="NA",
                        waste_factor="NA",
                        enabled=False,
                        room_name='',
                    )
                )
            self._walls_2d.append(wall)

    def _extrude_polygon_perimeter(self, line, scale, outer_drywall_surface=None):
        polygons = list()
        scale_x, scale_y = scale
        X1_unnormalized, Y1_unnormalized, X2_unnormalized, Y2_unnormalized = line[0][0], line[0][1], line[0][2], line[0][3]
        orientation = self.classify_line(X1_unnormalized, Y1_unnormalized, X2_unnormalized, Y2_unnormalized)
        X1, Y1, X2, Y2 = round(scale_x * line[0][0]), round(scale_y * line[0][1]), round(scale_x * line[0][2]), round(scale_y * line[0][3])
        if orientation == "horizontal":
            polygon_a = [
                dict(x=X1+20, y=Y1-20),
                dict(x=X2-20, y=Y2-20),
                dict(x=X2-60, y=Y2-60),
                dict(x=X1+60, y=Y1-60)
            ]
            polygon_b = [
                dict(x=X1+20, y=Y1+20),
                dict(x=X2-20, y=Y2+20),
                dict(x=X2-60, y=Y2+60),
                dict(x=X1+60, y=Y1+60)
            ]
            if outer_drywall_surface == "UP" or outer_drywall_surface == "INVALID":
                polygons.append(dict(coordinates=polygon_a, enabled=False))
            if outer_drywall_surface == "DOWN" or outer_drywall_surface == "INVALID":
                polygons.append(dict(coordinates=polygon_b, enabled=False))

        if orientation == "vertical":
            polygon_a = [
                dict(x=X1-20, y=Y1+20),
                dict(x=X2-20, y=Y2-20),
                dict(x=X2-60, y=Y2-60),
                dict(x=X1-60, y=Y1+60)
            ]
            polygon_b = [
                dict(x=X1+20, y=Y1+20),
                dict(x=X2+20, y=Y2-20),
                dict(x=X2+60, y=Y2-60),
                dict(x=X1+60, y=Y1+60)
            ]
            if outer_drywall_surface == "LEFT" or outer_drywall_surface == "INVALID":
                polygons.append(dict(coordinates=polygon_a, enabled=False))
            if outer_drywall_surface == "RIGHT" or outer_drywall_surface == "INVALID":
                polygons.append(dict(coordinates=polygon_b, enabled=False))

        if orientation == "inclined":
            dx = X2 - X1
            dy = Y2 - Y1
            length = math.hypot(dx, dy)
            if length == 0:
                return polygons

            tx = dx / length
            ty = dy / length

            nx = -dy / length
            ny =  dx / length

            polygon_a = [
                dict(
                    x=int(X1 + nx * 20 + tx * 20),
                    y=int(Y1 + ny * 20 + ty * 20),
                ),
                dict(
                    x=int(X2 + nx * 20 - tx * 20),
                    y=int(Y2 + ny * 20 - ty * 20),
                ),
                dict(
                    x=int(X2 + nx * 60 - tx * 60),
                    y=int(Y2 + ny * 60 - ty * 60),
                ),
                dict(
                    x=int(X1 + nx * 60 + tx * 60),
                    y=int(Y1 + ny * 60 + ty * 60),
                ),
            ]
            polygons.append(dict(coordinates=polygon_a, enabled=False))

            polygon_b = [
                dict(
                    x=int(X1 - nx * 20 + tx * 20),
                    y=int(Y1 - ny * 20 + ty * 20),
                ),
                dict(
                    x=int(X2 - nx * 20 - tx * 20),
                    y=int(Y2 - ny * 20 - ty * 20),
                ),
                dict(
                    x=int(X2 - nx * 60 - tx * 60),
                    y=int(Y2 - ny * 60 - ty * 60),
                ),
                dict(
                    x=int(X1 - nx * 60 + tx * 60),
                    y=int(Y1 - ny * 60 + ty * 60),
                ),
            ]
            polygons.append(dict(coordinates=polygon_b, enabled=False))

        return polygons

    def _extrude_polygon_drywalls(self, polygon_perimeter_lines, polygon_vertices, scale):
        polygons = list()
        scale_x, scale_y = scale
        for polygon_perimeter_line in polygon_perimeter_lines:
            X1, Y1, X2, Y2 = polygon_perimeter_line[0][0], polygon_perimeter_line[0][1], polygon_perimeter_line[0][2], polygon_perimeter_line[0][3]
            orientation = self.classify_line(round(X1 / scale_x), round(Y1 / scale_y), round(X2 / scale_x), round(Y2 / scale_y))
            sampled_test_points = self.sample_wall_line_test_points(X1, Y1, X2, Y2)
            if orientation == "horizontal":
                polygon_up = [
                    dict(x=X1+20, y=Y1-20),
                    dict(x=X2-20, y=Y2-20),
                    dict(x=X2-60, y=Y2-60),
                    dict(x=X1+60, y=Y1-60)
                ]
                polygon_down = [
                    dict(x=X1+20, y=Y1+20),
                    dict(x=X2-20, y=Y2+20),
                    dict(x=X2-60, y=Y2+60),
                    dict(x=X1+60, y=Y1+60)
                ]
                bounded_up_down_count = dict()
                bounded_count, up_count, down_count = 0, 0, 0
                for test_point in sampled_test_points:
                    X, Y = test_point
                    if self.is_inside_polygon((X, Y - 50), polygon_vertices) and self.is_inside_polygon((X, Y + 50), polygon_vertices):
                        bounded_count += 1
                    elif self.is_inside_polygon((X, Y - 50), polygon_vertices):
                        up_count += 1
                    else:
                        down_count += 1
                bounded_up_down_count["bounded"] = bounded_count
                bounded_up_down_count["up"] = up_count
                bounded_up_down_count["down"] = down_count
                face = list(bounded_up_down_count.keys())[list(bounded_up_down_count.values()).index(max(list(bounded_up_down_count.values())))]
                if face == "bounded":
                    polygons.append([dict(coordinates=polygon_up, enabled=True), dict(coordinates=polygon_down, enabled=True)])
                elif face == "up":
                    polygons.append(dict(coordinates=polygon_up, enabled=True))
                else:
                   polygons.append(dict(coordinates=polygon_down, enabled=True)) 
            if orientation == "vertical":
                polygon_left = [
                        dict(x=X1-20, y=Y1+20),
                        dict(x=X2-20, y=Y2-20),
                        dict(x=X2-60, y=Y2-60),
                        dict(x=X1-60, y=Y1+60)
                    ]
                polygon_right = [
                    dict(x=X1+20, y=Y1+20),
                    dict(x=X2+20, y=Y2-20),
                    dict(x=X2+60, y=Y2-60),
                    dict(x=X1+60, y=Y1+60)
                ]
                bounded_left_right_count = dict()
                bounded_count, left_count, right_count = 0, 0, 0
                for test_point in sampled_test_points:
                    X, Y = test_point
                    if self.is_inside_polygon((X - 50, Y), polygon_vertices) and self.is_inside_polygon((X + 50, Y), polygon_vertices):
                        bounded_count += 1
                    elif self.is_inside_polygon((X - 50, Y), polygon_vertices):
                        left_count += 1
                    else:
                        right_count += 1
                bounded_left_right_count["bounded"] = bounded_count
                bounded_left_right_count["left"] = left_count
                bounded_left_right_count["right"] = right_count
                face = list(bounded_left_right_count.keys())[list(bounded_left_right_count.values()).index(max(list(bounded_left_right_count.values())))]
                if face == "bounded":
                    polygons.append([dict(coordinates=polygon_left, enabled=True), dict(coordinates=polygon_right, enabled=True)])
                elif face == "left":
                    polygons.append(dict(coordinates=polygon_left, enabled=True))
                else:
                    polygons.append(dict(coordinates=polygon_right, enabled=True))
            if orientation == "inclined":
                dx = X2 - X1
                dy = Y2 - Y1
                length = math.hypot(dx, dy)
                if length == 0:
                    continue

                tx = dx / length
                ty = dy / length

                nx = -dy / length
                ny =  dx / length

                polygon_A = [
                    dict(x=int(X1 + nx * 20 + tx * 20), y=int(Y1 + ny * 20 + ty * 20)),
                    dict(x=int(X2 + nx * 20 - tx * 20), y=int(Y2 + ny * 20 - ty * 20)),
                    dict(x=int(X2 + nx * 60 - tx * 60), y=int(Y2 + ny * 60 - ty * 60)),
                    dict(x=int(X1 + nx * 60 + tx * 60), y=int(Y1 + ny * 60 + ty * 60)),
                ]
                polygon_B = [
                    dict(x=int(X1 - nx * 20 + tx * 20), y=int(Y1 - ny * 20 + ty * 20)),
                    dict(x=int(X2 - nx * 20 - tx * 20), y=int(Y2 - ny * 20 - ty * 20)),
                    dict(x=int(X2 - nx * 60 - tx * 60), y=int(Y2 - ny * 60 - ty * 60)),
                    dict(x=int(X1 - nx * 60 + tx * 60), y=int(Y1 - ny * 60 + ty * 60)),
                ]

                bounded_side_A_side_B_count = dict()
                bounded_count, side_A_count, side_B_count = 0, 0, 0
                for test_point in sampled_test_points:
                    X, Y = test_point
                    test_coordinate_A = (
                        round(X + nx * 50),
                        round(Y + ny * 50)
                    )
                    test_coordinate_B = (
                        round(X - nx * 50),
                        round(Y - ny * 50)
                    )
                    if self.is_inside_polygon(test_coordinate_A, polygon_vertices) and self.is_inside_polygon(test_coordinate_B, polygon_vertices):
                        bounded_count += 1
                    elif self.is_inside_polygon(test_coordinate_A, polygon_vertices):
                        side_A_count += 1
                    else:
                        side_B_count += 1
                bounded_side_A_side_B_count["bounded"] = bounded_count
                bounded_side_A_side_B_count["side_A"] = side_A_count
                bounded_side_A_side_B_count["side_B"] = side_B_count
                face = list(bounded_side_A_side_B_count.keys())[list(bounded_side_A_side_B_count.values()).index(max(list(bounded_side_A_side_B_count.values())))]
                if face == "bounded":
                    polygons.append([dict(coordinates=polygon_A, enabled=True), dict(coordinates=polygon_B, enabled=True)])
                elif face == "side_A":
                    polygons.append(dict(coordinates=polygon_A, enabled=True))
                else:
                    polygons.append(dict(coordinates=polygon_B, enabled=True))

        return polygons

    def _normalize_walls_2d(
        self,
        walls_2d,
        scale,
        height_default,
        remove_drywall_disabled=False,
        impute_drywall_disabled=False,
        polygon_vertices_external=None
    ):
        scale_x, scale_y = scale
        if self._hyperparameters["modelling"]["scale_adoption"]["imperial_sampling"]:
            imperial_scale_X = np.median(self._imperial_scales_sampled['X'])
            imperial_scale_Y = np.median(self._imperial_scales_sampled['Y'])
        else:
            imperial_scale_X, imperial_scale_Y = self.compute_imperial_scale_from_DPI(self._scale)
        drywall_skus = [drywall_template["sku_variant"] for drywall_template in self._drywall_templates]
        drywall_sku_color_codes = [drywall_template["color_code"] for drywall_template in self._drywall_templates]
        for wall in walls_2d[:]:
            if wall["openings"]:
                walls_openings_normalized = list()
                for opening in wall["openings"]:
                    if opening["count"] != 0 or opening["length"] != 0 or opening["height"] != 0:
                        walls_openings_normalized.append(opening)
                wall["openings"] = walls_openings_normalized
            for opening in wall["openings"][:]:
                opening_types_levenshtein = list(map(lambda opening_type_choice: Levenshtein.distance(opening["opening_type"], opening_type_choice), OPENING_TYPE_CHOICES.keys()))
                target_opening_type_index = opening_types_levenshtein.index(min(opening_types_levenshtein))
                opening["opening_type"] = list(OPENING_TYPE_CHOICES.keys())[target_opening_type_index]
                opening["color"] = OPENING_TYPE_CHOICES[opening["opening_type"]]
            X1, Y1, X2, Y2 = wall["wall_line"][0]['x'], wall["wall_line"][0]['y'], wall["wall_line"][1]['x'], wall["wall_line"][1]['y']
            length_X = (X2 - X1) * imperial_scale_X
            length_Y = (Y2 - Y1) * imperial_scale_Y
            wall["length"] = round(math.hypot(length_X, length_Y), 3)
            for polygon_drywall in wall["polygons_drywall"][:]:
                polygon_drywall["layers"] = 1
                if not polygon_drywall["height"]:
                    polygon_drywall["height"] = height_default
            if impute_drywall_disabled and len(wall["polygons_drywall"]) == 2:
                if not wall["polygons_drywall"][0]["enabled"] or not wall["polygons_drywall"][1]["enabled"]:
                    centroid_A = (round(sum([vertex['x'] for vertex in wall["polygons_drywall"][0]["polygon"]]) / 4), round(sum([vertex['y'] for vertex in wall["polygons_drywall"][0]["polygon"]]) / 4))
                    centroid_B = (round(sum([vertex['x'] for vertex in wall["polygons_drywall"][1]["polygon"]]) / 4), round(sum([vertex['y'] for vertex in wall["polygons_drywall"][1]["polygon"]]) / 4))
                    reference_line = [[wall["wall_line"][0]['x'], wall["wall_line"][0]['y'], wall["wall_line"][1]['x'], wall["wall_line"][1]['y']]]
                    reference_line_unnormalized = [round(wall["wall_line"][0]['x'] / scale_x), round(wall["wall_line"][0]['y'] / scale_y), round(wall["wall_line"][1]['x'] / scale_x), round(wall["wall_line"][1]['y'] / scale_y)]
                    if self.is_inside_polygon(centroid_A, polygon_vertices_external) and self.is_inside_polygon(centroid_B, polygon_vertices_external):
                        target_lines = [[[wall_target["wall_line"][0]['x'], wall_target["wall_line"][0]['y'], wall_target["wall_line"][1]['x'], wall_target["wall_line"][1]['y']]] for wall_target in walls_2d[:]]
                        neighbors = self.nearest_neighbor(reference_line, 'A', target_lines, top_k=5)
                        valid_neighbor_found = False
                        for neighbor in neighbors:
                            neighbor_unnormalized = [round(neighbor[0][0] / scale_x), round(neighbor[0][1] / scale_y), round(neighbor[0][2] / scale_x), round(neighbor[0][3] / scale_y)]
                            for wall_ in walls_2d:
                                if neighbor == [[wall_["wall_line"][0]['x'], wall_["wall_line"][0]['y'], wall_["wall_line"][1]['x'], wall_["wall_line"][1]['y']]] and (wall_["polygons_drywall"][0]["enabled"] or wall_["polygons_drywall"][1]["enabled"]):
                                    if self.classify_line(*reference_line_unnormalized) == self.classify_line(*neighbor_unnormalized):
                                        if not wall["polygons_drywall"][0]["enabled"]:
                                            wall["polygons_drywall"][0]["color"] = wall_["polygons_drywall"][0]["color"]
                                            wall["polygons_drywall"][0]["enabled"] = wall_["polygons_drywall"][0]["enabled"]
                                            wall["polygons_drywall"][0]["fire_rating"] = wall_["polygons_drywall"][0]["fire_rating"]
                                            wall["polygons_drywall"][0]["layers"] = wall_["polygons_drywall"][0]["layers"]
                                            wall["polygons_drywall"][0]["thickness"] = wall_["polygons_drywall"][0]["thickness"]
                                            wall["polygons_drywall"][0]["type"] = wall_["polygons_drywall"][0]["type"]
                                            wall["polygons_drywall"][0]["waste_factor"] = wall_["polygons_drywall"][0]["waste_factor"]
                                            wall["polygons_drywall"][0]["recommendation"] = f"FP - Drywall material has been imputed from the nearby wall from Room: {wall_["polygons_drywall"][0]["room_name"]}."
                                        if not wall["polygons_drywall"][1]["enabled"]:
                                            wall["polygons_drywall"][1]["color"] = wall_["polygons_drywall"][1]["color"]
                                            wall["polygons_drywall"][1]["enabled"] = wall_["polygons_drywall"][1]["enabled"]
                                            wall["polygons_drywall"][1]["fire_rating"] = wall_["polygons_drywall"][1]["fire_rating"]
                                            wall["polygons_drywall"][1]["layers"] = wall_["polygons_drywall"][1]["layers"]
                                            wall["polygons_drywall"][1]["thickness"] = wall_["polygons_drywall"][1]["thickness"]
                                            wall["polygons_drywall"][1]["type"] = wall_["polygons_drywall"][1]["type"]
                                            wall["polygons_drywall"][1]["waste_factor"] = wall_["polygons_drywall"][1]["waste_factor"]
                                            wall["polygons_drywall"][1]["recommendation"] = f"FP - Drywall material has been imputed from nearby wall from Room: {wall_["polygons_drywall"][1]["room_name"]}."
                                    else:
                                        if not wall["polygons_drywall"][0]["enabled"]:
                                            wall["polygons_drywall"][0]["color"] = wall_["polygons_drywall"][1]["color"]
                                            wall["polygons_drywall"][0]["enabled"] = wall_["polygons_drywall"][1]["enabled"]
                                            wall["polygons_drywall"][0]["fire_rating"] = wall_["polygons_drywall"][1]["fire_rating"]
                                            wall["polygons_drywall"][0]["layers"] = wall_["polygons_drywall"][1]["layers"]
                                            wall["polygons_drywall"][0]["thickness"] = wall_["polygons_drywall"][1]["thickness"]
                                            wall["polygons_drywall"][0]["type"] = wall_["polygons_drywall"][1]["type"]
                                            wall["polygons_drywall"][0]["waste_factor"] = wall_["polygons_drywall"][1]["waste_factor"]
                                            wall["polygons_drywall"][0]["recommendation"] = f"FP - Drywall material has been imputed from the nearby wall from Room: {wall_["polygons_drywall"][1]["room_name"]}."
                                        if not wall["polygons_drywall"][1]["enabled"]:
                                            wall["polygons_drywall"][1]["color"] = wall_["polygons_drywall"][0]["color"]
                                            wall["polygons_drywall"][1]["enabled"] = wall_["polygons_drywall"][0]["enabled"]
                                            wall["polygons_drywall"][1]["fire_rating"] = wall_["polygons_drywall"][0]["fire_rating"]
                                            wall["polygons_drywall"][1]["layers"] = wall_["polygons_drywall"][0]["layers"]
                                            wall["polygons_drywall"][1]["thickness"] = wall_["polygons_drywall"][0]["thickness"]
                                            wall["polygons_drywall"][1]["type"] = wall_["polygons_drywall"][0]["type"]
                                            wall["polygons_drywall"][1]["waste_factor"] = wall_["polygons_drywall"][0]["waste_factor"]
                                            wall["polygons_drywall"][1]["recommendation"] = f"FP - Drywall material has been imputed from nearby wall from Room: {wall_["polygons_drywall"][0]["room_name"]}."
                                    valid_neighbor_found = True
                                    break
                            if valid_neighbor_found:
                                break
            if remove_drywall_disabled:
                if not wall["polygons_drywall"] or (not wall["polygons_drywall"][0]["enabled"] and not wall["polygons_drywall"][1]["enabled"]):
                    walls_2d.remove(wall)
                    continue
            if len(wall["polygons_drywall"]) == 2 and wall["polygons_drywall"][0]["polygon"] != wall["polygons_drywall"][1]["polygon"]:
                continue
            for polygon_drywall in wall["polygons_drywall"]:
                if polygon_drywall["type"] != "DISABLED" and polygon_drywall["type"] != "--" and polygon_drywall["type"] not in drywall_skus:
                    skus_levenshtein = list(map(lambda drywall_sku: Levenshtein.distance(polygon_drywall["type"], drywall_sku), drywall_skus))
                    target_sku_index = skus_levenshtein.index(min(skus_levenshtein))
                    polygon_drywall["type"] = drywall_skus[target_sku_index]
                    polygon_drywall["color"] = drywall_sku_color_codes[target_sku_index]
            wall_line_vertices = wall["wall_line"]
            X1, Y1, X2, Y2 = wall_line_vertices[0]['x'], wall_line_vertices[0]['y'], wall_line_vertices[1]['x'], wall_line_vertices[1]['y']
            orientation = self.classify_line(round(X1 / scale_x), round(Y1 / scale_y), round(X2 / scale_x), round(Y2 / scale_y))
            if not wall["polygons_drywall"]:
                for index in ['a', 'b']:
                    if orientation == "horizontal":
                        if index == 'a':
                            polygon_vertices = [
                                dict(x=X1+20, y=Y1-20),
                                dict(x=X2-20, y=Y2-20),
                                dict(x=X2-60, y=Y2-60),
                                dict(x=X1+60, y=Y1-60)
                            ]
                        else:
                            polygon_vertices = [
                                dict(x=X1+20, y=Y1+20),
                                dict(x=X2-20, y=Y2+20),
                                dict(x=X2-60, y=Y2+60),
                                dict(x=X1+60, y=Y1+60)
                            ]
                    if orientation == "vertical":
                        if index == 'a':   
                            polygon_vertices = [
                                dict(x=X1-20, y=Y1+20),
                                dict(x=X2-20, y=Y2-20),
                                dict(x=X2-60, y=Y2-60),
                                dict(x=X1-60, y=Y1+60)
                            ]
                        else:
                            polygon_vertices = [
                                dict(x=X1+20, y=Y1+20),
                                dict(x=X2+20, y=Y2-20),
                                dict(x=X2+60, y=Y2-60),
                                dict(x=X1+60, y=Y1+60)
                            ]
                    if orientation == "inclined":
                        dx = X2 - X1
                        dy = Y2 - Y1
                        length = math.hypot(dx, dy)
                        if length == 0:
                            continue

                        tx = dx / length
                        ty = dy / length

                        nx = -dy / length
                        ny =  dx / length

                        if index == 'a':
                            polygon_vertices = [
                                dict(
                                    x=int(X1 + nx * 20 + tx * 20),
                                    y=int(Y1 + ny * 20 + ty * 20),
                                ),
                                dict(
                                    x=int(X2 + nx * 20 - tx * 20),
                                    y=int(Y2 + ny * 20 - ty * 20),
                                ),
                                dict(
                                    x=int(X2 + nx * 60 - tx * 60),
                                    y=int(Y2 + ny * 60 - ty * 60),
                                ),
                                dict(
                                    x=int(X1 + nx * 60 + tx * 60),
                                    y=int(Y1 + ny * 60 + ty * 60),
                                 ),
                                ]
                        else:
                            polygon_vertices = [
                                dict(
                                    x=int(X1 - nx * 20 + tx * 20),
                                    y=int(Y1 - ny * 20 + ty * 20),
                                ),
                                dict(
                                    x=int(X2 - nx * 20 - tx * 20),
                                    y=int(Y2 - ny * 20 - ty * 20),
                                ),
                                dict(
                                    x=int(X2 - nx * 60 - tx * 60),
                                    y=int(Y2 - ny * 60 - ty * 60),
                                ),
                                dict(
                                    x=int(X1 - nx * 60 + tx * 60),
                                    y=int(Y1 - ny * 60 + ty * 60),
                                 ),
                                ]
                    wall["polygons_drywall"].append(
                        dict(
                            id=f"{wall["id"]}.{index}",
                            room_name='',
                            polygon=polygon_vertices,
                            type="DISABLED",
                            height=self._hyperparameters["modelling"]["height_in_feet"],
                            color=[0, 0, 255],
                            type_stacked=[],
                            color_stacked=[],
                            height_stacked=[],
                            thickness=-1,
                            layers=0,
                            fire_rating=0,
                            recommendation='',
                            waste_factor="NA",
                            enabled=False
                        )
                    )
            else:
                polygon_vertices = wall["polygons_drywall"][0]["polygon"]
                centroid_polygon_X = round(sum([vertex['x'] for vertex in polygon_vertices]) / 4)
                centroid_polygon_Y = round(sum([vertex['y'] for vertex in polygon_vertices]) / 4)
                if orientation == "horizontal":
                    if centroid_polygon_Y <= np.median([Y1, Y2]):
                        polygon_vertices = [
                            dict(x=X1+20, y=Y1+20),
                            dict(x=X2-20, y=Y2+20),
                            dict(x=X2-60, y=Y2+60),
                            dict(x=X1+60, y=Y1+60)
                        ]
                    else:
                        polygon_vertices = [
                            dict(x=X1+20, y=Y1-20),
                            dict(x=X2-20, y=Y2-20),
                            dict(x=X2-60, y=Y2-60),
                            dict(x=X1+60, y=Y1-60)
                        ]
                if orientation == "vertical":
                    if centroid_polygon_X <= np.median([X1, X2]):
                        polygon_vertices = [
                            dict(x=X1+20, y=Y1+20),
                            dict(x=X2+20, y=Y2-20),
                            dict(x=X2+60, y=Y2-60),
                            dict(x=X1+60, y=Y1+60)
                        ]
                    else:
                        polygon_vertices = [
                            dict(x=X1-20, y=Y1+20),
                            dict(x=X2-20, y=Y2-20),
                            dict(x=X2-60, y=Y2-60),
                            dict(x=X1-60, y=Y1+60)
                        ]
                if orientation == "inclined":
                    dx = X2 - X1
                    dy = Y2 - Y1
                    length = math.hypot(dx, dy)
                    if length == 0:
                        continue

                    tx = dx / length
                    ty = dy / length

                    nx = -dy / length
                    ny =  dx / length

                    mx = (X1 + X2) / 2
                    my = (Y1 + Y2) / 2

                    vx = centroid_polygon_X - mx
                    vy = centroid_polygon_Y - my

                    dot = nx * vx + ny * vy

                    if dot > 0:
                        nx, ny = -nx, -ny

                    polygon_vertices = [
                        dict(
                            x=int(X1 + nx * 20 + tx * 20),
                            y=int(Y1 + ny * 20 + ty * 20),
                        ),
                        dict(
                            x=int(X2 + nx * 20 - tx * 20),
                            y=int(Y2 + ny * 20 - ty * 20),
                        ),
                        dict(
                            x=int(X2 + nx * 60 - tx * 60),
                            y=int(Y2 + ny * 60 - ty * 60),
                        ),
                        dict(
                            x=int(X1 + nx * 60 + tx * 60),
                            y=int(Y1 + ny * 60 + ty * 60),
                        ),
                    ]
                if len(wall["polygons_drywall"]) == 2:
                    wall["polygons_drywall"].pop()
                wall["polygons_drywall"].append(
                    dict(
                        id=f"{wall["id"]}.b",
                        room_name='',
                        polygon=polygon_vertices,
                        type="DISABLED",
                        height=self._hyperparameters["modelling"]["height_in_feet"],
                        color=[0, 0, 255],
                        type_stacked=[],
                        color_stacked=[],
                        height_stacked=[],
                        thickness=-1,
                        layers=0,
                        fire_rating=0,
                        recommendation='',
                        waste_factor="NA",
                        enabled=False
                    )
                )

        return walls_2d

    def _normalize_polygons(self, polygons, walls_2d):
        polygons_valid = list()
        walls_2d_ids = [wall["id"] for wall in walls_2d]
        imperial_scale_A = np.median(self._imperial_scales_sampled['A'])
        drywall_skus = [drywall_template["sku_variant"] for drywall_template in self._drywall_templates]
        drywall_sku_color_codes = [drywall_template["color_code"] for drywall_template in self._drywall_templates]
        for polygon in polygons:
            if self._hyperparameters["modelling"]["scale_adoption"]["imperial_sampling"]:
                polygon_area_pixels = cv2.contourArea(np.array(polygon["vertices"], dtype=np.float32))
                polygon["area"] = polygon_area_pixels * imperial_scale_A
            if polygon["type"] != "--":
                ceiling_types_levenshtein = list(map(lambda ceiling_type: Levenshtein.distance(polygon["type"], ceiling_type), CEILING_CHOICES))
                target_ceiling_type_index = ceiling_types_levenshtein.index(min(ceiling_types_levenshtein))
                polygon["type"] = CEILING_CHOICES[target_ceiling_type_index]
            polygon_area_shoelace = cv2.contourArea(np.array(polygon["vertices"], dtype=np.float32))
            polygon["polygon_area_shoelace"] = polygon_area_shoelace
            perimeter_wall_missing = False
            polygon["polygon_drywall"]["layers"] = 1
            if polygon["polygon_drywall"]["type"] != "DISABLED" and polygon["polygon_drywall"]["type"] != "--" and polygon["polygon_drywall"]["type"] not in drywall_skus:
                skus_levenshtein = list(map(lambda drywall_sku: Levenshtein.distance(polygon["polygon_drywall"]["type"], drywall_sku), drywall_skus))
                target_sku_index = skus_levenshtein.index(min(skus_levenshtein))
                polygon["polygon_drywall"]["type"] = drywall_skus[target_sku_index]
                polygon["polygon_drywall"]["color"] = drywall_sku_color_codes[target_sku_index]
            if polygon["polygon_drywall"]["type"] == "DISABLED" or not polygon["polygon_drywall"]["enabled"]:
                polygon["polygon_drywall"]["color"] = [137, 137, 137]
            for drywall_id in polygon["polygon_ids_drywall_interior"]:
                wall_id = int(drywall_id.split('.')[0])
                if wall_id not in walls_2d_ids:
                    perimeter_wall_missing = True
                    break
            if not perimeter_wall_missing:
                polygons_valid.append(polygon)
        return polygons_valid

    def load_drywall_choices(self, walls_2d_JSON, polygons_2d_JSON):
        drywall_choices = ["DISABLED"] + [drywall_template["sku_variant"] for drywall_template in self._drywall_templates]
        for wall in walls_2d_JSON:
            wall["drywall_choices"] = drywall_choices
            for polygon_drywall in wall["polygons_drywall"]:
                polygon_drywall["color"] = polygon_drywall["color"][::-1]
        for polygon in polygons_2d_JSON:
            polygon["drywall_choices"] = drywall_choices
            polygon["polygon_drywall"]["color"] = polygon["polygon_drywall"]["color"][::-1]

    @property
    def drywall_choices_color_codes(self):
        drywall_choices_color_codes={drywall_template["sku_variant"]: drywall_template["color_code"][::-1] for drywall_template in self._drywall_templates}
        drywall_choices_color_codes.update(dict(DISABLED=[255, 0, 0]))
        return drywall_choices_color_codes

    def load_ceiling_choices(self, polygons_2d_JSON):
        for polygon in polygons_2d_JSON:
            polygon["type_choices"] = CEILING_CHOICES

    def load_wall_choices(self, walls_2d_JSON):
        for wall in walls_2d_JSON:
            wall["type_choices"] = WALL_CHOICES

    def _load_missing_polygons(self, walls_2d, scale, polygons_neighbor, floor_plan_path, offset):
        def load_wall_payload(wall_line):
            X1, Y1, X2, Y2 = wall_line[0]
            wall_line_structured = [
                dict(x=int(X1), y=int(Y1)),
                dict(x=int(X2), y=int(Y2))
            ]
            for wall_2d in walls_2d:
                if wall_2d["wall_line"] == wall_line_structured:
                    return wall_2d

        def load_polygon(shape, lines_isolated, tolerance=10):
            polygon_vertices = list()
            for wall_line in shape:
                X1, Y1, X2, Y2 = wall_line[0]
                if all([math.hypot(polygon_vertex[0] - X1, polygon_vertex[1] - Y1) > tolerance for polygon_vertex in polygon_vertices]):
                    polygon_vertices.append([round(X1), round(Y1)])
                if all([math.hypot(polygon_vertex[0] - X2, polygon_vertex[1] - Y2) > tolerance for polygon_vertex in polygon_vertices]):
                    polygon_vertices.append([round(X2), round(Y2)])
            for wall_line in lines_isolated:
                X1, Y1, X2, Y2 = wall_line[0]
                if all([math.hypot(polygon_vertex[0] - X1, polygon_vertex[1] - Y1) > tolerance for polygon_vertex in polygon_vertices]):
                    polygon_vertices.append([round(X1), round(Y1)])
                if all([math.hypot(polygon_vertex[0] - X2, polygon_vertex[1] - Y2) > tolerance for polygon_vertex in polygon_vertices]):
                    polygon_vertices.append([round(X2), round(Y2)])

            polygon_vertices_array = np.asarray(polygon_vertices, dtype=np.int32)
            centroid = polygon_vertices_array.mean(axis=0)
            vectors = polygon_vertices_array - centroid
            angles = np.arctan2(vectors[:, 1], vectors[:, 0])
            distances = np.sum(vectors**2, axis=1)

            order = np.lexsort((distances, angles))
            polygon_vertices = polygon_vertices_array[order].tolist()

            vertices_sorted = deepcopy(polygon_vertices)
            for index in range(len(polygon_vertices)-2):
                reference = vertices_sorted[index]
                if math.hypot(vertices_sorted[index + 1][0] - reference[0], vertices_sorted[index + 1][1] - reference[1]) > math.hypot(vertices_sorted[index + 2][0] - reference[0], vertices_sorted[index + 2][1] - reference[1]):
                    sorted_vertex_a = vertices_sorted[index + 1]
                    sorted_vertex_b = vertices_sorted[index + 2]
                    vertices_sorted.remove(sorted_vertex_a)
                    vertices_sorted.remove(sorted_vertex_b)
                    vertices_sorted.insert(index + 1, sorted_vertex_b)
                    vertices_sorted.insert(index + 2, sorted_vertex_a)
            return vertices_sorted

        def is_valid(polygon_vertices, iou_threshold=0.25):
            polygon_target = Polygon(polygon_vertices)
            if not polygon_target.is_valid:
                return False
            for polygon in polygons_neighbor:
                try:
                    polygon_neighbor = Polygon(polygon)
                except ValueError:
                    continue
                if not polygon_neighbor.is_valid:
                    polygon_neighbor = polygon_neighbor.buffer(0)
                intersection = polygon_target.intersection(polygon_neighbor).area
                union = polygon_target.union(polygon_neighbor).area
                if union == 0:
                    return False
                iou = intersection / union
                if iou > iou_threshold:
                    return False
            return True

        walls_null_room, walls_null_id, is_valid_futures = list(), list(), list()
        with ThreadPoolExecutor(max_workers=10) as executor:
            for wall in walls_2d:
                if wall["polygons_drywall"][0]["type"] == wall["polygons_drywall"][1]["type"] == "DISABLED":
                    wall_line = [[wall["wall_line"][0]['x'], wall["wall_line"][0]['y'], wall["wall_line"][1]['x'], wall["wall_line"][1]['y']]]
                    drywall_polygons = list()
                    for drywall in wall["polygons_drywall"]:
                        pts = np.array([
                            [drywall["polygon"][0]['x'], drywall["polygon"][0]['y']],
                            [drywall["polygon"][1]['x'], drywall["polygon"][1]['y']],
                            [drywall["polygon"][2]['x'], drywall["polygon"][2]['y']],
                            [drywall["polygon"][3]['x'], drywall["polygon"][3]['y']]
                        ], np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        drywall_polygons.append(pts)
                    is_valid_futures.append(executor.submit(
                        self._is_wall_valid,
                        wall_line,
                        drywall_polygons,
                        floor_plan_path,
                        offset,
                    ))
            is_wall_valid = [future.result() for future in is_valid_futures]
        is_valid_index = 0
        for wall in walls_2d[:]:
            if wall["polygons_drywall"][0]["type"] == wall["polygons_drywall"][1]["type"] == "DISABLED":
                wall_line = [[wall["wall_line"][0]['x'], wall["wall_line"][0]['y'], wall["wall_line"][1]['x'], wall["wall_line"][1]['y']]]
                if not is_wall_valid[is_valid_index]:
                    walls_2d.remove(wall)
                    is_valid_index += 1
                    continue
                walls_null_room.append(wall_line)
                walls_null_id.append(wall["id"])
                is_valid_index += 1
        shapes = self.disconnected_shapes(walls_null_room, tolerance=10*(sum(scale)//2))
        shapes_isolated = list(filter(lambda shape: len(shape) <= 2, shapes))
        lines_isolated = [shape[0] for shape in shapes_isolated]
        shapes_null = list(filter(lambda shape: len(shape) > 2, shapes))
        polygonized, perimeter_lines_contours = list(), list()
        for shape in shapes_null:
            shape_bounding_box_x_min = min([min(line[0][0], line[0][2]) for line in shape])
            shape_bounding_box_x_max = max([max(line[0][0], line[0][2]) for line in shape])
            shape_bounding_box_y_min = min([min(line[0][1], line[0][3]) for line in shape])
            shape_bounding_box_y_max = max([max(line[0][1], line[0][3]) for line in shape])
            lines_isolated_included = list()
            for line_isolated in lines_isolated:
                X1, Y1, X2, Y2 = line_isolated[0]
                X1_in_bound = (X1 >= shape_bounding_box_x_min - 20 and X1 <= shape_bounding_box_x_max + 20)
                X2_in_bound = (X2 >= shape_bounding_box_x_min - 20 and X2 <= shape_bounding_box_x_max + 20)
                Y1_in_bound = (Y1 >= shape_bounding_box_y_min - 20 and Y1 <= shape_bounding_box_y_max + 20)
                Y2_in_bound = (Y2 >= shape_bounding_box_y_min - 20 and Y2 <= shape_bounding_box_y_max + 20)
                if X1_in_bound and X2_in_bound and Y1_in_bound and Y2_in_bound:
                    lines_isolated_included.append(line_isolated)
            polygon_vertices = load_polygon(shape, lines_isolated_included)
            if not is_valid(polygon_vertices):
                continue
            perimeter_lines_contour_all = self.load_perimeter(polygon_vertices, [[[wall["wall_line"][0]['x'], wall["wall_line"][0]['y'], wall["wall_line"][1]['x'], wall["wall_line"][1]['y']]] for wall in walls_2d], scale=scale)
            imperial_scale_X, imperial_scale_Y = self.compute_imperial_scale_from_DPI(self._scale)
            imperial_scale_A = imperial_scale_X * imperial_scale_Y
            polygon_area = cv2.contourArea(np.array(polygon_vertices, np.float32)) * imperial_scale_A
            for wall_line in perimeter_lines_contour_all:
                wall_payload = load_wall_payload(wall_line)
                if wall_line in shape:
                    walls_2d.remove(wall_payload)
                    continue
                if wall_line in lines_isolated_included:
                    walls_2d.remove(wall_payload)
                    continue
                polygon_payload = wall_payload["polygons_drywall"][0] if not wall_payload["polygons_drywall"][0]["enabled"] else wall_payload["polygons_drywall"][1]
                wall_payload["polygons_drywall"].remove(polygon_payload)

            perimeter_lines_contours.append(perimeter_lines_contour_all)
            polygonized.append((polygon_area, polygon_vertices))

        return polygonized, perimeter_lines_contours, walls_2d

    def _load_schema_ceiling_drywall_assembly_given_preselection(self, payload_ceiling_drywall_preselected):
        remove_fields_polygon_drywall = set()
        remove_validators_polygon_drywall = set()
        for payload_ceiling_drywall_attribute, payload_ceiling_drywall_value in payload_ceiling_drywall_preselected.items():
            if payload_ceiling_drywall_attribute == "type" and payload_ceiling_drywall_value != "--":
                remove_fields_polygon_drywall.add("material")
            elif payload_ceiling_drywall_attribute == "color" and payload_ceiling_drywall_value != [25, 25, 25]:
                remove_fields_polygon_drywall.add("color_code")
                remove_validators_polygon_drywall.add("color_code")
            elif payload_ceiling_drywall_attribute == "type_stacked" and payload_ceiling_drywall_value:
                remove_fields_polygon_drywall.add("materials_vertically_stacked")
                remove_validators_polygon_drywall.add("stacked_layers")
            elif payload_ceiling_drywall_attribute == "color_stacked" and payload_ceiling_drywall_value:
                remove_fields_polygon_drywall.add("color_codes_stacked")
                remove_validators_polygon_drywall.add("stacked_layers")
            elif payload_ceiling_drywall_attribute == "thickness" and payload_ceiling_drywall_value != -1:
                remove_fields_polygon_drywall.add("thickness")
                remove_validators_polygon_drywall.add("thickness")
            elif payload_ceiling_drywall_attribute == "layers" and payload_ceiling_drywall_value != -1:
                remove_fields_polygon_drywall.add("layers")
                remove_validators_polygon_drywall.add("layers")
            elif payload_ceiling_drywall_attribute == "fire_rating" and payload_ceiling_drywall_value != -1:
                remove_fields_polygon_drywall.add("fire_rating")
            elif payload_ceiling_drywall_attribute == "waste_factor" and payload_ceiling_drywall_value != -1:
                remove_fields_polygon_drywall.add("waste_factor")

        return prune_model(
            DrywallAssemblyCeiling,
            remove_fields=remove_fields_polygon_drywall,
            remove_validators=remove_validators_polygon_drywall
        )

    def _load_schema_ceiling_given_preselection(self, payload_ceiling_preselected):
        drywall_assembly_ceiling_pydantic = self._load_schema_ceiling_drywall_assembly_given_preselection(
            payload_ceiling_preselected["polygon_drywall"]
        )
        remove_fields_polygon = set()
        remove_validators_polygon = set()
        if payload_ceiling_preselected["polygon_drywall"]["recommendation"] != '':
            remove_fields_polygon.add("recommendation")
        for payload_ceiling_attribute, payload_ceiling_value in payload_ceiling_preselected.items():
            if payload_ceiling_attribute == "room_name" and payload_ceiling_value != '':
                remove_fields_polygon.add("room_name")
            elif payload_ceiling_attribute == "area" and payload_ceiling_value != -1:
                remove_fields_polygon.add("area")
                remove_fields_polygon.add("confidence_area")
                remove_validators_polygon.add("area")
            elif payload_ceiling_attribute == "type" and payload_ceiling_value != "--":
                remove_fields_polygon.add("ceiling_type")
                remove_fields_polygon.add("slope_enabled")
            elif payload_ceiling_attribute == "height" and payload_ceiling_value != -1:
                remove_fields_polygon.add("height")
                remove_fields_polygon.add("confidence_height")
                remove_validators_polygon.add("height")
            elif payload_ceiling_attribute == "pitch" and payload_ceiling_value != dict(rise=-1, run=-1):
                remove_fields_polygon.add("pitch")
            elif payload_ceiling_attribute == "tilt_axis" and payload_ceiling_value != '':
                remove_fields_polygon.add("tilt_axis")

        return prune_model(
            CeilingModelAndPredict,
            remove_fields=remove_fields_polygon,
            remove_validators=remove_validators_polygon,
            fields_custom=dict(drywall_assembly=drywall_assembly_ceiling_pydantic)
        )

    def _load_schema_wall_drywall_assembly_given_preselection(self, payload_wall_drywall_preselected):
        remove_fields_wall_drywall = set()
        remove_validators_wall_drywall = set()
        for payload_wall_drywall_attribute, payload_wall_drywall_value in payload_wall_drywall_preselected.items():
            if payload_wall_drywall_attribute == "type" and payload_wall_drywall_value != "--":
                remove_fields_wall_drywall.add("material")
            elif payload_wall_drywall_attribute == "height" and payload_wall_drywall_value != -1:
                remove_fields_wall_drywall.add("height")
                remove_fields_wall_drywall.add("confidence_height")
                remove_validators_wall_drywall.add("height")
            elif payload_wall_drywall_attribute == "color" and payload_wall_drywall_value != [0, 0, 0]:
                remove_fields_wall_drywall.add("color_code")
                remove_validators_wall_drywall.add("color_code")
            elif payload_wall_drywall_attribute == "type_stacked" and payload_wall_drywall_value:
                remove_fields_wall_drywall.add("materials_vertically_stacked")
                remove_validators_wall_drywall.add("stack_count")
            elif payload_wall_drywall_attribute == "color_stacked" and payload_wall_drywall_value:
                remove_fields_wall_drywall.add("color_codes_stacked")
                remove_validators_wall_drywall.add("stack_count")
            elif payload_wall_drywall_attribute == "thickness" and payload_wall_drywall_value != -1:
                remove_fields_wall_drywall.add("thickness")
                remove_validators_wall_drywall.add("thickness")
            elif payload_wall_drywall_attribute == "layers" and payload_wall_drywall_value != -1:
                remove_fields_wall_drywall.add("layers")
            elif payload_wall_drywall_attribute == "fire_rating" and payload_wall_drywall_value != -1:
                remove_fields_wall_drywall.add("fire_rating")
            elif payload_wall_drywall_attribute == "waste_factor" and payload_wall_drywall_value != '':
                remove_fields_wall_drywall.add("waste_factor")

        return prune_model(
            DrywallAssemblyWall,
            remove_fields=remove_fields_wall_drywall,
            remove_validators=remove_validators_wall_drywall
        )

    def _load_schema_wall_parameter_given_preselection(self, payload_wall_parameter_preselected, payload_wall_drywall_preselected):
        drywall_assembly_wall_pydantic = self._load_schema_wall_drywall_assembly_given_preselection(payload_wall_drywall_preselected)
        remove_fields_wall_parameter = set()
        remove_validators_wall_parameter = set()
        if payload_wall_drywall_preselected["room_name"] != '':
            remove_fields_wall_parameter.add("room_name")
        if payload_wall_drywall_preselected["recommendation"] != '':
            remove_fields_wall_parameter.add("recommendation")
        for payload_wall_parameter_attribute, payload_wall_parameter_value in payload_wall_parameter_preselected.items():
            if payload_wall_parameter_attribute == "length" and payload_wall_parameter_value != -1:
                remove_fields_wall_parameter.add("length")
                remove_fields_wall_parameter.add("confidence_length")
                remove_validators_wall_parameter.add("length")
            elif payload_wall_parameter_attribute == "thickness" and payload_wall_parameter_value != -1:
                remove_fields_wall_parameter.add("width")
                remove_validators_wall_parameter.add("width")
            elif payload_wall_parameter_attribute == "type" and payload_wall_parameter_value != '':
                remove_fields_wall_parameter.add("wall_type")
            elif payload_wall_parameter_attribute == "openings" and payload_wall_parameter_value:
                remove_fields_wall_parameter.add("openings")

        return prune_model(
            WallParameterModelAndPredict,
            remove_fields=remove_fields_wall_parameter,
            remove_validators=remove_validators_wall_parameter,
            fields_custom=dict(drywall_assembly=drywall_assembly_wall_pydantic)
        )

    def _load_schema_polygon_detector_and_drywall_predictor_given_preselection(
        self,
        payload_ceiling_preselected,
        payload_wall_parameters_preselected,
        payload_wall_drywalls_preselected
    ):
        ceiling_pydantic = self._load_schema_ceiling_given_preselection(payload_ceiling_preselected)
        wall_parameters_pydantic = list()
        for payload_wall_parameter_preselected, payload_wall_drywall_preselected in zip(payload_wall_parameters_preselected, payload_wall_drywalls_preselected):
            wall_parameter_pydantic = self._load_schema_wall_parameter_given_preselection(payload_wall_parameter_preselected, payload_wall_drywall_preselected)
            wall_parameters_pydantic.append(wall_parameter_pydantic)

        polygon_detector_and_drywall_predictor_response = prune_model(
            PolygonDetectorAndDrywallPredictorResponse,
            fields_custom=dict(
                ceiling=ceiling_pydantic,
                wall_parameters=Tuple[*wall_parameters_pydantic]
            )
        )
        return polygon_detector_and_drywall_predictor_response, load_schema_pydantic(polygon_detector_and_drywall_predictor_response)

    def save_plot_2d(
        self,
        model_2d_path,
        floor_plan_path="/tmp/floor_plan.png",
        overlay_enabled=False
    ):
        def put_text(canvas, text, origin, fontFace, fontScale, color, thickness, angle):
            text_image = np.zeros_like(canvas)
            cv2.putText(text_image, text, origin, fontFace, fontScale, color, thickness, cv2.LINE_AA)
            M = cv2.getRotationMatrix2D(origin, angle, 1)
            rotated_text_image = cv2.warpAffine(text_image, M, (canvas.shape[1], canvas.shape[0]))
            canvas_text_added = cv2.add(canvas, rotated_text_image)
            return canvas_text_added

        with open(model_2d_path, 'r') as f:
            walls_2d_with_polygons = json.load(f)
        walls_2d, polygons = walls_2d_with_polygons

        if overlay_enabled:
            canvas = cv2.imread(floor_plan_path)
        else:
            floor_plan_image = cv2.imread(floor_plan_path)
            height, width, _ = floor_plan_image.shape
            canvas = np.ones((height, width), dtype=np.uint8) * 255
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        for polygon in polygons:
            canvas_to_overlay = canvas.copy()
            vertices = np.array(polygon["vertices"]).reshape((-1, 1, 2))
            color = tuple(polygon["polygon_drywall"]["color"])
            cv2.fillPoly(canvas_to_overlay, pts=[vertices], color=color)
            canvas = cv2.addWeighted(canvas_to_overlay, 0.3, canvas, 0.7, 0)

        for wall in walls_2d:
            self._draw_line(wall["wall_line"], canvas, overlay_enabled=overlay_enabled)
            for drywall in wall["polygons_drywall"]:
                pts = np.array([
                    [drywall["polygon"][0]['x'], drywall["polygon"][0]['y']],
                    [drywall["polygon"][1]['x'], drywall["polygon"][1]['y']],
                    [drywall["polygon"][2]['x'], drywall["polygon"][2]['y']],
                    [drywall["polygon"][3]['x'], drywall["polygon"][3]['y']]
                ], np.int32)
                pts = pts.reshape((-1, 1, 2))
                if drywall["enabled"]:
                    canvas = cv2.fillPoly(canvas, pts=[pts], color=drywall["color"])
                else:
                    canvas = cv2.fillPoly(canvas, pts=[pts], color=(0, 0, 255))
                if overlay_enabled:
                    canvas_annotation_origin_X = int(round((drywall["polygon"][0]['x'] + drywall["polygon"][1]['x'] + drywall["polygon"][2]['x'] + drywall["polygon"][3]['x']) / 4))
                    canvas_annotation_origin_Y = int(round((drywall["polygon"][0]['y'] + drywall["polygon"][1]['y'] + drywall["polygon"][2]['y'] + drywall["polygon"][3]['y']) / 4))
                    if drywall["enabled"]:
                        drywall_type = drywall["type"]
                    else:
                        drywall_type = "NULL"
                    orientation = self.classify_line(
                        wall["wall_line"][0]['x'],
                        wall["wall_line"][0]['y'],
                        wall["wall_line"][1]['x'],
                        wall["wall_line"][1]['y']
                    )
                    if orientation == "horizontal":
                        angle = 0
                    if orientation == "vertical":
                        angle = 90
                    if orientation == "inclined":
                        delta_x = wall["wall_line"][1]['x'] - wall["wall_line"][0]['x']
                        delta_y = wall["wall_line"][1]['y'] - wall["wall_line"][0]['y']
                        angle_radians = math.atan2(delta_y, delta_x)
                        angle = math.degrees(angle_radians)
                    canvas = put_text(
                        canvas,
                        drywall_type,
                        (canvas_annotation_origin_X, canvas_annotation_origin_Y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                        angle
                    )

        if overlay_enabled:
            image_path = "/tmp/blueprint_model_2d_overlay_enabled.png"
        else:
            image_path = "/tmp/blueprint_model_2d.png"
        cv2.imwrite(image_path, canvas)
        return Path(image_path)

    def _load_nearest_transcription_blocks(
        self,
        polygon_centroid,
        transcription_block_with_centroids,
        threshold=1000
    ):
        nearest_neighbors = dict()
        X, Y = polygon_centroid
        for transcription, centroid in transcription_block_with_centroids.items():
            X_target, Y_target = centroid
            if math.hypot(X - X_target, Y - Y_target) <= threshold:
                nearest_neighbors[transcription] = centroid

        return nearest_neighbors

    def model_and_predict(
        self,
        offset,
        image_path="/tmp/floor_plan_wall_segmented.png",
        elevation_paths=list(),
        model_2d_path="/tmp/walls_2d.json",
        floor_plan_path="/tmp/floor_plan.png",
        transcription_block_with_centroids=dict(),
        architectural_scale=None,
        standard_ceiling_height=None,
        allow_none_scale=False,
        trust_scale=True,
    ):
        image_GRAY = self.read_floor_plan(image_path)

        canvas = cv2.imread(floor_plan_path)
        height, width, _ = canvas.shape
        scale_x = width / 1920
        scale_y = height / 1080
        if not allow_none_scale or not trust_scale:
            height_default = self._load_ceiling_height_and_scale(
                offset,
                canvas,
                transcription_block_with_centroids,
                architectural_scale=architectural_scale,
                standard_ceiling_height=standard_ceiling_height,
                trust_scale=trust_scale,
            )["ceiling_height"]
        else:
            height_default = standard_ceiling_height if standard_ceiling_height else self._height_in_feet
            if architectural_scale:
                self._scale = self.scale_canonical(architectural_scale)
                self._is_scale_detected = True
        if not self._is_scale_detected:
            return None, None, None, None
        wall_lines = self._patch_to_line(image_GRAY, floor_plan_path, offset, (scale_x, scale_y))
        if not wall_lines:
            return None, None, None, None
        polygons, polygons_perimeter_walls, external_contour = self.polygonize(wall_lines)
        if not polygons:
            return None, None, None, None
        external_contour_normalized = [(round(scale_x * coordinate[0]), round(scale_y * coordinate[1])) for coordinate in external_contour]
        perimeter_lines, outer_drywall_surfaces = self.perimeter_lines(wall_lines)
        polygon_vertices_normalized_all = list()
        futures = list()
        with ThreadPoolExecutor(max_workers=10) as executor:
            for index, ((polygon_area, polygon_vertices), polygon_perimeter_walls) in enumerate(zip(polygons, polygons_perimeter_walls)):
                index += 1
                polygon_vertices_normalized = [(round(scale_x * vertex[0]), round(scale_y * vertex[1])) for vertex in polygon_vertices]
                polygon_vertices_normalized_all.append(polygon_vertices_normalized)
                polygon_perimeter_walls_normalized = list()
                for polygon_perimeter_wall in polygon_perimeter_walls:
                    X1, Y1, X2, Y2 = polygon_perimeter_wall[0]
                    polygon_perimeter_wall_normalized = [[round(scale_x * X1), round(scale_y * Y1), round(scale_x * X2), round(scale_y * Y2)]]
                    polygon_perimeter_walls_normalized.append(polygon_perimeter_wall_normalized)
                imperial_scale_A = self._hyperparameters["pixel_aspect_ratio_to_feet"]["area"]
                polygon_area_normalized = polygon_area * imperial_scale_A
                if not self._hyperparameters["modelling"]["scale_adoption"]["imperial_sampling"]:
                    imperial_scale_X, imperial_scale_Y = self.compute_imperial_scale_from_DPI(self._scale)
                    polygon_area_pixels = cv2.contourArea(np.array(polygon_vertices_normalized, dtype=np.float32))
                    imperial_scale_A = imperial_scale_X * imperial_scale_Y
                    polygon_area_normalized = polygon_area_pixels * imperial_scale_A
                drywall_polygons = self._extrude_polygon_drywalls(
                    polygon_perimeter_walls_normalized,
                    polygon_vertices_normalized,
                    (scale_x, scale_y)
                )
                futures.append(executor.submit(
                    self._add_walls_polygon,
                    polygon_vertices_normalized,
                    polygon_area_normalized,
                    polygon_perimeter_walls_normalized,
                    drywall_polygons,
                    (scale_x, scale_y),
                    height_default,
                    floor_plan_path,
                    elevation_paths,
                    transcription_block_with_centroids,
                    index,
                    offset,
                ))
            [future.result() for future in futures]
            futures = list()
            for perimeter_line, outer_drywall_surface in zip(perimeter_lines, outer_drywall_surfaces):
                perimeter_polygons = self._extrude_polygon_perimeter(
                    perimeter_line,
                    (scale_x, scale_y),
                    outer_drywall_surface=outer_drywall_surface
                )
                futures.append(executor.submit(
                    self._add_wall_perimeter,
                    perimeter_line,
                    perimeter_polygons,
                    height_default,
                    (scale_x, scale_y),
                ))
            [future.result() for future in futures]

        self._walls_2d = self._normalize_walls_2d(self._walls_2d, (scale_x, scale_y), height_default)
        missing_polygons, missing_polygons_perimeter_walls, self._walls_2d = self._load_missing_polygons(
            self._walls_2d,
            (scale_x, scale_y),
            polygon_vertices_normalized_all,
            floor_plan_path,
            offset,
        )
        external_contour_normalized = self.merge_polygons(external_contour_normalized, [polygon[1] for polygon in missing_polygons])
        futures = list()
        with ThreadPoolExecutor(max_workers=10) as executor:
            for index, ((polygon_area, polygon_vertices), polygon_perimeter_walls) in enumerate(zip(missing_polygons, missing_polygons_perimeter_walls)):
                index += len(polygons)
                drywall_polygons = self._extrude_polygon_drywalls(polygon_perimeter_walls, polygon_vertices, (scale_x, scale_y))
                futures.append(executor.submit(
                    self._add_walls_polygon,
                    polygon_vertices,
                    polygon_area,
                    polygon_perimeter_walls,
                    drywall_polygons,
                    (scale_x, scale_y),
                    height_default,
                    floor_plan_path,
                    elevation_paths,
                    transcription_block_with_centroids,
                    index,
                    offset,
                ))
            [future.result() for future in futures]
        self._walls_2d = self._normalize_walls_2d(self._walls_2d, (scale_x, scale_y), height_default)
        self._polygons = self._normalize_polygons(self._polygons, self._walls_2d)
        if model_2d_path:
            with open(model_2d_path, 'w') as f:
                json.dump([self._walls_2d, self._polygons], f, indent=2)
        return self._walls_2d, self._polygons, model_2d_path, external_contour_normalized

    def model(
        self,
        offset,
        image_path="/tmp/floor_plan_wall_segmented.png",
        model_2d_path="/tmp/walls_2d.json",
        floor_plan_path="/tmp/floor_plan.png",
        transcription_block_with_centroids=dict(),
        architectural_scale=None,
        standard_ceiling_height=None,
        allow_none_scale=False,
        trust_scale=True,
    ):
        image_GRAY = self.read_floor_plan(image_path)

        canvas = cv2.imread(floor_plan_path)
        height, width, _ = canvas.shape
        scale_x = width / 1920
        scale_y = height / 1080
        if not allow_none_scale or not trust_scale:
            height_default = self._load_ceiling_height_and_scale(
                offset,
                canvas,
                transcription_block_with_centroids,
                architectural_scale=architectural_scale,
                standard_ceiling_height=standard_ceiling_height,
                trust_scale=trust_scale,
            )["ceiling_height"]
        else:
            height_default = standard_ceiling_height if standard_ceiling_height else self._height_in_feet
            if architectural_scale:
                self._scale = self.scale_canonical(architectural_scale)
                self._is_scale_detected = True
        if not self._is_scale_detected:
            return None, None, None, None
        wall_lines = self._patch_to_line(image_GRAY, floor_plan_path, offset, (scale_x, scale_y))
        if not wall_lines:
            return None, None, None, None
        polygons, polygons_perimeter_walls, external_contour = self.polygonize(wall_lines)
        if not polygons:
            return None, None, None, None
        external_contour_normalized = [(round(scale_x * coordinate[0]), round(scale_y * coordinate[1])) for coordinate in external_contour]
        perimeter_lines, outer_drywall_surfaces = self.perimeter_lines(wall_lines)
        polygon_vertices_normalized_all = list()
        futures = list()
        with ThreadPoolExecutor(max_workers=10) as executor:
            for index, ((_, polygon_vertices), polygon_perimeter_walls) in enumerate(zip(polygons, polygons_perimeter_walls)):
                index += 1
                polygon_vertices_normalized = [(round(scale_x * vertex[0]), round(scale_y * vertex[1])) for vertex in polygon_vertices]
                polygon_vertices_normalized_all.append(polygon_vertices_normalized)
                polygon_perimeter_walls_normalized = list()
                for polygon_perimeter_wall in polygon_perimeter_walls:
                    X1, Y1, X2, Y2 = polygon_perimeter_wall[0]
                    polygon_perimeter_wall_normalized = [[round(scale_x * X1), round(scale_y * Y1), round(scale_x * X2), round(scale_y * Y2)]]
                    polygon_perimeter_walls_normalized.append(polygon_perimeter_wall_normalized)
                imperial_scale_X, imperial_scale_Y = self.compute_imperial_scale_from_DPI(self._scale)
                polygon_area_pixels = cv2.contourArea(np.array(polygon_vertices_normalized, dtype=np.float32))
                imperial_scale_A = imperial_scale_X * imperial_scale_Y
                polygon_area_normalized = polygon_area_pixels * imperial_scale_A
                drywall_polygons = self._extrude_polygon_drywalls(
                    polygon_perimeter_walls_normalized,
                    polygon_vertices_normalized,
                    (scale_x, scale_y)
                )
                futures.append(executor.submit(
                    self._sketch_walls_polygon,
                    polygon_vertices_normalized,
                    polygon_area_normalized,
                    polygon_perimeter_walls_normalized,
                    drywall_polygons,
                    index,
                ))
            [future.result() for future in futures]
            futures = list()
            for perimeter_line, outer_drywall_surface in zip(perimeter_lines, outer_drywall_surfaces):
                perimeter_polygons = self._extrude_polygon_perimeter(
                    perimeter_line,
                    (scale_x, scale_y),
                    outer_drywall_surface=outer_drywall_surface
                )
                futures.append(executor.submit(
                    self._add_wall_perimeter,
                    perimeter_line,
                    perimeter_polygons,
                    height_default,
                    (scale_x, scale_y),
                ))
            [future.result() for future in futures]

        self._walls_2d = self._normalize_walls_2d(self._walls_2d, (scale_x, scale_y), height_default)
        missing_polygons, missing_polygons_perimeter_walls, self._walls_2d = self._load_missing_polygons(
            self._walls_2d,
            (scale_x, scale_y),
            polygon_vertices_normalized_all,
            floor_plan_path,
            offset,
        )
        external_contour_normalized = self.merge_polygons(external_contour_normalized, [polygon[1] for polygon in missing_polygons])
        futures = list()
        with ThreadPoolExecutor(max_workers=10) as executor:
            for index, ((_, polygon_vertices), polygon_perimeter_walls) in enumerate(zip(missing_polygons, missing_polygons_perimeter_walls)):
                index += len(polygons)
                drywall_polygons = self._extrude_polygon_drywalls(polygon_perimeter_walls, polygon_vertices, (scale_x, scale_y))
                futures.append(executor.submit(
                    self._sketch_walls_polygon,
                    polygon_vertices_normalized,
                    polygon_area_normalized,
                    polygon_perimeter_walls_normalized,
                    drywall_polygons,
                    index,
                ))
            [future.result() for future in futures]
        self._walls_2d = self._normalize_walls_2d(self._walls_2d, (scale_x, scale_y), height_default)
        self._polygons = self._normalize_polygons(self._polygons, self._walls_2d)
        if model_2d_path:
            with open(model_2d_path, 'w') as f:
                json.dump([self._walls_2d, self._polygons], f, indent=2)
        return self._walls_2d, self._polygons, model_2d_path, external_contour_normalized

    def predict(
        self,
        offset,
        elevation_paths=list(),
        layout_2d_path="/tmp/layout_2d.json",
        model_2d_path="/tmp/walls_2d.json",
        floor_plan_path="/tmp/floor_plan.png",
        transcription_block_with_centroids=dict(),
        architectural_scale=None,
        standard_ceiling_height=None,
    ):
        def load_wall_payload(drywall_index):
            for wall_2d in self._walls_2d:
                if wall_2d["id"] == int(drywall_index.split('.')[0]):
                    return wall_2d

        def load_wall_polygon_drywall_payload(drywall_index):
            for wall_2d in self._walls_2d:
                for polygon_drywall in wall_2d["polygons_drywall"]:
                    if polygon_drywall["id"] == drywall_index:
                        return polygon_drywall

        with open(layout_2d_path, 'r') as f:
            self._walls_2d, self._polygons = json.load(f)

        canvas = cv2.imread(floor_plan_path)
        height, width, _ = canvas.shape
        scale_x = width / 1920
        scale_y = height / 1080
        height_default = self._load_ceiling_height_and_scale(
            offset,
            canvas,
            transcription_block_with_centroids,
            architectural_scale=architectural_scale,
            standard_ceiling_height=standard_ceiling_height,
            trust_scale=True,
        )["ceiling_height"]
        futures = list()
        with ThreadPoolExecutor(max_workers=10) as executor:
            for polygon in self._polygons:
                polygon_detector_and_drywall_predictor_response, output_schema_custom = self._load_schema_polygon_detector_and_drywall_predictor_given_preselection(
                    polygon,
                    [load_wall_payload(drywall_id) for drywall_id in polygon["polygon_ids_drywall_interior"]],
                    [load_wall_polygon_drywall_payload(drywall_id) for drywall_id in polygon["polygon_ids_drywall_interior"]],
                )
                futures.append(executor.submit(
                    self._add_drywalls_polygon_given_preselection,
                    polygon,
                    [load_wall_payload(drywall_id) for drywall_id in polygon["polygon_ids_drywall_interior"]],
                    [load_wall_polygon_drywall_payload(drywall_id) for drywall_id in polygon["polygon_ids_drywall_interior"]],
                    output_schema_custom,
                    polygon_detector_and_drywall_predictor_response,
                    (scale_x, scale_y,),
                    height_default,
                    floor_plan_path,
                    elevation_paths,
                    transcription_block_with_centroids,
                    offset
                ))
            [future.result() for future in futures]

        if model_2d_path:
            with open(model_2d_path, 'w') as f:
                json.dump([self._walls_2d, self._polygons], f, indent=2)
        return self._walls_2d, self._polygons, model_2d_path
