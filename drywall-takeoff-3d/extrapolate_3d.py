from copy import deepcopy
import json
from pathlib import Path

import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from floor_plan import FloorPlan
from gltf_generator import load_gltf

__all__ = ["Extrapolate3D"]


class Extrapolate3D(FloorPlan):

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)

        self._hyperparameters = hyperparameters["modelling"]
        self._width_in_feet = self._hyperparameters["width_in_feet"]
        self._height_in_feet = self._hyperparameters["height_in_feet"]
        self._width_in_pixels_horizontal = int(round(self._width_in_feet / self._hyperparameters["pixel_aspect_ratio"]["horizontal"]))
        self._width_in_pixels_vertical = int(round(self._width_in_feet / self._hyperparameters["pixel_aspect_ratio"]["vertical"]))
        self._height_in_pixels = int(round(self._height_in_feet / min(self._hyperparameters["pixel_aspect_ratio"]["vertical"], self._hyperparameters["pixel_aspect_ratio"]["horizontal"])))
        self._walls_3d = list()

    def _load_wall_width_in_pixels(self, wall_line, half=True):
        X1, Y1, X2, Y2 = wall_line["wall_line"][0]['x'], wall_line["wall_line"][0]['y'], wall_line["wall_line"][1]['x'], wall_line["wall_line"][1]['y']
        orientation = self.classify_line(X1, Y1, X2, Y2)
        if orientation == "horizontal":
            if half:
                return self._width_in_pixels_horizontal / 2
            return self._width_in_pixels_horizontal
        if orientation == "vertical":
            if half:
                return self._width_in_pixels_vertical / 2
            return self._width_in_pixels_vertical
        if half:
            return math.hypot(self._width_in_pixels_horizontal, self._width_in_pixels_vertical) / 2
        return math.hypot(self._width_in_pixels_horizontal, self._width_in_pixels_vertical)

    def _load_model_2d(self, model_2d_path):
        with open(model_2d_path, 'r') as f:
            lines = json.load(f)
        return lines

    def _extrude_height_polygon(self, polygon):
        height_extruded = list()
        for line in polygon:
            line_bottom = list()
            for coordinate in deepcopy(line):
                coordinate['x'] = int(coordinate['x'])
                coordinate['y'] = int(coordinate['y'])
                coordinate['z'] = 0
                line_bottom.append(coordinate)
            line_top = list()
            for coordinate in deepcopy(line[::-1]):
                coordinate['x'] = int(coordinate['x'])
                coordinate['y'] = int(coordinate['y'])
                coordinate['z'] = self._height_in_pixels
                line_top.append(coordinate)
            height_extruded.append(line_bottom+line_top)
        return height_extruded

    def _extrude_width_with_arbritrary_orientation(self, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        length = (dx**2 + dy**2) ** 0.5

        nx = -dy / length
        ny =  dx / length

        half = self._load_wall_width_in_pixels(dict(wall_line=[dict(x=x1, y=y1), dict(x=x2, y=y2)]))
        ox = nx * half
        oy = ny * half

        front_face = [
            { 'x': x1 + ox, 'y': y1 + oy},
            { 'x': x2 + ox, 'y': y2 + oy}
        ]
        back_face = [
            { 'x': x2 - ox, 'y': y2 - oy},
            { 'x': x1 - ox, 'y': y1 - oy}
        ]
        return front_face, back_face

    def _extrude_width(self, wall_line):
        X1, Y1, X2, Y2 = wall_line["wall_line"][0]['x'], wall_line["wall_line"][0]['y'], wall_line["wall_line"][1]['x'], wall_line["wall_line"][1]['y']
        orientation = self.classify_line(X1, Y1, X2, Y2)
        width_half = self._load_wall_width_in_pixels(wall_line)
        if orientation == "horizontal":
            front_face = [
                { 'x': X1, 'y': max(0, Y1 - width_half)},
                { 'x': X2, 'y': max(0, Y2 - width_half)}
            ]
            back_face = [
                { 'x': X1, 'y': Y1 + width_half},
                { 'x': X2, 'y': Y2 + width_half}
            ]
            return front_face, back_face
        if orientation == "vertical":
            front_face = [
                { 'x': max(0, X1 - width_half), 'y': Y1},
                { 'x': max(0, X2 - width_half), 'y': Y2}
            ]
            back_face = [
                { 'x': X1 + width_half, 'y': Y1},
                { 'x': X2 + width_half, 'y': Y2}
            ]
            return front_face, back_face
        if orientation == "inclined":
            return self._extrude_width_with_arbritrary_orientation(X1, Y1, X2, Y2)
        return None, None

    def _is_mitered_butt(self, wall_line, wall_line_orientation, horizontal_wall_lines, vertical_wall_lines):
        X1, Y1, X2, Y2 = wall_line["wall_line"][0]['x'], wall_line["wall_line"][0]['y'], wall_line["wall_line"][1]['x'], wall_line["wall_line"][1]['y']
        if wall_line_orientation == "horizontal":
            target_neighbors = vertical_wall_lines
        if wall_line_orientation == "vertical":
            target_neighbors = horizontal_wall_lines
        is_mitered_butt = dict(A=set(), B=set())
        for target_wall_line in target_neighbors:
            target_X1, target_Y1, target_X2, target_Y2 = target_wall_line["x1"], target_wall_line["y1"], target_wall_line["x2"], target_wall_line["y2"]
            if min(math.hypot((X1 - target_X1), (Y1 - target_Y1)), math.hypot((X1 - target_X2), (Y1 - target_Y2))) <= self._hyperparameters["tolerance_euclidean_join"]:
                if wall_line_orientation == "horizontal" and math.hypot((X1 - target_X1), (Y1 - target_Y1)) <= self._hyperparameters["tolerance_euclidean_join"]:
                    if 0 <= (Y1 - target_Y1) <= self._hyperparameters["tolerance_vertical_join"]:
                        is_mitered_butt['A'].add("orientation_A")
                    if 0 <= (target_Y1 - Y1) <= self._hyperparameters["tolerance_vertical_join"]:
                        is_mitered_butt['A'].add("orientation_B")
                if wall_line_orientation == "horizontal" and math.hypot((X1 - target_X2), (Y1 - target_Y2)) <= self._hyperparameters["tolerance_euclidean_join"]:
                    if 0 <= (Y1 - target_Y2) <= self._hyperparameters["tolerance_vertical_join"]:
                        is_mitered_butt['A'].add("orientation_A")
                    if 0 <= (target_Y2 - Y1) <= self._hyperparameters["tolerance_vertical_join"]:
                        is_mitered_butt['A'].add("orientation_B")
                if wall_line_orientation == "vertical" and math.hypot((X1 - target_X1), (Y1 - target_Y1)) <= self._hyperparameters["tolerance_euclidean_join"]:
                    if 0 <= (X1 - target_X1) <= self._hyperparameters["tolerance_horizontal_join"]:
                        is_mitered_butt['A'].add("orientation_A")
                    if 0 <= (target_X1 - X1) <= self._hyperparameters["tolerance_horizontal_join"]:
                        is_mitered_butt['A'].add("orientation_B")
                if wall_line_orientation == "vertical" and math.hypot((X1 - target_X2), (Y1 - target_Y2)) <= self._hyperparameters["tolerance_euclidean_join"]:
                    if 0 <= (X1 - target_X2) <= self._hyperparameters["tolerance_horizontal_join"]:
                        is_mitered_butt['A'].add("orientation_A")
                    if 0 <= (target_X2 - X1) <= self._hyperparameters["tolerance_horizontal_join"]:
                        is_mitered_butt['A'].add("orientation_B")
            if min(math.hypot((X2 - target_X1), (Y2 - target_Y1)), math.hypot((X2 - target_X2), (Y2 - target_Y2))) <= self._hyperparameters["tolerance_euclidean_join"]:
                if wall_line_orientation == "horizontal" and math.hypot((X2 - target_X1), (Y2 - target_Y1)) <= self._hyperparameters["tolerance_euclidean_join"]:
                    if 0 <= (Y2 - target_Y1) <= self._hyperparameters["tolerance_vertical_join"]:
                        is_mitered_butt['B'].add("orientation_A")
                    if 0 <= (target_Y1 - Y2) <= self._hyperparameters["tolerance_vertical_join"]:
                        is_mitered_butt['B'].add("orientation_B")
                if wall_line_orientation == "horizontal" and math.hypot((X2 - target_X2), (Y2 - target_Y2)) <= self._hyperparameters["tolerance_euclidean_join"]:
                    if 0 <= (Y2 - target_Y2) <= self._hyperparameters["tolerance_vertical_join"]:
                        is_mitered_butt['B'].add("orientation_A")
                    if 0 <= (target_Y2 - Y2) <= self._hyperparameters["tolerance_vertical_join"]:
                        is_mitered_butt['B'].add("orientation_B")
                if wall_line_orientation == "vertical" and math.hypot((X2 - target_X1), (Y2 - target_Y1)) <= self._hyperparameters["tolerance_euclidean_join"]:
                    if 0 <= (X2 - target_X1) <= self._hyperparameters["tolerance_horizontal_join"]:
                        is_mitered_butt['B'].add("orientation_A")
                    if 0 <= (target_X1 - X2) <= self._hyperparameters["tolerance_horizontal_join"]:
                        is_mitered_butt['B'].add("orientation_B")
                if wall_line_orientation == "vertical" and math.hypot((X2 - target_X2), (Y2 - target_Y2)) <= self._hyperparameters["tolerance_euclidean_join"]:
                    if 0 <= (X2 - target_X2) <= self._hyperparameters["tolerance_horizontal_join"]:
                        is_mitered_butt['B'].add("orientation_A")
                    if 0 <= (target_X2 - X2) <= self._hyperparameters["tolerance_horizontal_join"]:
                        is_mitered_butt['B'].add("orientation_B")

        return is_mitered_butt

    def _extrude_width_mitered_butt(self, wall_line, horizontal_wall_lines, vertical_wall_lines):
        X1, Y1, X2, Y2 = wall_line["wall_line"][0]['x'], wall_line["wall_line"][0]['y'], wall_line["wall_line"][1]['x'], wall_line["wall_line"][1]['y']
        orientation = self.classify_line(X1, Y1, X2, Y2)
        width_half = self._load_wall_width_in_pixels(wall_line)
        if orientation == "horizontal":
            mitered_butt = self._is_mitered_butt(wall_line, orientation, horizontal_wall_lines, vertical_wall_lines)
            if "orientation_A" in list(mitered_butt['A']) and "orientation_B" in list(mitered_butt['A']):
                front_face_A = {'x': X1, 'y': max(0, Y1 - width_half)}
                back_face_A = {'x': X1, 'y': Y1 + width_half}
            elif "orientation_A" in list(mitered_butt['A']):
                front_face_A = {'x': max(0, X1 - width_half), 'y': max(0, Y1 - width_half)}
                back_face_A = {'x': X1 + width_half, 'y': Y1 + width_half}
            elif "orientation_B" in list(mitered_butt['A']):
                front_face_A = {'x': X1 + width_half, 'y': max(0, Y1 - width_half)}
                back_face_A = {'x': max(0, X1 - width_half), 'y': Y1 + width_half}
            else:
                front_face_A = {'x': X1, 'y': max(0, Y1 - width_half)}
                back_face_A = {'x': X1, 'y': Y1 + width_half}

            if "orientation_A" in list(mitered_butt['B']) and "orientation_B" in list(mitered_butt['B']):
                front_face_B = {'x': X2, 'y': max(0, Y2 - width_half)}
                back_face_B = {'x': X2, 'y': Y2 + width_half}
            elif "orientation_A" in list(mitered_butt['B']):
                front_face_B = {'x': max(0, X2 - width_half), 'y': max(0, Y2 - width_half)}
                back_face_B = {'x': X2 + width_half, 'y': Y2 + width_half}
            elif "orientation_B" in list(mitered_butt['B']):
                front_face_B = {'x': X2 + width_half, 'y': max(0, Y2 - width_half)}
                back_face_B = {'x': max(0, X2 - width_half), 'y': Y2 + width_half}
            else:
                front_face_B = {'x': X2, 'y': max(0, Y2 - width_half)}
                back_face_B = {'x': X2, 'y': Y2 + width_half}

            front_face = [front_face_A, front_face_B]
            back_face = [back_face_A, back_face_B]
            return front_face, back_face
        if orientation == "vertical":
            mitered_butt = self._is_mitered_butt(wall_line, orientation, horizontal_wall_lines, vertical_wall_lines)
            if "orientation_A" in list(mitered_butt['A']) and "orientation_B" in list(mitered_butt['A']):
                front_face_A = {'x': max(0, X1 - width_half), 'y': Y1}
                back_face_A  = {'x': X1 + width_half, 'y': Y1}
            elif "orientation_A" in list(mitered_butt['A']):
                front_face_A = {'x': max(0, X1 - width_half), 'y': max(0, Y1 - width_half)}
                back_face_A  = {'x': X1 + width_half, 'y': Y1 + width_half}
            elif "orientation_B" in list(mitered_butt['A']):
                front_face_A = {'x': X1 + width_half, 'y': max(0, Y1 - width_half)}
                back_face_A  = {'x': max(0, X1 - width_half), 'y': Y1 + width_half}
            else:
                front_face_A = {'x': max(0, X1 - width_half), 'y': Y1}
                back_face_A  = {'x': X1 + width_half, 'y': Y1}

            if "orientation_A" in list(mitered_butt['B']) and "orientation_B" in list(mitered_butt['B']):
                front_face_B = {'x': max(0, X2 - width_half), 'y': Y2}
                back_face_B  = {'x': X2 + width_half, 'y': Y2}
            elif "orientation_A" in list(mitered_butt['B']):
                front_face_B = {'x': max(0, X2 - width_half), 'y': max(0, Y2 - width_half)}
                back_face_B  = {'x': X2 + width_half, 'y': Y2 + width_half}
            elif "orientation_B" in list(mitered_butt['B']):
                front_face_B = {'x': X2 + width_half, 'y': max(0, Y2 - width_half)}
                back_face_B  = {'x': max(0, X2 - width_half), 'y': Y2 + width_half}
            else:
                front_face_B = {'x': max(0, X2 - width_half), 'y': Y2}
                back_face_B  = {'x': X2 + width_half, 'y': Y2}

            front_face = [front_face_A, front_face_B]
            back_face = [back_face_A, back_face_B]
            return front_face, back_face
        if orientation == "inclined":
            return self._extrude_width_with_arbritrary_orientation(X1, Y1, X2, Y2)
        return None, None

    def _extrude_3d(self, wall_line, horizontal_wall_lines=list(), vertical_wall_lines=list()):
        if horizontal_wall_lines and vertical_wall_lines:
            front_face, back_face = self._extrude_width_mitered_butt(wall_line, horizontal_wall_lines, vertical_wall_lines)
        else:
            front_face, back_face = self._extrude_width(wall_line)
        if front_face and back_face:
            polygons = self._extrude_height_polygon([front_face, back_face])
            return polygons

    def _add_wall(self, wall_line, polygons, index):
        wall = dict(
            id=index,
            thickness=self._width_in_feet,
            height=self._height_in_feet,
            length=math.hypot((wall_line["wall_line"][0]['x'] - wall_line["wall_line"][1]['x']) * self._hyperparameters["pixel_aspect_ratio"]["horizontal"], (wall_line["wall_line"][0]['y'] - wall_line["wall_line"][1]['y']) * self._hyperparameters["pixel_aspect_ratio"]["vertical"]),
            surfaces_drywall=list()
        )
        for polygon, polygon_type in zip(polygons, wall_line["polygons_drywall"]):
            wall["surfaces_drywall"].append(
                dict(
                    polygon=polygon,
                    type=polygon_type["type"],
                    enabled=polygon_type["enabled"],
                    room_name=polygon_type["room_name"]
                )
            )
        self._walls_3d.append(wall)

    def save_plot_3d(self, model_3d_path):
        def add_side_face(ax, p1_i, p2_i, p1_o, p2_o):
            face = [
                (p1_i["x"], 1080 - p1_i["y"], p1_i["z"]),
                (p2_i["x"], 1080 - p2_i["y"], p2_i["z"]),
                (p2_o["x"], 1080 - p2_o["y"], p2_o["z"]),
                (p1_o["x"], 1080 - p1_o["y"], p1_o["z"]),
            ]

            coll = Poly3DCollection([face], alpha=0.4)
            coll.set_edgecolor('k')
            ax.add_collection3d(coll)

        with open(model_3d_path, 'r') as f:
            data = json.load(f)

        xs, ys, zs = list(), list(), list()

        dpi = 100
        fig = plt.figure(figsize=(1920 / dpi, 1080 / dpi), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        for wall in data:
            surfaces = [s for s in wall["surfaces_drywall"]]

            if len(surfaces) != 2:
                continue

            surf_A = surfaces[0]["polygon"]
            surf_B = surfaces[1]["polygon"]

            if len(surf_A) != 4 or len(surf_B) != 4:
                continue
    
            for i in range(4):
                p1_A = surf_A[i]
                p2_A = surf_A[(i + 1) % 4]

                p1_B = surf_B[i]
                p2_B = surf_B[(i + 1) % 4]

                add_side_face(ax, p1_A, p2_A, p1_B, p2_B)

            for surf in surfaces:
                poly = surf["polygon"]
                verts = [(p["x"], 1080 - p["y"], p["z"]) for p in poly]

                poly3d = [verts]
                coll = Poly3DCollection(poly3d, alpha=0.4)
                coll.set_edgecolor('k')
                ax.add_collection3d(coll)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        if xs and ys and zs:
            ax.set_xlim(0, 1920)
            ax.set_ylim(1080, 0)
            ax.set_zlim(min(zs), max(zs))

        ax.view_init(elev=90, azim=-90)

        plt.tight_layout()
        image_path = "/tmp/blueprint_model_3d.png"
        plt.savefig(image_path, dpi=dpi)
        return Path(image_path)

    def gltf(self, model_2d_path="/tmp/walls_2d.json"):
        wall_lines = self._load_model_2d(model_2d_path)
        walls = list()
        for wall_line in wall_lines:
            wall_identity = dict()
            wall_width = self._load_wall_width_in_pixels(wall_line)
            walls.append(
                dict(
                    x1=wall_line["wall_line"][0]['x'], 
                    y1=wall_line["wall_line"][0]['y'], 
                    x2=wall_line["wall_line"][1]['x'], 
                    y2=wall_line["wall_line"][1]['y'], 
                    height=self._height_in_pixels, 
                    thickness=wall_width
                )
            )
        load_gltf(walls, "/tmp/walls.gltf")
        return ["/tmp/walls.gltf", "/tmp/walls.bin"]    

    def extrapolate(self, model_2d_path="/tmp/walls_2d.json", model_3d_path="/tmp/walls_3d.json", mitered_butt_enabled=False):
        lines = self._load_model_2d(model_2d_path)
        horizontal_wall_lines, vertical_wall_lines = list(), list()
        if mitered_butt_enabled:
            for wall_line in lines:
                X1, Y1, X2, Y2 = wall_line["wall_line"][0]['x'], wall_line["wall_line"][0]['y'], wall_line["wall_line"][1]['x'], wall_line["wall_line"][1]['y']
                orientation = self.classify_line(X1, Y1, X2, Y2)
                if orientation == "horizontal":
                    horizontal_wall_lines.append(wall_line)
                if orientation == "vertical":
                    vertical_wall_lines.append(wall_line)
        for index, wall_line in enumerate(lines):
            polygons = self._extrude_3d(
                wall_line,
                horizontal_wall_lines=horizontal_wall_lines,
                vertical_wall_lines=vertical_wall_lines
            )
            if polygons:
                self._add_wall(wall_line, polygons, index)

        if model_3d_path:
            with open(model_3d_path, 'w') as f:
                json.dump(self._walls_3d, f, indent=2)
        return self._walls_3d, model_3d_path
