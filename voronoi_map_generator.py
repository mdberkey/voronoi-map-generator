import numpy as np
import random
from scipy.spatial import Voronoi
from PIL import Image, ImageDraw
import argparse
import sys
from scipy.ndimage import label
import json

class VoronoiMapGenerator:
    def __init__(self, num_points, width, height, iterations, seed):
        self.num_points = num_points
        self.width = width
        self.height = height
        self.iterations = iterations
        self.seed = seed
        self.bbox = np.array([0, width, 0, height])
        self.colors = {} # region_idx -> rgb color

        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        self.points = np.random.rand(self.num_points, 2) * np.array([width, height])
        self.vor = None

    def _relax_points(self):
        for _ in range(self.iterations):
            self.vor = Voronoi(self.points)
            new_points = []
            for idx in range(len(self.points)):
                region_index = self.vor.point_region[idx]
                vertices_indices = self.vor.regions[region_index]

                if -1 in vertices_indices:
                    new_points.append(self.points[idx])
                    continue

                poly = self.vor.vertices[vertices_indices]
                
                if poly.shape[0] == 0:
                    new_points.append(self.points[idx])
                    continue

                centroid = np.mean(poly, axis=0)
                
                if not (0 <= centroid[0] < self.width and 0 <= centroid[1] < self.height):
                    new_points.append(self.points[idx])
                    continue
                
                new_points.append(centroid)

            self.points = np.array(new_points)

    def _get_polygon_vertices(self, region_index):
        if region_index == -1:
            return []
        
        vertices_indices = self.vor.regions[region_index]
        if -1 in vertices_indices:
            return self._reconstruct_infinite_region(region_index)
        
        return self.vor.vertices[vertices_indices]

    def _reconstruct_infinite_region(self, region_index):
        self.vor.ridge_vertices = np.asarray(self.vor.ridge_vertices)
        
        vertices = []
        region = self.vor.regions[region_index]
        
        if self.vor.ridge_vertices.ndim < 2 or self.vor.ridge_vertices.size == 0:
            return []
            
        point_index = np.where(self.vor.point_region == region_index)[0][0]
        
        for i, idx in enumerate(region):
            if idx != -1:
                vertices.append(self.vor.vertices[idx])
        
        start_v_idx = -1
        end_v_idx = -1
        for i in range(len(region)):
            if region[i] == -1:
                start_v_idx = region[i-1]
                end_v_idx = region[(i+1)%len(region)]
                break

        if start_v_idx == -1 or end_v_idx == -1:
            return []

        start_v = self.vor.vertices[start_v_idx]
        end_v = self.vor.vertices[end_v_idx]

        center_point = self.points[point_index]
        
        mask = np.any(self.vor.ridge_vertices == start_v_idx, axis=1) & np.any(self.vor.ridge_vertices == end_v_idx, axis=1)
        matching_ridges = self.vor.ridge_vertices[mask]
        
        if matching_ridges.size == 0:
            return []

        ridge_vertices_pair = matching_ridges[0]

        ridge_index = np.where(np.all(self.vor.ridge_vertices == ridge_vertices_pair, axis=1))[0]
        
        if ridge_index.size == 0:
            ridge_index = np.where(np.all(self.vor.ridge_vertices == ridge_vertices_pair[::-1], axis=1))[0]
            
        if ridge_index.size == 0:
            return []

        direction = np.array([end_v[1] - start_v[1], start_v[0] - end_v[0]])
        midpoint = (start_v + end_v) / 2
        
        if np.dot(direction, center_point - midpoint) < 0:
            direction = -direction

        far_point1 = start_v + direction * max(self.width, self.height) * 2
        far_point2 = end_v + direction * max(self.width, self.height) * 2
        
        new_vertices = list(vertices)

        idx = 0
        while idx < len(new_vertices):
            if np.array_equal(new_vertices[idx], start_v):
                new_vertices.insert(idx + 1, far_point1)
                break
            idx += 1
            
        idx = 0
        while idx < len(new_vertices):
            if np.array_equal(new_vertices[idx], end_v):
                new_vertices.insert(idx, far_point2)
                break
            idx += 1
            
        if len(new_vertices) == len(vertices) + 2:
            vertices = new_vertices
            
        return self._clip_polygon(vertices)


    def _clip_polygon(self, polygon):
        clipped = list(polygon)
        for i in range(4):
            edge = i
            input_list = list(clipped)
            clipped.clear()
            
            if not input_list:
                return []

            s = input_list[-1]
            for j in range(len(input_list)):
                e = input_list[j]
                s_inside = self._is_inside(s, edge)
                e_inside = self._is_inside(e, edge)

                if e_inside:
                    if not s_inside:
                        intersection = self._compute_intersection(s, e, edge)
                        clipped.append(intersection)
                    clipped.append(e)
                elif s_inside:
                    intersection = self._compute_intersection(s, e, edge)
                    clipped.append(intersection)
                s = e
        return np.array(clipped)

    def _is_inside(self, p, edge):
        if edge == 0: return p[0] >= self.bbox[0]
        if edge == 1: return p[0] <= self.bbox[1]
        if edge == 2: return p[1] >= self.bbox[2]
        if edge == 3: return p[1] <= self.bbox[3]

    def _compute_intersection(self, s, e, edge):
        x1, y1 = s
        x2, y2 = e

        if edge == 0: # Left
            x = self.bbox[0]
            y = y1 + (y2 - y1) * (x - x1) / (x2 - x1) if x1 != x2 else y1
            return [x, y]
        if edge == 1: # Right
            x = self.bbox[1]
            y = y1 + (y2 - y1) * (x - x1) / (x2 - x1) if x1 != x2 else y1
            return [x, y]
        if edge == 2: # Top
            y = self.bbox[2]
            x = x1 + (x2 - x1) * (y - y1) / (y2 - y1) if y1 != y2 else x1
            return [x, y]
        if edge == 3: # Bottom
            y = self.bbox[3]
            x = x1 + (x2 - x1) * (y - y1) / (y2 - y1) if y1 != y2 else x1
            return [x, y]
    
    def _fill_gaps_with_color(self, img):
        img_array = np.array(img)
        
        gap_mask = np.sum(img_array, axis=2) < 10

        labeled_array, num_features = label(gap_mask)
        
        if num_features == 0:
            return Image.fromarray(img_array)
        
        region_idx = np.int64(len(self.colors))
        for i in range(1, num_features + 1):
            new_color = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255)
            )
            gap_pixels = labeled_array == i
            
            img_array[gap_pixels] = new_color

            self.colors[region_idx] = new_color
            region_idx += 1
        
        return Image.fromarray(img_array)

    def _generate_regions_txt(self):
        formatted_colors = {f"#{r:02x}{g:02x}{b:02x}": int(k) for k, (r, g, b) in sorted(self.colors.items())}

        with open(f"regions_{self.seed}.json", "w") as f:
            json.dump(formatted_colors, f, indent=2)

    def generate(self):
        self._relax_points()
        self.vor = Voronoi(self.points)
        
        id_map = Image.new('RGB', (self.width, self.height))
        border_map = Image.new('L', (self.width, self.height), 0)
        id_draw = ImageDraw.Draw(id_map)
        border_draw = ImageDraw.Draw(border_map)

        for i in range(len(self.points)):
            region_idx = self.vor.point_region[i]
            
            if region_idx not in self.colors:
                self.colors[region_idx] = (
                    random.randint(50, 255),
                    random.randint(50, 255),
                    random.randint(50, 255)
                )

            polygon_vertices = self._get_polygon_vertices(region_idx)
            
            if polygon_vertices is None or len(polygon_vertices) < 3:
                continue

            clipped_poly = self._clip_polygon(polygon_vertices)

            if len(clipped_poly) > 0:
                flat_poly = [tuple(p) for p in clipped_poly]
                id_draw.polygon(flat_poly, fill=self.colors[region_idx])
                border_draw.line(flat_poly + [flat_poly[0]], fill=255, width=1)

        id_map = self._fill_gaps_with_color(id_map)
        self._generate_regions_txt()

        w = self.width - 1
        h = self.height - 1
        border_width = 1
        border_draw.line([(0, 0), (w, 0)], fill=255, width=border_width)
        border_draw.line([(0, h), (w, h)], fill=255, width=border_width)
        border_draw.line([(0, 0), (0, h)], fill=255, width=border_width)
        border_draw.line([(w, 0), (w, h)], fill=255, width=border_width)

        id_map.save(f"id_map_{self.seed}.png")
        border_map.save(f"border_map_{self.seed}.png")
        print("Generated maps")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate relaxed Voronoi diagrams.")
    parser.add_argument('--points', type=int, default=100, help='Number of points to generate.')
    parser.add_argument('--width', type=int, default=800, help='Width of the output image.')
    parser.add_argument('--height', type=int, default=600, help='Height of the output image.')
    parser.add_argument('--iterations', type=int, default=2, help="Number of Lloyd's algorithm iterations.")
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')
    
    args = parser.parse_args()

    if args.points <= 0 or args.width <= 0 or args.height <= 0 or args.iterations < 0:
        print("Error: points, width, height must be positive, and iterations must be non-negative.", file=sys.stderr)
        sys.exit(1)
        
    generator = VoronoiMapGenerator(
        num_points=args.points,
        width=args.width,
        height=args.height,
        iterations=args.iterations,
        seed=args.seed
    )
    generator.generate()