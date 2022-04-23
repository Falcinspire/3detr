import open3d as o3d
import numpy as np

def _render_point_cloud(point_cloud, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    if color:
        pcd.paint_uniform_color(color)
    return pcd

def _render_sphere(position, size=0.2, color=[0.2, 0.2, 0.2]):
    geo = o3d.geometry.TriangleMesh.create_sphere(size)
    geo.translate(position)
    geo.paint_uniform_color(color)
    return geo

def _render_box(corners, color=[0.2, 0.2, 0.2]):
    #refs https://stackoverflow.com/a/66442894

    lines = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5], [5, 6], [6, 7], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7]]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color(color)
    return line_set

class Open3dInteractiveRendererUtil:
    def __init__(self):
        self.renderer = o3d.visualization.Visualizer()
        self.renderer.create_window()

    def draw_point_cloud(self, point_cloud):
        self.renderer.add_geometry(_render_point_cloud(point_cloud))

    def draw_sphere(self, position, size=0.2, color=[0.2, 0.2, 0.2]):
        self.renderer.add_geometry(_render_sphere(position, size, color))

    def draw_box(self, corners, color=[0.2, 0.2, 0.2]):
        self.renderer.add_geometry(_render_box(corners, color))

    def show(self):
        self.renderer.update_renderer()
        return self.renderer.run()

class Open3dOfflineRendererUtil:
    def __init__(self, img_width, img_height):
        #refs https://stackoverflow.com/a/67613280

        self.img_width = img_width
        self.img_height = img_height

        self.mat = o3d.visualization.rendering.MaterialRecord()
        self.mat.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
        self.mat.shader = "defaultUnlit"

        self.line_mat = o3d.visualization.rendering.MaterialRecord()
        self.line_mat.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
        self.line_mat.shader = "unlitLine"

        self.renderer_pc = o3d.visualization.rendering.OffscreenRenderer(img_width, img_height)
        self.renderer_pc.scene.set_background(np.array([1.0, 1.0, 1.0, 1.0]))

        vertical_field_of_view = 60.0 
        aspect_ratio = img_width / img_height  # azimuth over elevation
        near_plane = 0.1
        far_plane = 1000.0
        fov_type = o3d.visualization.rendering.Camera.FovType.Horizontal
        self.renderer_pc.scene.camera.set_projection(vertical_field_of_view, aspect_ratio, near_plane, far_plane, fov_type)

        center = [20, 0, 0]  # look_at target
        eye = [-10, -10, 0]  # camera position
        up = [0, -1, 0]  # camera orientation
        self.renderer_pc.scene.camera.look_at(center, eye, up)

        self.object_count = 0

    def draw_point_cloud(self, point_cloud):
        self.renderer_pc.scene.add_geometry(f'{self.object_count}', _render_point_cloud(point_cloud), self.mat)
        self.object_count += 1

    def draw_sphere(self, position, size=0.2, color=[0.2, 0.2, 0.2]):
        self.renderer_pc.scene.add_geometry(f'{self.object_count}', _render_sphere(position, size, color), self.mat)
        self.object_count += 1

    def draw_box(self, corners, color=[0.2, 0.2, 0.2]):
        self.renderer_pc.scene.add_geometry(f'{self.object_count}', _render_box(corners, color), self.line_mat)
        self.object_count += 1

    def clear_scene(self):
        for idx in range(self.object_count):
            self.renderer_pc.scene.remove_geometry(f'{idx}')
        self.object_count += 1

    def render_image(self, img_name, quality=9):
        image = self.renderer_pc.render_to_image()
        o3d.io.write_image(img_name, image, quality)    