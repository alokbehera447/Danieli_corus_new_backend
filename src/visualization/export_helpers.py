"""
Visualization export helpers for automatic generation of all visualization formats.
Exports: STEP, STL, GLB, HTML (Plotly), PNG (Matplotlib 2D views)
"""

import os
import numpy as np
import trimesh
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cadquery as cq
from cadquery import exporters
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray
    name: str
    color: str = "steelblue"
    opacity: float = 0.7


class GeometryExporter:
    def __init__(self, output_dir: str = "outputs/visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _get_paths(self, name: str) -> dict:
        return {
            "step": f"{self.output_dir}/{name}.step",
            "stl": f"{self.output_dir}/{name}.stl",
            "glb": f"{self.output_dir}/{name}.glb",
            "html": f"{self.output_dir}/{name}.html",
            "png": f"{self.output_dir}/{name}.png"
        }

    def _stl_to_glb(self, stl_path: str, glb_path: str):
        mesh = trimesh.load(stl_path)
        mesh.export(glb_path, file_type='glb')

    def _load_stl_mesh(self, stl_path: str) -> MeshData:
        mesh = trimesh.load(stl_path)
        return MeshData(
            vertices=np.array(mesh.vertices),
            faces=np.array(mesh.faces),
            name=os.path.basename(stl_path).replace('.stl', '')
        )

    def _create_plotly_3d(self, meshes: List[MeshData], title: str, output_path: str):
        fig = go.Figure()

        for mesh in meshes:
            fig.add_trace(go.Mesh3d(
                x=mesh.vertices[:, 0],
                y=mesh.vertices[:, 1],
                z=mesh.vertices[:, 2],
                i=mesh.faces[:, 0],
                j=mesh.faces[:, 1],
                k=mesh.faces[:, 2],
                name=mesh.name,
                color=mesh.color,
                opacity=mesh.opacity,
                flatshading=True
            ))

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, t=40, b=0)
        )

        fig.write_html(output_path)

    def _create_matplotlib_views(self, stl_path: str, title: str, output_path: str):
        mesh = trimesh.load(stl_path)
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)

        fig = plt.figure(figsize=(12, 12))

        # 3D view
        ax3d = fig.add_subplot(2, 2, 1, projection='3d')
        poly3d = Poly3DCollection(vertices[faces], alpha=0.7, edgecolor='k', linewidth=0.1)
        poly3d.set_facecolor('steelblue')
        ax3d.add_collection3d(poly3d)
        ax3d.set_xlabel('X (mm)')
        ax3d.set_ylabel('Y (mm)')
        ax3d.set_zlabel('Z (mm)')
        ax3d.set_title('3D View')

        scale = vertices.max() - vertices.min()
        mid = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
        ax3d.set_xlim(mid[0] - scale/2, mid[0] + scale/2)
        ax3d.set_ylim(mid[1] - scale/2, mid[1] + scale/2)
        ax3d.set_zlim(mid[2] - scale/2, mid[2] + scale/2)

        # Top view (XY)
        ax_top = fig.add_subplot(2, 2, 2)
        for face in faces:
            pts = vertices[face]
            ax_top.plot(np.append(pts[:, 0], pts[0, 0]), np.append(pts[:, 1], pts[0, 1]), 'k-', linewidth=0.3)
        ax_top.fill(vertices[faces][:, :, 0].flatten(), vertices[faces][:, :, 1].flatten(), alpha=0.3, color='steelblue')
        ax_top.set_xlabel('X (mm)')
        ax_top.set_ylabel('Y (mm)')
        ax_top.set_title('Top View (XY)')
        ax_top.set_aspect('equal')
        ax_top.grid(True, alpha=0.3)

        # Front view (XZ)
        ax_front = fig.add_subplot(2, 2, 3)
        for face in faces:
            pts = vertices[face]
            ax_front.plot(np.append(pts[:, 0], pts[0, 0]), np.append(pts[:, 2], pts[0, 2]), 'k-', linewidth=0.3)
        ax_front.fill(vertices[faces][:, :, 0].flatten(), vertices[faces][:, :, 2].flatten(), alpha=0.3, color='steelblue')
        ax_front.set_xlabel('X (mm)')
        ax_front.set_ylabel('Z (mm)')
        ax_front.set_title('Front View (XZ)')
        ax_front.set_aspect('equal')
        ax_front.grid(True, alpha=0.3)

        # Side view (YZ)
        ax_side = fig.add_subplot(2, 2, 4)
        for face in faces:
            pts = vertices[face]
            ax_side.plot(np.append(pts[:, 1], pts[0, 1]), np.append(pts[:, 2], pts[0, 2]), 'k-', linewidth=0.3)
        ax_side.fill(vertices[faces][:, :, 1].flatten(), vertices[faces][:, :, 2].flatten(), alpha=0.3, color='steelblue')
        ax_side.set_xlabel('Y (mm)')
        ax_side.set_ylabel('Z (mm)')
        ax_side.set_title('Side View (YZ)')
        ax_side.set_aspect('equal')
        ax_side.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def export(self, geometry: cq.Workplane, name: str, title: Optional[str] = None,
               formats: Optional[List[str]] = None, verbose: bool = True) -> dict:
        """
        Export CadQuery geometry to multiple formats.

        Args:
            geometry: CadQuery Workplane object
            name: Base filename (without extension)
            title: Title for visualizations (defaults to name)
            formats: List of formats to export. Options: ['step', 'stl', 'glb', 'html', 'png']
                    Default: all formats
            verbose: Print export status

        Returns:
            dict of generated file paths
        """
        if title is None:
            title = name.replace('_', ' ').title()

        if formats is None:
            formats = ['step', 'stl', 'glb', 'html', 'png']

        paths = self._get_paths(name)
        exported = {}

        if 'step' in formats:
            exporters.export(geometry, paths['step'])
            exported['step'] = paths['step']

        if 'stl' in formats or 'glb' in formats or 'html' in formats or 'png' in formats:
            exporters.export(geometry, paths['stl'])
            exported['stl'] = paths['stl']

        if 'glb' in formats:
            self._stl_to_glb(paths['stl'], paths['glb'])
            exported['glb'] = paths['glb']

        if 'html' in formats:
            mesh_data = self._load_stl_mesh(paths['stl'])
            self._create_plotly_3d([mesh_data], title, paths['html'])
            exported['html'] = paths['html']

        if 'png' in formats:
            self._create_matplotlib_views(paths['stl'], title, paths['png'])
            exported['png'] = paths['png']

        if verbose:
            ext_list = ', '.join([f"{name}.{ext}" for ext in exported.keys()])
            print(f"  Exported: {ext_list}")

        return exported

    def export_combined(self, geometries: List[Tuple[cq.Workplane, str, str, float]],
                       name: str, title: str, verbose: bool = True) -> str:
        """
        Export combined visualization of multiple geometries.

        Args:
            geometries: List of (geometry, label, color, opacity) tuples
            name: Output filename (without extension)
            title: Title for visualization

        Returns:
            Path to generated HTML file
        """
        meshes = []
        union_workplane = None

        for idx, (geom, label, color, opacity) in enumerate(geometries):
            safe_label = label.replace(' ', '_').replace('/', '_')
            temp_stl = f"{self.output_dir}/_temp_{idx}_{safe_label}.stl"
            exporters.export(geom, temp_stl)

            mesh_data = self._load_stl_mesh(temp_stl)
            mesh_data.name = label
            mesh_data.color = color
            mesh_data.opacity = opacity
            meshes.append(mesh_data)

            os.remove(temp_stl)
            if label == "Stock":
                continue
            if union_workplane == None:
                union_workplane = geom
            else:
                union_workplane = union_workplane.union(geom)

        #self.export(union_workplane,name,title)
        output_path = f"{self.output_dir}/{name}.html"
        self._create_plotly_3d(meshes, title, output_path)

        if verbose:
            print(f"  Exported: {name}.html")

        return output_path


# Convenience functions for quick exports
_default_exporter = None

def get_exporter(output_dir: str = "outputs/visualizations") -> GeometryExporter:
    global _default_exporter
    if _default_exporter is None or _default_exporter.output_dir != output_dir:
        _default_exporter = GeometryExporter(output_dir)
    return _default_exporter


def export_geometry(geometry: cq.Workplane, name: str, title: Optional[str] = None,
                   output_dir: str = "outputs/visualizations",
                   formats: Optional[List[str]] = None, verbose: bool = True) -> dict:
    """
    Quick export of CadQuery geometry to all visualization formats.

    Args:
        geometry: CadQuery Workplane object
        name: Base filename
        title: Title for visualizations
        output_dir: Output directory
        formats: List of formats ['step', 'stl', 'glb', 'html', 'png']
        verbose: Print status

    Returns:
        dict of generated file paths
    """
    exporter = get_exporter(output_dir)
    return exporter.export(geometry, name, title, formats, verbose)


def export_combined_view(geometries: List[Tuple[cq.Workplane, str, str, float]],
                        name: str, title: str,
                        output_dir: str = "outputs/visualizations",
                        verbose: bool = True) -> str:
    """
    Quick export of combined visualization.

    Args:
        geometries: List of (geometry, label, color, opacity) tuples
        name: Output filename
        title: Title for visualization
        output_dir: Output directory
        verbose: Print status

    Returns:
        Path to generated HTML file
    """
    exporter = get_exporter(output_dir)
    return exporter.export_combined(geometries, name, title, verbose)
