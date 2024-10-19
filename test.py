import random
import numpy as np
from pathlib import Path 
import trimesh
from utils.teeth_numbering import color_mesh

from lightning.pytorch import seed_everything
import torch
import pyfqmr

from dataset.mesh_dataset import process_mesh
from dataset.preprocessing import MoveToOriginTransform, PreTransform
from models.dilated_tooth_seg_network import LitDilatedToothSegmentationNetwork

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(SEED)

torch.set_float32_matmul_precision('medium')

random.seed(SEED)

seed_everything(SEED, workers=True)



__all__ = ['infer']


def _downscale_mesh(mesh):
    """
    Downscale the mesh to a target number of faces.
    
    Args:
        mesh (trimesh.Trimesh): The input mesh.
    
    Returns:
        trimesh.Trimesh: The downscaled mesh.
    """
    mesh_simplifier = pyfqmr.Simplify()
    mesh_simplifier.setMesh(mesh.vertices, mesh.faces)
    mesh_simplifier.simplify_mesh(target_count=16000, aggressiveness=3, preserve_border=True, verbose=0,
                                  max_iterations=2000)
    new_positions, new_face, _ = mesh_simplifier.getMesh()
    mesh_simple = trimesh.Trimesh(vertices=new_positions, faces=new_face)
    vertices = mesh_simple.vertices
    faces = mesh_simple.faces
    if faces.shape[0] < 16000:
        fs_diff = 16000 - faces.shape[0]
        faces = np.append(faces, np.zeros((fs_diff, 3), dtype="int"), 0)
    elif faces.shape[0] > 16000:
        mesh_simple = trimesh.Trimesh(vertices=vertices, faces=faces)
        samples, face_index = trimesh.sample.sample_surface_even(mesh_simple, 16000)
        mesh_simple = trimesh.Trimesh(vertices=mesh_simple.vertices, faces=mesh_simple.faces[face_index])
        faces = mesh_simple.faces
        vertices = mesh_simple.vertices
    mesh_simple = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh_simple


def read_individual_obj_file(file_path):
    """
    Read a single obj file, apply the preprocessing and return the processed data for the inference.
    
    Args:
        file_path (str): The path to the obj file.
    
    Returns:
        tuple: The processed data for the inference.
    """
    
    # Load the mesh
    mesh = trimesh.load_mesh(file_path)
    
    # Downscale the mesh
    mesh = _downscale_mesh(mesh)
    
    # Apply preprocessing transformations
    move_to_origin = MoveToOriginTransform()
    mesh = move_to_origin(mesh)
    
    # Process the mesh
    mesh_faces, mesh_triangles, mesh_vertices_normals, mesh_face_normals, _ = process_mesh(mesh)
    
    # Apply pre-transform
    pre_transform = PreTransform(classes=17)
    pos, x, _ = pre_transform((mesh_faces, mesh_triangles, mesh_vertices_normals, mesh_face_normals, None))
    
    return pos, x


def infer(ckpt_path, obj_file_path, save_mesh=False, out_dir='plots', return_scene=False, use_gpu=True):
    print(f"Running inference on {obj_file_path} using checkpoint {ckpt_path}. Use GPU: {use_gpu}")

    model = LitDilatedToothSegmentationNetwork.load_from_checkpoint(ckpt_path)

    if use_gpu:
        model = model.cuda()

    # Process the individual .obj file
    pos, x = read_individual_obj_file(obj_file_path)
    data = (pos, x)

    # Perform inference
    pre_labels = model.predict_labels(data, inference=True).cpu().numpy()
    triangles = x[:, :9].reshape(-1, 3, 3)
    mesh = trimesh.Trimesh(**trimesh.triangles.to_kwargs(triangles.cpu().detach().numpy()))
    mesh_pred = color_mesh(mesh, pre_labels)
    
    ## removing the gum from the mesh prediction
    mesh_pred = mesh_pred.submesh([np.where(pre_labels != 0)[0]], append=True)
    
    ## moving the mesh to the original
    mesh_pred = MoveToOriginTransform()(mesh_pred)
    
    ## save the predicted mesh in the original shape and obj format
    mesh_pred.export(f'{out_dir}/pred.obj')
    
    if save_mesh:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        mesh_pred.export(f'{out_dir}/pred.ply')
    if return_scene:
        scene = trimesh.Scene([mesh_pred])
        return scene


if __name__ == "__main__":
    
  
    obj_file = "E:\\github\\teeth\\dilated_tooth_seg_net\\data\\3dteethseg\\raw\lower\\0AAQ6BO3\\0AAQ6BO3_lower.obj"
    ckpt_path = "E:\\github\\teeth\\dilated_tooth_seg_net\\logs\\training\\1\\epoch=89-step=54000.ckpt"
    save_mesh = True
    out_dir = 'output'
    return_scene = True
    
    scene = infer(ckpt_path, obj_file, save_mesh=save_mesh, out_dir=out_dir, return_scene=return_scene, use_gpu=True)
    print("Inference completed successfully.")



    
