import os
import sys
from typing import Optional

import cv2
import numpy as np
import torch
import time
import tyro
import joblib

sys.path.insert(0, os.path.dirname(__file__) + '/..')
from data_config import SMPLX_PATH
from prompt_hmr.smpl_family import SMPLX as SMPLX_Layer
from prompt_hmr.utils.rotation_conversions import axis_angle_to_matrix
from prompt_hmr.vis.viser import viser_vis_human, viser_vis_world4d
from prompt_hmr.vis.traj import get_floor_mesh
from pipeline import Pipeline
from convert_to_gvhmr import prompthmr_to_gvhmr_structure


def main(input_video='data/examples/boxing_short.mp4', 
         static_camera=False,
         run_viser=True,
         viser_total=1500, 
         viser_subsample=1,
         convert_to_gvhmr: bool = False,
         gvhmr_output: Optional[str] = None):
    smplx = SMPLX_Layer(SMPLX_PATH).cuda()

    output_folder = 'results/' + os.path.basename(input_video).split('.')[0]
    results_path = os.path.join(output_folder, "results.pkl")

    def _run_conversion(results_obj=None):
        if not convert_to_gvhmr:
            return
        target_path = gvhmr_output or os.path.join(output_folder, "converted_gvhmr_results.pt")
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        prompthmr_data = results_obj if results_obj is not None else joblib.load(results_path)
        converted = prompthmr_to_gvhmr_structure(prompthmr_data)
        torch.save(converted, target_path)
        print(f'Saved GVHMR-formatted results to {target_path}')

    if os.path.exists(results_path):
        _run_conversion()
        return 
    pipeline = Pipeline(static_cam=static_camera)
    results = pipeline.__call__(input_video, 
                                output_folder, 
                                save_only_essential=True)
    _run_conversion(results)
    # Viser
    if run_viser:
        # Downsample for viser visualization
        images = pipeline.images[:viser_total][::viser_subsample]
        world4d = pipeline.create_world4d(step=viser_subsample, total=viser_total)
        world4d = {i:world4d[k] for i,k in enumerate(world4d)}

        # Get vertices
        all_verts = []
        for k in world4d:
            world3d = world4d[k]
            if len(world3d['track_id']) == 0: # no people
                continue
            rotmat = axis_angle_to_matrix(world3d['pose'].reshape(-1, 55, 3))
            verts = smplx(global_orient = rotmat[:,:1].cuda(),
                        body_pose = rotmat[:,1:22].cuda(),
                        betas = world3d['shape'].cuda(),
                        transl = world3d['trans'].cuda()).vertices.cpu().numpy()
            
            world3d['vertices'] = verts
            all_verts.append(torch.tensor(verts, dtype=torch.bfloat16))

        all_verts = torch.cat(all_verts)
        [gv, gf, gc] = get_floor_mesh(all_verts, scale=2)

    
        server, gui = viser_vis_world4d(images, 
                                        world4d, 
                                        smplx.faces, 
                                        floor=[gv, gf],
                                        init_fps=30/viser_subsample)
        
        url = f'https://localhost:{server.get_port()}'
        print(f'Please use this url to view the results: {url}')
        print('For longer video, it will take a few seconds for the webpage to load.')

        gui_playing, gui_timestep, gui_framerate, num_frames = gui
        while True:
            # Update the timestep if we're playing.
            if gui_playing.value:
                gui_timestep.value = (gui_timestep.value + 1) % num_frames

            time.sleep(1.0 / gui_framerate.value)
        


if __name__ == '__main__':
    tyro.cli(main)