# RobustSkinWeightsTransferCode
![Teaser](https://www.dgp.toronto.edu/~rinat/projects/RobustSkinWeightsTransfer/teaser.jpg)
Sample code for the Siggraph Asia 2023 Technical Communications paper - [Robust Skin Weights Transfer via Weight Inpainting](https://www.dgp.toronto.edu/~rinat/projects/RobustSkinWeightsTransfer/index.html)

## Dependencies

Python bindings for [libigl](https://github.com/libigl/libigl-python-bindings)

[Polyscope](https://polyscope.run/py/installing/)

## Running
### Simple Transfer

```python
python src/sphere_to_plane_transfer.py
```

This will perform a simple transfer of weights from a sphere to the plane above it.

![SphereToPlane](imgs/SphereToPlane.png)

However, the code contains the full implementation of the method, and you can swap 
the meshes with any other meshes and load the source skinning weights.

### Body to garment transfer
(Coming soon) Load fbx files of a body and cloth meshes. Do the transfer from 
the body to cloth and write the result of the transfer into another fbx that can 
be loaded in other 3D software (Blender, Unreal, etc.).

## Cite
If you use this code for an academic publication, cite it as:
```bib
@incollection{abdrashitov2023robust,
  title={Robust Skin Weights Transfer via Weight Inpainting},
  author={Abdrashitov, Rinat and Raichstat, Kim and Monsen, Jared and Hill, David},
  booktitle={SIGGRAPH Asia 2023 Technical Communications},
  pages={1--4},
  year={2023}
}
```