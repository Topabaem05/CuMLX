"""Does OpenUSD read and write a physics scene on this Mac? A real asset is the job of probe_usd_genesis.py."""
import atexit
import json
import os
import shutil
import tempfile

from pxr import Gf, Usd, UsdGeom, UsdPhysics

folder = tempfile.mkdtemp()
atexit.register(shutil.rmtree, folder, ignore_errors=True)
path = os.path.join(folder, 'scene.usda')
stage = Usd.Stage.CreateNew(path)
UsdGeom.Xform.Define(stage, '/World')
box = UsdGeom.Cube.Define(stage, '/World/Box')
box.AddTranslateOp().Set(Gf.Vec3d(0, 0, 2))
UsdPhysics.RigidBodyAPI.Apply(box.GetPrim())
stage.Save()
reopened = Usd.Stage.Open(path)  # keep the stage: a prim expires with its stage
prim = reopened.GetPrimAtPath('/World/Box')
print(json.dumps({'probe': 'usd', 'version': list(Usd.GetVersion()), 'rigid_body': prim.HasAPI(UsdPhysics.RigidBodyAPI)}))
