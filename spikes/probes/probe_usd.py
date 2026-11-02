"""Does OpenUSD read and write a physics scene on this Mac? Reading a real Isaac asset is a manual step of M0."""
import json
import os
import tempfile

from pxr import Gf, Usd, UsdGeom, UsdPhysics

path = os.path.join(tempfile.mkdtemp(), 'scene.usda')
stage = Usd.Stage.CreateNew(path)
UsdGeom.Xform.Define(stage, '/World')
box = UsdGeom.Cube.Define(stage, '/World/Box')
box.AddTranslateOp().Set(Gf.Vec3d(0, 0, 2))
UsdPhysics.RigidBodyAPI.Apply(box.GetPrim())
stage.Save()
prim = Usd.Stage.Open(path).GetPrimAtPath('/World/Box')
print(json.dumps({'probe': 'usd', 'version': list(Usd.GetVersion()), 'rigid_body': prim.HasAPI(UsdPhysics.RigidBodyAPI)}))
