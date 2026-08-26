Progress Report — Grasp Pose Estimation & Collision Detection (Transfer Learning Phase)
1. Project Context

Extending the existing robotic phone-disassembly pipeline (PointNet++ segmentation + classical FCL/CoACD collision pipeline + CollisionNet v2) with two new components:

Stage A: Grasp candidate generation + ranking (given a segmented component, propose 6-DOF grasp poses and score them)
Stage B: Pose-conditioned collision detection (given a specific grasp pose, classify collision-free vs. colliding against surrounding geometry)

Decided approach: pretrain both stages via transfer learning on an external dataset, then fine-tune on the existing 30 labeled phone meshes.

2. Dataset Selection
Considered and rejected a 2D screw-image dataset (Roboflow) for this purpose — wrong modality (2D bounding boxes vs. 3D point clouds/poses needed).
Considered and set aside GraspNet-1Billion (real RGB-D clutter, heavier to integrate) and OCID (partial-view point clouds, 2D grasp rectangles, no true collision labels) — both filed as potential future validation sets rather than primary training data.
Selected ACRONYM (NVlabs) as the primary pretraining source: ~8,836 objects, Franka Panda parallel-jaw gripper (matches project's target gripper), 2000 sampled grasp poses per object with physics-based labels.
Selected ShapeNetSem as the companion mesh dataset (ACRONYM's .h5 files reference ShapeNetSem model IDs but don't bundle geometry). Requested and received gated access via Hugging Face.
3. ACRONYM Data Exploration
Wrote explore_acronym.py to inspect .h5 internal structure.
Discovered actual schema: grasps/transforms (2000×4×4 pose matrices), grasps/qualities/flex/object_in_gripper (binary success flag — not a continuous quality score as initially assumed), plus four motion-during-closing/shaking fields (linear/angular).
Corrected the quality metric: built a combined continuous score from success × inverse motion magnitude (see filter_acronym.py), since raw ACRONYM only provides a binary label per grasp.
Noted architecturally: raw ACRONYM has no scene-level collision labels — grasps are evaluated against a single isolated object. This directly affects Stage B planning (see Section 6).
4. Category & Size Filtering
Parsed all 8,836 filenames (Category_ModelID_Scale.h5 pattern) to build a full category index.
First filtering attempt used object/mass as a secondary criterion — discarded after catching a real bug: Couch/TV/Desk passed a naive "≤500g" filter, revealing object/mass is unreliable (likely placeholder values) in this dataset.
Switched to category whitelist only (17 phone-adjacent categories: CellPhone, Mug, Bowl, Book, Pencil, DeskLamp, ToyFigure, Speaker, Camera, Stapler, Calculator, USBStick, Battery, Flashlight, PowerStrip, Watch, Wallet).
Also attempted a scale-based sanity flag; found object/scale spans ~7 orders of magnitude even within one category, so its unit convention is unclear from the field alone — flag deprioritized in favor of checking ShapeNetSem's own metadata.csv dimensions later if needed.
Final shortlist: 960 grasp files / 810 unique mesh objects across the 17 categories.
5. ShapeNetSem Download & Extraction
Downloaded only ShapeNetSem.zip (single bundled archive on the HF mirror, 12.2GB) rather than individual component files.
Inspected internal structure before extracting anything (ShapeNetSem-backup/models-OBJ/models/<model_id>.obj+.mtl, flat naming).
Wrote extract_shapenetsem.py to selectively pull only metadata CSVs + the 810 shortlisted meshes (1,620 files total, exact match confirming no silent extraction gaps) — avoided extracting the full ~12,000-object, ~12GB archive.
6. Mesh + Grasp Alignment Sanity Check (the key debugging episode)
Built visualize_grasp.py to overlay grasp-pose axis markers on loaded meshes.
Initial result looked wrong: grasp markers floated in a loose ring around each mesh (CellPhone, Mug, Book), not touching the surface.
Investigated two hypotheses systematically rather than assuming:
Mesh recentering (bounding-box center vs. raw origin) — tested directly, ruled out (centroid math confirmed correct; markers still floated by the same margin).
Missing gripper geometry — our script only drew a bare coordinate frame at the wrist pose, not the actual gripper fingers reaching forward to the contact point.
Cross-checked using ACRONYM's own official visualization tool (acronym_visualize_grasps.py from the cloned NVlabs repo), which uses the real gripper mesh.
Found and fixed a real path-resolution issue along the way: load_mesh() expects meshes nested as <mesh_root>/meshes/<Category>/<model_id>.obj (matching the object/file field inside each .h5), not our flat extraction layout — wrote restructure_for_acronym.py to copy files into the expected structure.
Result: with correct mesh path and real gripper geometry, grasps land exactly on the object edges/surface as expected (confirmed visually on CellPhone and a second thin object). Confirmed the data itself is correct — the original concern was a visualization/tooling gap, not a data defect.
7. Tooling / Housekeeping
Wrote a project .gitignore excluding datasets, mesh files, generated CSVs, model checkpoints, and other regeneratable/large artifacts.
Current Status

Data pipeline is validated end-to-end: filtered shortlist → extracted meshes → correct folder structure → confirmed-correct grasp poses via official tooling. Ready to build the actual PyTorch dataloader for Stage A training.