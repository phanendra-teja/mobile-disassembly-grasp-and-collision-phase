import zipfile

with zipfile.ZipFile("shapenetsem_download/ShapeNetSem.zip", "r") as z:
    names = z.namelist()
    obj_entries = [n for n in names if n.startswith("ShapeNetSem-backup/models-OBJ/")]
    print(f"Total entries under models-OBJ/: {len(obj_entries)}")
    print("\nFirst 20:")
    for n in obj_entries[:20]:
        print(n)