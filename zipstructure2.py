import zipfile

with zipfile.ZipFile("shapenetsem_download/ShapeNetSem.zip", "r") as z:
    names = z.namelist()
    print(f"Total entries: {len(names)}")

    # second-level folders, e.g. ShapeNetSem-backup/models-OBJ/
    second_level = sorted(set(
        n.split("/")[1] for n in names
        if n.count("/") >= 1 and len(n.split("/")) > 1 and n.split("/")[1] != ""
    ))
    print("\nSecond-level folders/files under ShapeNetSem-backup/:")
    for s in second_level:
        print(s)