import zipfile

with zipfile.ZipFile("shapenetsem_download/ShapeNetSem.zip", "r") as z:
    names = z.namelist()
    print(f"Total entries: {len(names)}")
    # print unique top-level folders/prefixes
    prefixes = sorted(set(n.split("/")[0] for n in names if "/" in n))
    for p in prefixes:
        print(p)
    print("\nFirst 30 entries:")
    for n in names[:30]:
        print(n)