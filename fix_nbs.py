import nbformat
import glob
import uuid

for path in glob.glob("notebooks/*.ipynb"):
    print("Fixing", path)
    nb = nbformat.read(path, as_version=4)
    # Ensure every cell has a unique string id
    for cell in nb.cells:
        if 'id' not in cell:
            cell['id'] = str(uuid.uuid4())[:8]
    # Write back
    nbformat.write(nb, path, version=4)
