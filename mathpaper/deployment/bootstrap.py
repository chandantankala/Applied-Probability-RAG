"""Build the local corpus from explicitly supplied, fingerprinted source PDFs."""
import hashlib
import json

from mathpaper.ingest import ingest_pdf
from mathpaper.store import connect, data_path


def main():
    root = data_path()
    entries = json.loads((root/"papers.json").read_text())
    missing = [entry["filename"] for entry in entries if not (root/"papers"/(entry["id"]+".pdf")).exists()]
    if missing:
        raise SystemExit("The checkout is missing required source PDFs under data/papers. Restore the fingerprinted files from the repository before building. Missing: "+", ".join(missing))
    for entry in entries:
        source = root/"papers"/(entry["id"]+".pdf")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != entry["sha256"] or digest[:16] != entry["id"]:
            raise ValueError("PDF fingerprint differs from the corpus manifest: "+entry["filename"])
        ingest_pdf(source, title=entry["title"], notes=entry["notes"], root=root)
        with connect(root) as db:
            db.execute("UPDATE papers SET filename=? WHERE id=?", (entry["filename"],entry["id"]))
    print(f"Bootstrapped {len(entries)} fingerprinted source papers.",flush=True)


if __name__ == "__main__":
    main()
