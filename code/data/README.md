# Data provenance

**Dataset:** Google Quantum AI surface-code hardware record (Zenodo 6804040) with a secondary simulated-QEC archive (Zenodo 11166209).

**Original source:** https://zenodo.org/records/6804040

**Split:** Held-out depth/condition cells with development-only model selection.

**Integrity:** Source and derived data are checksum recorded under code/data/source.

Run from the repository root:

```bash
python -m pip install -e code
python code/scripts/download_data.py
```

Downloaded third-party files remain governed by the source terms documented in
`THIRD_PARTY.md`. When redistribution is not explicit, the package fetches
the data into an external cache instead of committing the source bytes.
