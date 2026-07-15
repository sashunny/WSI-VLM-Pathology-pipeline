# Automated Digital Pathology Report (Demo)

## Slide-level patch classification summary
- **debris**: 300 patches (100.0%)

## Content-based retrieval (CLIP + FAISS)
Query results below show, for each free-text query, the most visually similar patches retrieved from the slide's embedding index:

**Query:** "densely packed dark purple cell nuclei"
  - `patch_0067_x7168_y2016.png` (similarity=0.312)
  - `patch_0009_x9184_y672.png` (similarity=0.305)
  - `patch_0066_x6944_y2016.png` (similarity=0.304)

**Query:** "pink fibrous connective tissue"
  - `patch_0296_x1344_y6272.png` (similarity=0.329)
  - `patch_0092_x10528_y2240.png` (similarity=0.329)
  - `patch_0269_x1120_y5824.png` (similarity=0.327)

**Query:** "empty white background region"
  - `patch_0064_x10752_y1792.png` (similarity=0.223)
  - `patch_0101_x8512_y2464.png` (similarity=0.214)
  - `patch_0259_x2240_y5600.png` (similarity=0.214)

**Query:** "fatty tissue with large clear vacuoles"
  - `patch_0060_x9632_y1792.png` (similarity=0.324)
  - `patch_0269_x1120_y5824.png` (similarity=0.324)
  - `patch_0114_x8288_y2688.png` (similarity=0.323)

## Caveats
- Classifier trained on PathMNIST (colorectal tissue); applied here to a different WSI purely to demonstrate pipeline mechanics.
- Retrieval uses general-purpose CLIP, not a pathology-specific VLM (e.g. PLIP/CONCH); intended as an architecture demo, not clinical output.