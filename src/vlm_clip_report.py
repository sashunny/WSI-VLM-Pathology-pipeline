"""
The VLM/RAG layer

Core Idea:
  1. Embed every WSI patch with CLIP's image encoder.
  2. Build a FAISS index over those embeddings (the "vector store").
  3. Given a free-text pathology query (e.g. "densely packed dark nuclei"), embed the
     query with CLIP's text encoder and retrieve the most similar patches -- this is
     text-to-image retrieval, the same mechanism behind RAG-for-images.
  4. Combine retrieval results + the classifier stats into a structured,
     templated report -- a simple stand-in for "VLM auto-generates diagnostic summary".


Outputs: outputs/retrieval_examples/*.png, outputs/generated_report.md
"""
import os
import glob
import re
import json
import numpy as np
import torch
import faiss
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

OUT_DIR = "outputs"
PATCH_DIR = os.path.join(OUT_DIR, "patches")
RETRIEVAL_DIR = os.path.join(OUT_DIR, "retrieval_examples")
os.makedirs(RETRIEVAL_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Example free-text queries a pathologist / downstream agent might issue.
# NOTE: vanilla CLIP is not medically trained, so treat these as a mechanics demo,
# not a claim of clinical retrieval quality
QUERIES = [
    "densely packed dark purple cell nuclei",
    "pink fibrous connective tissue",
    "empty white background region",
    "fatty tissue with large clear vacuoles",
]


def embed_images(model, processor, image_paths, batch_size=32):
    embeddings = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings.append(feats.cpu().numpy())
    return np.concatenate(embeddings, axis=0).astype("float32")


def embed_text(model, processor, texts):
    inputs = processor(text=texts, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype("float32")


def build_report(class_dist, query_results):
    """Templated structured report. Swap this for a small local LLM (e.g. flan-t5,
    or an API call) to turn the structured facts into freer prose -- the structure
    below is exactly the kind of grounding context you'd feed an LLM in a RAG setup
    so it can't hallucinate findings that aren't actually in the data."""
    lines = ["# Automated Digital Pathology Report (Demo)", ""]
    lines.append("## Slide-level patch classification summary")
    total = sum(c for _, c in class_dist)
    for name, count in class_dist:
        lines.append(f"- **{name}**: {count} patches ({count/total*100:.1f}%)")
    lines.append("")
    lines.append("## Content-based retrieval (CLIP + FAISS)")
    lines.append("Query results below show, for each free-text query, the most visually "
                  "similar patches retrieved from the slide's embedding index:")
    lines.append("")
    for query, results in query_results.items():
        lines.append(f"**Query:** \"{query}\"")
        for r in results:
            lines.append(f"  - `{r['file']}` (similarity={r['score']:.3f})")
        lines.append("")
    lines.append("## Caveats")
    lines.append("- Classifier trained on PathMNIST (colorectal tissue); applied here to a "
                  "different WSI purely to demonstrate pipeline mechanics.")
    lines.append("- Retrieval uses general-purpose CLIP, not a pathology-specific VLM "
                  "(e.g. PLIP/CONCH); intended as an architecture demo, not clinical output.")
    return "\n".join(lines)


def main():
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    patch_paths = sorted(glob.glob(os.path.join(PATCH_DIR, "*.png")))
    print(f"Embedding {len(patch_paths)} patches with CLIP...")
    image_embeds = embed_images(model, processor, patch_paths)

    #Build FAISS index (the "vector store")
    dim = image_embeds.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine sim since embeddings are L2-normalized
    index.add(image_embeds)
    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")

    # Text-to-image retrieval for each query
    query_results = {}
    for query in QUERIES:
        q_embed = embed_text(model, processor, [query])
        scores, idxs = index.search(q_embed, k=3)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            fname = os.path.basename(patch_paths[idx])
            results.append({"file": fname, "score": float(score)})
        query_results[query] = results

        # save a small contact sheet of top matches for this query
        imgs = [Image.open(patch_paths[i]).resize((150, 150)) for i in idxs[0]]
        sheet = Image.new("RGB", (150 * len(imgs), 170), "white")
        for i, im in enumerate(imgs):
            sheet.paste(im, (i * 150, 0))
        safe_name = re.sub(r"\W+", "_", query)[:40]
        sheet.save(os.path.join(RETRIEVAL_DIR, f"{safe_name}.png"))

    print(f"Saved retrieval contact sheets to {RETRIEVAL_DIR}/")

    # Pull classifier stats from last srteps console output isn't persisted, so
    # recompute a light version here if a classifier checkpoint is available; else
    # skip gracefully so this script can run standalone.
    class_dist = []
    stats_path = os.path.join(OUT_DIR, "class_distribution.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            class_dist = list(json.load(f).items())
    else:
        class_dist = [("(run step 3 first for classifier stats)", len(patch_paths))]

    report = build_report(class_dist, query_results)
    with open(os.path.join(OUT_DIR, "generated_report.md"), "w") as f:
        f.write(report)
    print(f"Saved outputs/generated_report.md")
    print("\n--- Report preview ---\n")
    print(report[:800])


if __name__ == "__main__":
    main()
