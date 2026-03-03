"""
Benchmark: bge-m3-onnx (ONNX INT8) vs pplx-embed-v1-0.6B (SentenceTransformers)

Compares embedding quality and latency on French notarial domain data.
Models are loaded locally (no API calls).

Metrics:
  - Precision@3: fraction of top-3 retrieved docs that are relevant
  - Top-1 accuracy: whether the most similar doc is relevant
  - Cosine similarity distributions (relevant vs irrelevant)
  - Score spread (gap between best relevant and best irrelevant)
  - Average encoding latency

Usage:
    python scripts/bench_embedding_compare.py
    python scripts/bench_embedding_compare.py --model bge-m3-onnx
    python scripts/bench_embedding_compare.py --model pplx-embed-v1-0.6B
"""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Test cases: French notarial domain
# Each case has a query, a list of documents, and indices of relevant docs.
# ---------------------------------------------------------------------------
TEST_CASES = [
    # --- 1. Procuration notariee ---
    {
        "id": "FR-NOTA-1",
        "topic": "Procuration notariee",
        "query": "procuration notariee pour vente immobiliere",
        "documents": [
            # Relevant
            "La procuration notariee est un acte authentique par lequel une personne donne pouvoir a une autre d'agir en son nom pour la vente d'un bien immobilier.",
            "Le mandataire designe dans la procuration peut signer l'acte de vente devant le notaire au nom et pour le compte du mandant.",
            "La procuration doit etre etablie par acte notarie lorsqu'elle est destinee a la vente d'un bien immobilier, conformement a l'article 1988 du Code civil.",
            # Distractors
            "Le bail commercial est un contrat de location portant sur un local utilise pour l'exploitation d'un fonds de commerce.",
            "Le diagnostic de performance energetique (DPE) est obligatoire pour toute mise en vente ou location d'un bien immobilier.",
            "L'assurance decennale couvre les dommages compromettant la solidite de l'ouvrage pendant dix ans apres la reception des travaux.",
            "Le cadastre est un registre public qui recense les proprietes foncieres d'une commune et sert de base au calcul de la taxe fonciere.",
        ],
        "relevant_indices": [0, 1, 2],
    },
    # --- 2. Succession et partage ---
    {
        "id": "FR-NOTA-2",
        "topic": "Succession et partage",
        "query": "acte de succession et partage entre heritiers",
        "documents": [
            # Relevant
            "L'acte de partage successoral est dresse par le notaire pour repartir les biens du defunt entre les heritiers conformement a la devolution legale ou testamentaire.",
            "En cas de desaccord entre heritiers, le partage judiciaire peut etre ordonne par le tribunal, le notaire etant alors designe comme expert pour evaluer les biens.",
            "L'attestation de propriete immobiliere est un acte notarie obligatoire pour transferer la propriete d'un bien immobilier aux heritiers.",
            # Distractors
            "Le pret a taux zero (PTZ) est un dispositif d'aide a l'accession a la propriete reserve aux primo-accedants.",
            "Le syndic de copropriete est charge de la gestion courante de l'immeuble et de l'execution des decisions de l'assemblee generale.",
            "La taxe d'amenagement est due lors de la construction ou de l'agrandissement d'un batiment.",
            "Le droit de preemption urbain permet a la commune d'acquerir en priorite un bien immobilier mis en vente.",
        ],
        "relevant_indices": [0, 1, 2],
    },
    # --- 3. Hypotheque conventionnelle ---
    {
        "id": "FR-NOTA-3",
        "topic": "Hypotheque conventionnelle",
        "query": "hypotheque conventionnelle sur bien immobilier",
        "documents": [
            # Relevant
            "L'hypotheque conventionnelle est une surete reelle immobiliere consentie par le debiteur au profit du creancier, inscrite au service de la publicite fonciere.",
            "En cas de defaillance de l'emprunteur, le creancier hypothecaire peut faire saisir et vendre le bien immobilier pour se faire rembourser par priorite.",
            "La mainlevee d'hypotheque est un acte notarie qui constate la radiation de l'inscription hypothecaire apres remboursement integral du pret.",
            # Distractors
            "Le contrat de mariage permet aux epoux de choisir un regime matrimonial different du regime legal de la communaute reduite aux acquets.",
            "La servitude de passage est un droit reel qui permet au proprietaire d'un fonds enclave d'acceder a la voie publique en traversant le fonds voisin.",
            "Le permis de construire est une autorisation administrative prealable a la realisation de certains travaux de construction.",
            "Les frais d'agence immobiliere representent generalement entre 3% et 8% du prix de vente du bien.",
        ],
        "relevant_indices": [0, 1, 2],
    },
    # --- 4. Constitution de SCI familiale ---
    {
        "id": "FR-NOTA-4",
        "topic": "Constitution de SCI",
        "query": "constitution de SCI familiale pour gestion de patrimoine immobilier",
        "documents": [
            # Relevant
            "La Societe Civile Immobiliere (SCI) familiale permet a des membres d'une meme famille de detenir et gerer ensemble un patrimoine immobilier.",
            "Les statuts de la SCI doivent etre rediges par acte notarie lorsqu'un bien immobilier est apporte au capital social.",
            "La SCI facilite la transmission du patrimoine immobilier par donation progressive de parts sociales, beneficiant des abattements fiscaux renouveles tous les 15 ans.",
            # Distractors
            "L'etat des lieux est un document etabli contradictoirement entre le bailleur et le locataire a l'entree et a la sortie du logement.",
            "Le PACS (Pacte Civil de Solidarite) est un contrat conclu entre deux personnes majeures pour organiser leur vie commune.",
            "La plus-value immobiliere est l'impot du sur le gain realise lors de la revente d'un bien immobilier autre que la residence principale.",
            "Le certificat d'urbanisme renseigne sur les regles d'urbanisme applicables a un terrain donne.",
        ],
        "relevant_indices": [0, 1, 2],
    },
    # --- 5. Donation entre epoux ---
    {
        "id": "FR-NOTA-5",
        "topic": "Donation entre epoux",
        "query": "donation entre epoux au dernier vivant",
        "documents": [
            # Relevant
            "La donation au dernier vivant, aussi appelee donation entre epoux, permet au conjoint survivant de beneficier d'une part plus importante de la succession.",
            "Le conjoint survivant peut opter pour l'usufruit de la totalite des biens ou un quart en pleine propriete et trois quarts en usufruit, selon les dispositions de la donation.",
            "La donation entre epoux est revocable a tout moment par le donateur, sauf si elle est consentie par contrat de mariage.",
            # Distractors
            "Le droit de retraction est le delai de dix jours dont dispose l'acquereur non professionnel apres la signature du compromis de vente pour se retracter sans penalite.",
            "La loi Carrez impose de mentionner la surface privative du lot de copropriete dans tout avant-contrat et acte de vente.",
            "Le geometre-expert est le seul professionnel habilite a fixer les limites d'un bien foncier par un bornage.",
            "L'acte de notoriete est un document etabli par le notaire pour constater la qualite d'heritier.",
        ],
        "relevant_indices": [0, 1, 2],
    },
    # --- 6. Multilingual: FR query, mixed FR/EN docs ---
    {
        "id": "MULTI-1",
        "topic": "Acte authentique (multilingual)",
        "query": "role du notaire dans la redaction de l'acte authentique de vente",
        "documents": [
            # Relevant (FR)
            "Le notaire est un officier public charge de conferer l'authenticite aux actes juridiques, notamment l'acte de vente immobiliere qu'il redige et conserve au rang de ses minutes.",
            # Relevant (EN)
            "A French notaire (notary public) plays a central role in real estate transactions by drafting the authentic deed of sale, verifying the legal status of the property, and ensuring compliance with all regulatory requirements.",
            # Relevant (FR)
            "L'acte authentique de vente est signe devant le notaire qui verifie l'identite des parties, la conformite du bien et procede au deblocage des fonds.",
            # Distractor (EN)
            "Machine learning algorithms can be classified into supervised, unsupervised, and reinforcement learning categories.",
            # Distractor (FR)
            "La garantie des vices caches protege l'acheteur lorsque le bien presente un defaut non apparent au moment de la vente.",
            # Distractor (EN)
            "The European Central Bank sets monetary policy for the eurozone countries.",
            # Distractor (FR)
            "Le plan local d'urbanisme (PLU) definit les regles d'occupation des sols a l'echelle communale.",
        ],
        "relevant_indices": [0, 1, 2],
    },
]


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_bge_m3_onnx() -> object:
    """
    Load bge-m3-onnx-int8 via ONNX Runtime, mimicking OnnxEmbeddingModel
    from model_manager.py.
    """
    import onnxruntime as ort
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download

    hub_id = "gpahal/bge-m3-onnx-int8"
    local_path = Path("models/bge-m3-onnx-int8")

    # Resolve model directory
    if local_path.exists():
        model_dir = str(local_path)
        print(f"  bge-m3-onnx: loading from local path {model_dir}")
    else:
        print(f"  bge-m3-onnx: downloading from {hub_id} ...")
        model_dir = snapshot_download(repo_id=hub_id, token=False)
        print(f"  bge-m3-onnx: downloaded to {model_dir}")

    # Find ONNX file
    model_file = None
    for candidate in ["model_quantized.onnx", "model.onnx"]:
        p = Path(model_dir) / candidate
        if p.exists():
            model_file = str(p)
            break
    if model_file is None:
        raise FileNotFoundError(f"No ONNX model file found in {model_dir}")

    # Session options (same as model_manager.py)
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_cpu_mem_arena = False
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = int(os.getenv("ORT_NUM_THREADS", "4"))

    session = ort.InferenceSession(model_file, providers=["CPUExecutionProvider"], sess_options=opts)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Detect output format
    output_names = [o.name for o in session.get_outputs()]
    if "dense_vecs" in output_names:
        output_name = "dense_vecs"
        needs_pooling = False
    elif "last_hidden_state" in output_names:
        output_name = "last_hidden_state"
        needs_pooling = True
    else:
        output_name = output_names[0]
        needs_pooling = len(session.get_outputs()[0].shape) == 3

    print(f"  bge-m3-onnx: output={output_name}, pooling={needs_pooling}")

    class _BgeM3OnnxEncoder:
        """Thin wrapper matching the encode() interface."""

        name = "bge-m3-onnx"
        instruction_prefix = "Represent this sentence: "

        def encode(self, texts: List[str]) -> np.ndarray:
            inputs = tokenizer(
                texts, padding=True, truncation=True, max_length=8192, return_tensors="np",
            )
            outputs = session.run(
                [output_name],
                {
                    "input_ids": inputs["input_ids"].astype(np.int64),
                    "attention_mask": inputs["attention_mask"].astype(np.int64),
                },
            )
            embeddings = outputs[0]
            if needs_pooling:
                mask = inputs["attention_mask"].astype(np.float32)
                mask_exp = np.expand_dims(mask, axis=-1)
                embeddings = np.sum(embeddings * mask_exp, axis=1) / np.clip(
                    np.sum(mask_exp, axis=1), 1e-9, None
                )
            return embeddings

    return _BgeM3OnnxEncoder()


def load_pplx_embed() -> object:
    """
    Load pplx-embed-v1-0.6B via SentenceTransformers.
    """
    from sentence_transformers import SentenceTransformer

    hub_id = "perplexity-ai/pplx-embed-v1-0.6B"
    print(f"  pplx-embed: loading from {hub_id} (trust_remote_code=True) ...")
    st_model = SentenceTransformer(hub_id, trust_remote_code=True)
    print(f"  pplx-embed: loaded (dim={st_model.get_sentence_embedding_dimension()})")

    class _PplxEmbedEncoder:
        """Thin wrapper matching the encode() interface."""

        name = "pplx-embed-v1-0.6B"
        instruction_prefix = ""  # No instruction prefix needed

        def encode(self, texts: List[str]) -> np.ndarray:
            embs = st_model.encode(texts, show_progress_bar=False)
            return np.array(embs)

    return _PplxEmbedEncoder()


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query vector(s) a and document vectors b.

    Args:
        a: shape (D,) or (1, D) -- query embedding
        b: shape (N, D) -- document embeddings

    Returns:
        shape (N,) cosine similarities
    """
    if a.ndim == 1:
        a = a[np.newaxis, :]
    # Normalise
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-9, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-9, None)
    return (a_norm @ b_norm.T).flatten()


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_case(
    scores: np.ndarray,
    relevant_indices: List[int],
    k: int = 3,
) -> Dict:
    """Compute metrics for a single test case."""
    ranked_indices = np.argsort(-scores).tolist()
    top_k = ranked_indices[:k]

    expected_set = set(relevant_indices[:k])
    hits = len(set(top_k) & expected_set)
    precision_at_k = hits / min(k, len(expected_set))
    top1_correct = ranked_indices[0] in set(relevant_indices)

    # Score distributions
    rel_scores = scores[relevant_indices].tolist()
    irrel_indices = [i for i in range(len(scores)) if i not in set(relevant_indices)]
    irrel_scores = scores[irrel_indices].tolist()

    best_relevant = max(rel_scores)
    best_irrelevant = max(irrel_scores) if irrel_scores else 0.0
    spread = best_relevant - best_irrelevant

    return {
        "precision_at_k": precision_at_k,
        "top1_correct": top1_correct,
        "hits": hits,
        "spread": spread,
        "avg_relevant_score": statistics.mean(rel_scores),
        "avg_irrelevant_score": statistics.mean(irrel_scores) if irrel_scores else 0.0,
        "best_relevant": best_relevant,
        "best_irrelevant": best_irrelevant,
        "ranked_top_k": top_k,
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(model_names: List[str]):
    """Run the embedding comparison benchmark."""

    # Load requested models
    models = {}
    print("\nLoading models ...")
    for name in model_names:
        t0 = time.perf_counter()
        if name == "bge-m3-onnx":
            models[name] = load_bge_m3_onnx()
        elif name == "pplx-embed-v1-0.6B":
            models[name] = load_pplx_embed()
        else:
            print(f"  Unknown model: {name}, skipping.")
            continue
        load_time = time.perf_counter() - t0
        print(f"  {name}: loaded in {load_time:.1f}s\n")

    if not models:
        print("No models loaded. Exiting.")
        sys.exit(1)

    # Header
    print(f"\n{'=' * 100}")
    print(f"  EMBEDDING BENCHMARK: {' vs '.join(models.keys())}")
    print(f"  Test cases: {len(TEST_CASES)} (French notarial domain + 1 multilingual)")
    print(f"{'=' * 100}")

    all_results: Dict[str, Dict] = {}

    for model_name, encoder in models.items():
        print(f"\n{'─' * 100}")
        print(f"  Model: {model_name}")
        print(f"{'─' * 100}")

        metrics_agg = {
            "latencies_ms": [],
            "precisions": [],
            "top1s": [],
            "spreads": [],
            "avg_rel_scores": [],
            "avg_irrel_scores": [],
            "details": [],
        }

        # Warmup pass (first encode can be slow due to JIT / graph optimisation)
        print("  Warmup ...", end=" ", flush=True)
        t0 = time.perf_counter()
        warmup_text = ["warmup sentence for benchmark"]
        if encoder.instruction_prefix:
            warmup_text = [encoder.instruction_prefix + warmup_text[0]]
        encoder.encode(warmup_text)
        print(f"{(time.perf_counter() - t0) * 1000:.0f}ms")

        for tc in TEST_CASES:
            query = tc["query"]
            documents = tc["documents"]
            relevant = tc["relevant_indices"]

            # Prepend instruction prefix for query if model requires it
            query_text = encoder.instruction_prefix + query if encoder.instruction_prefix else query

            # Encode and time it
            t0 = time.perf_counter()
            query_emb = encoder.encode([query_text])
            doc_embs = encoder.encode(documents)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            # Compute cosine similarities
            scores = cosine_similarity(query_emb[0], doc_embs)

            # Evaluate
            ev = evaluate_case(scores, relevant)

            metrics_agg["latencies_ms"].append(elapsed_ms)
            metrics_agg["precisions"].append(ev["precision_at_k"])
            metrics_agg["top1s"].append(1 if ev["top1_correct"] else 0)
            metrics_agg["spreads"].append(ev["spread"])
            metrics_agg["avg_rel_scores"].append(ev["avg_relevant_score"])
            metrics_agg["avg_irrel_scores"].append(ev["avg_irrelevant_score"])

            top1_mark = "OK" if ev["top1_correct"] else "MISS"
            print(
                f"  [{tc['id']:>10}] "
                f"P@3={ev['precision_at_k']:.2f}  "
                f"Top1={top1_mark:<4}  "
                f"Spread={ev['spread']:.3f}  "
                f"RelAvg={ev['avg_relevant_score']:.3f}  "
                f"IrrelAvg={ev['avg_irrelevant_score']:.3f}  "
                f"Latency={elapsed_ms:.0f}ms  "
                f"Rank: {ev['ranked_top_k']}"
            )

            metrics_agg["details"].append({
                "id": tc["id"],
                "topic": tc["topic"],
                "latency_ms": elapsed_ms,
                **ev,
            })

        if metrics_agg["latencies_ms"]:
            all_results[model_name] = {
                "avg_latency_ms": statistics.mean(metrics_agg["latencies_ms"]),
                "p50_latency_ms": statistics.median(metrics_agg["latencies_ms"]),
                "avg_precision": statistics.mean(metrics_agg["precisions"]),
                "top1_accuracy": statistics.mean(metrics_agg["top1s"]),
                "avg_spread": statistics.mean(metrics_agg["spreads"]),
                "avg_relevant_score": statistics.mean(metrics_agg["avg_rel_scores"]),
                "avg_irrelevant_score": statistics.mean(metrics_agg["avg_irrel_scores"]),
                "details": metrics_agg["details"],
            }

    # ── Summary table ──
    print(f"\n{'=' * 100}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 100}\n")

    header = (
        f"{'Model':<25} "
        f"{'P@3':>6} "
        f"{'Top-1':>6} "
        f"{'Spread':>8} "
        f"{'Rel-Avg':>8} "
        f"{'Irrel-Avg':>10} "
        f"{'Avg ms':>8} "
        f"{'P50 ms':>8}"
    )
    print(header)
    print("─" * len(header))

    for model_name, res in all_results.items():
        print(
            f"{model_name:<25} "
            f"{res['avg_precision']:>6.2f} "
            f"{res['top1_accuracy']:>6.2f} "
            f"{res['avg_spread']:>8.3f} "
            f"{res['avg_relevant_score']:>8.3f} "
            f"{res['avg_irrelevant_score']:>10.3f} "
            f"{res['avg_latency_ms']:>8.0f} "
            f"{res['p50_latency_ms']:>8.0f}"
        )

    # ── Per-case comparison table (if 2 models) ──
    if len(all_results) == 2:
        names = list(all_results.keys())
        print(f"\n{'=' * 100}")
        print(f"  PER-CASE COMPARISON: {names[0]} vs {names[1]}")
        print(f"{'=' * 100}\n")

        row_hdr = f"{'Case':>12}  {'P@3-A':>6} {'P@3-B':>6}  {'Top1-A':>6} {'Top1-B':>6}  {'Spread-A':>8} {'Spread-B':>8}  {'ms-A':>6} {'ms-B':>6}"
        print(row_hdr)
        print("─" * len(row_hdr))

        details_a = all_results[names[0]]["details"]
        details_b = all_results[names[1]]["details"]

        for da, db in zip(details_a, details_b):
            t1a = "OK" if da["top1_correct"] else "MISS"
            t1b = "OK" if db["top1_correct"] else "MISS"
            print(
                f"{da['id']:>12}  "
                f"{da['precision_at_k']:>6.2f} {db['precision_at_k']:>6.2f}  "
                f"{t1a:>6} {t1b:>6}  "
                f"{da['spread']:>8.3f} {db['spread']:>8.3f}  "
                f"{da['latency_ms']:>6.0f} {db['latency_ms']:>6.0f}"
            )

    # ── Winner ──
    valid = {k: v for k, v in all_results.items() if v}
    if len(valid) >= 2:
        best_quality = max(valid.items(), key=lambda x: x[1]["avg_precision"])
        best_spread = max(valid.items(), key=lambda x: x[1]["avg_spread"])
        fastest = min(valid.items(), key=lambda x: x[1]["avg_latency_ms"])
        print(f"\nBest quality (P@3):  {best_quality[0]} (P@3={best_quality[1]['avg_precision']:.2f})")
        print(f"Best spread:         {best_spread[0]} (spread={best_spread[1]['avg_spread']:.3f})")
        print(f"Fastest:             {fastest[0]} ({fastest[1]['avg_latency_ms']:.0f}ms avg)")

    # ── Cosine similarity distribution ──
    print(f"\n{'=' * 100}")
    print(f"  COSINE SIMILARITY DISTRIBUTIONS")
    print(f"{'=' * 100}\n")

    dist_hdr = f"{'Model':<25} {'Relevant':>12} {'Irrelevant':>12} {'Gap':>8}"
    print(dist_hdr)
    print("─" * len(dist_hdr))

    for model_name, res in all_results.items():
        print(
            f"{model_name:<25} "
            f"{res['avg_relevant_score']:>12.4f} "
            f"{res['avg_irrelevant_score']:>12.4f} "
            f"{res['avg_relevant_score'] - res['avg_irrelevant_score']:>8.4f}"
        )

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark embedding models on French notarial domain data"
    )
    parser.add_argument(
        "--model",
        choices=["bge-m3-onnx", "pplx-embed-v1-0.6B"],
        help="Test a single model only (default: both)",
    )
    args = parser.parse_args()

    if args.model:
        model_names = [args.model]
    else:
        model_names = ["bge-m3-onnx", "pplx-embed-v1-0.6B"]

    run_benchmark(model_names)


if __name__ == "__main__":
    main()
