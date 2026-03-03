"""
Benchmark: pplx-embed-v1 Matryoshka truncation at different dimensions.

Tests whether truncating pplx-embed-v1 INT8 embeddings from 1024D to
smaller dimensions (768, 512, 256, 128) retains quality above bge-m3 1024D.

Uses the same French notarial test cases as bench_embedding_compare.py.
Loads the ONNX Q4 model (same as production) and truncates post-inference.

Usage:
    python scripts/bench_matryoshka_truncation.py
"""

import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# Test cases (same as bench_embedding_compare.py)
# ---------------------------------------------------------------------------
TEST_CASES = [
    {
        "id": "FR-NOTA-1",
        "topic": "Procuration notariee",
        "query": "procuration notariee pour vente immobiliere",
        "documents": [
            "La procuration notariee est un acte authentique par lequel une personne donne pouvoir a une autre d'agir en son nom pour la vente d'un bien immobilier.",
            "Le mandataire designe dans la procuration peut signer l'acte de vente devant le notaire au nom et pour le compte du mandant.",
            "La procuration doit etre etablie par acte notarie lorsqu'elle est destinee a la vente d'un bien immobilier, conformement a l'article 1988 du Code civil.",
            "Le bail commercial est un contrat de location portant sur un local utilise pour l'exploitation d'un fonds de commerce.",
            "Le diagnostic de performance energetique (DPE) est obligatoire pour toute mise en vente ou location d'un bien immobilier.",
            "L'assurance decennale couvre les dommages compromettant la solidite de l'ouvrage pendant dix ans apres la reception des travaux.",
            "Le cadastre est un registre public qui recense les proprietes foncieres d'une commune et sert de base au calcul de la taxe fonciere.",
        ],
        "relevant_indices": [0, 1, 2],
    },
    {
        "id": "FR-NOTA-2",
        "topic": "Succession et partage",
        "query": "acte de succession et partage entre heritiers",
        "documents": [
            "L'acte de partage successoral est dresse par le notaire pour repartir les biens du defunt entre les heritiers conformement a la devolution legale ou testamentaire.",
            "En cas de desaccord entre heritiers, le partage judiciaire peut etre ordonne par le tribunal, le notaire etant alors designe comme expert pour evaluer les biens.",
            "L'attestation de propriete immobiliere est un acte notarie obligatoire pour transferer la propriete d'un bien immobilier aux heritiers.",
            "Le pret a taux zero (PTZ) est un dispositif d'aide a l'accession a la propriete reserve aux primo-accedants.",
            "Le syndic de copropriete est charge de la gestion courante de l'immeuble et de l'execution des decisions de l'assemblee generale.",
            "La taxe d'amenagement est due lors de la construction ou de l'agrandissement d'un batiment.",
            "Le droit de preemption urbain permet a la commune d'acquerir en priorite un bien immobilier mis en vente.",
        ],
        "relevant_indices": [0, 1, 2],
    },
    {
        "id": "FR-NOTA-3",
        "topic": "Hypotheque conventionnelle",
        "query": "hypotheque conventionnelle sur bien immobilier",
        "documents": [
            "L'hypotheque conventionnelle est une surete reelle immobiliere consentie par le debiteur au profit du creancier, inscrite au service de la publicite fonciere.",
            "En cas de defaillance de l'emprunteur, le creancier hypothecaire peut faire saisir et vendre le bien immobilier pour se faire rembourser par priorite.",
            "La mainlevee d'hypotheque est un acte notarie qui constate la radiation de l'inscription hypothecaire apres remboursement integral du pret.",
            "Le contrat de mariage permet aux epoux de choisir un regime matrimonial different du regime legal de la communaute reduite aux acquets.",
            "La servitude de passage est un droit reel qui permet au proprietaire d'un fonds enclave d'acceder a la voie publique en traversant le fonds voisin.",
            "Le permis de construire est une autorisation administrative prealable a la realisation de certains travaux de construction.",
            "Les frais d'agence immobiliere representent generalement entre 3% et 8% du prix de vente du bien.",
        ],
        "relevant_indices": [0, 1, 2],
    },
    {
        "id": "FR-NOTA-4",
        "topic": "Constitution de SCI",
        "query": "constitution de SCI familiale pour gestion de patrimoine immobilier",
        "documents": [
            "La Societe Civile Immobiliere (SCI) familiale permet a des membres d'une meme famille de detenir et gerer ensemble un patrimoine immobilier.",
            "Les statuts de la SCI doivent etre rediges par acte notarie lorsqu'un bien immobilier est apporte au capital social.",
            "La SCI facilite la transmission du patrimoine immobilier par donation progressive de parts sociales, beneficiant des abattements fiscaux renouveles tous les 15 ans.",
            "L'etat des lieux est un document etabli contradictoirement entre le bailleur et le locataire a l'entree et a la sortie du logement.",
            "Le PACS (Pacte Civil de Solidarite) est un contrat conclu entre deux personnes majeures pour organiser leur vie commune.",
            "La plus-value immobiliere est l'impot du sur le gain realise lors de la revente d'un bien immobilier autre que la residence principale.",
            "Le certificat d'urbanisme renseigne sur les regles d'urbanisme applicables a un terrain donne.",
        ],
        "relevant_indices": [0, 1, 2],
    },
    {
        "id": "FR-NOTA-5",
        "topic": "Donation entre epoux",
        "query": "donation entre epoux au dernier vivant",
        "documents": [
            "La donation au dernier vivant, aussi appelee donation entre epoux, permet au conjoint survivant de beneficier d'une part plus importante de la succession.",
            "Le conjoint survivant peut opter pour l'usufruit de la totalite des biens ou un quart en pleine propriete et trois quarts en usufruit, selon les dispositions de la donation.",
            "La donation entre epoux est revocable a tout moment par le donateur, sauf si elle est consentie par contrat de mariage.",
            "Le droit de retraction est le delai de dix jours dont dispose l'acquereur non professionnel apres la signature du compromis de vente pour se retracter sans penalite.",
            "La loi Carrez impose de mentionner la surface privative du lot de copropriete dans tout avant-contrat et acte de vente.",
            "Le geometre-expert est le seul professionnel habilite a fixer les limites d'un bien foncier par un bornage.",
            "L'acte de notoriete est un document etabli par le notaire pour constater la qualite d'heritier.",
        ],
        "relevant_indices": [0, 1, 2],
    },
    {
        "id": "MULTI-1",
        "topic": "Acte authentique (multilingual)",
        "query": "role du notaire dans la redaction de l'acte authentique de vente",
        "documents": [
            "Le notaire est un officier public charge de conferer l'authenticite aux actes juridiques, notamment l'acte de vente immobiliere qu'il redige et conserve au rang de ses minutes.",
            "A French notaire (notary public) plays a central role in real estate transactions by drafting the authentic deed of sale, verifying the legal status of the property, and ensuring compliance with all regulatory requirements.",
            "L'acte authentique de vente est signe devant le notaire qui verifie l'identite des parties, la conformite du bien et procede au deblocage des fonds.",
            "Machine learning algorithms can be classified into supervised, unsupervised, and reinforcement learning categories.",
            "La garantie des vices caches protege l'acheteur lorsque le bien presente un defaut non apparent au moment de la vente.",
            "The European Central Bank sets monetary policy for the eurozone countries.",
            "Le plan local d'urbanisme (PLU) definit les regles d'occupation des sols a l'echelle communale.",
        ],
        "relevant_indices": [0, 1, 2],
    },
]

TRUNCATION_DIMS = [1024, 768, 512, 256, 128]

# Reference baseline from previous benchmark
BGE_M3_BASELINE = {"P@3": 0.94, "spread": 0.066}


# ---------------------------------------------------------------------------
# Model loader — ONNX Q4 (production path)
# ---------------------------------------------------------------------------

def load_pplx_onnx():
    """Load pplx-embed-v1-0.6B ONNX Q4 model."""
    import onnxruntime as ort
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download

    hub_id = "perplexity-ai/pplx-embed-v1-0.6B"
    local_path = Path("models/pplx-embed-v1-q4")

    # Resolve model directory
    if local_path.exists():
        model_dir = local_path
        print(f"  pplx-embed ONNX: loading from local {model_dir}")
    else:
        print(f"  pplx-embed ONNX: downloading from {hub_id} ...")
        cache = snapshot_download(repo_id=hub_id, token=False)
        model_dir = Path(cache)
        # Check for onnx/ subfolder
        onnx_sub = model_dir / "onnx"
        if onnx_sub.exists():
            model_dir = onnx_sub
        print(f"  pplx-embed ONNX: downloaded to {model_dir}")

    # Find ONNX file
    model_file = None
    for candidate in ["model_q4.onnx", "model_quantized.onnx", "model.onnx"]:
        p = model_dir / candidate
        if p.exists():
            model_file = str(p)
            break
    if model_file is None:
        raise FileNotFoundError(f"No ONNX model file found in {model_dir}")

    # Session options
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_cpu_mem_arena = False
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = int(os.getenv("ORT_NUM_THREADS", "4"))

    print(f"  Loading ONNX session from {model_file} ...")
    session = ort.InferenceSession(
        model_file, providers=["CPUExecutionProvider"], sess_options=opts
    )

    # Tokenizer from parent dir (tokenizer files are at repo root, not in onnx/)
    tokenizer_dir = str(model_dir.parent) if model_dir.name == "onnx" else str(model_dir)
    # If using HF cache, tokenizer is at repo root
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)

    # Detect output
    output_names = [o.name for o in session.get_outputs()]
    print(f"  ONNX outputs: {output_names}")

    if "pooler_output_int8" in output_names:
        output_name = "pooler_output_int8"
        needs_pooling = False
    elif "pooler_output" in output_names:
        output_name = "pooler_output"
        needs_pooling = False
    elif "dense_vecs" in output_names:
        output_name = "dense_vecs"
        needs_pooling = False
    elif "last_hidden_state" in output_names:
        output_name = "last_hidden_state"
        needs_pooling = True
    else:
        output_name = output_names[0]
        needs_pooling = len(session.get_outputs()[0].shape) == 3

    print(f"  Using output: {output_name}, pooling: {needs_pooling}")

    # Check input names for token_type_ids
    input_names = [i.name for i in session.get_inputs()]
    has_token_type_ids = "token_type_ids" in input_names
    print(f"  Input names: {input_names}")

    def encode(texts: List[str]) -> np.ndarray:
        inputs = tokenizer(
            texts, padding=True, truncation=True, max_length=8192, return_tensors="np",
        )
        feed = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        if has_token_type_ids and "token_type_ids" in inputs:
            feed["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)

        outputs = session.run([output_name], feed)
        embeddings = outputs[0]

        if needs_pooling:
            mask = inputs["attention_mask"].astype(np.float32)
            mask_exp = np.expand_dims(mask, axis=-1)
            embeddings = np.sum(embeddings * mask_exp, axis=1) / np.clip(
                np.sum(mask_exp, axis=1), 1e-9, None
            )
        return embeddings.astype(np.float32)

    return encode


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.ndim == 1:
        a = a[np.newaxis, :]
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-9, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-9, None)
    return (a_norm @ b_norm.T).flatten()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_case(scores: np.ndarray, relevant_indices: List[int], k: int = 3) -> Dict:
    ranked_indices = np.argsort(-scores).tolist()
    top_k = ranked_indices[:k]
    expected_set = set(relevant_indices[:k])
    hits = len(set(top_k) & expected_set)
    precision_at_k = hits / min(k, len(expected_set))
    top1_correct = ranked_indices[0] in set(relevant_indices)

    rel_scores = scores[relevant_indices].tolist()
    irrel_indices = [i for i in range(len(scores)) if i not in set(relevant_indices)]
    irrel_scores = scores[irrel_indices].tolist()

    best_relevant = max(rel_scores)
    best_irrelevant = max(irrel_scores) if irrel_scores else 0.0
    spread = best_relevant - best_irrelevant

    return {
        "precision_at_k": precision_at_k,
        "top1_correct": top1_correct,
        "spread": spread,
        "avg_relevant_score": statistics.mean(rel_scores),
        "avg_irrelevant_score": statistics.mean(irrel_scores) if irrel_scores else 0.0,
        "ranked_top_k": top_k,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 90)
    print("  MATRYOSHKA TRUNCATION BENCHMARK: pplx-embed-v1 Q4 ONNX")
    print("  Dimensions: " + ", ".join(str(d) for d in TRUNCATION_DIMS))
    print("  Baseline: bge-m3-onnx 1024D (P@3=0.94, spread=0.066)")
    print("=" * 90)

    print("\nLoading pplx-embed-v1 ONNX Q4 ...")
    t0 = time.perf_counter()
    encode = load_pplx_onnx()
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    # Warmup
    print("\n  Warmup ...", end=" ", flush=True)
    t0 = time.perf_counter()
    encode(["warmup sentence"])
    print(f"{(time.perf_counter() - t0) * 1000:.0f}ms")

    # Pre-encode all texts at full 1024D (encode once, truncate many times)
    print("\n  Encoding all texts at 1024D ...")
    all_query_embs = []
    all_doc_embs = []
    latencies = []

    for tc in TEST_CASES:
        t0 = time.perf_counter()
        q_emb = encode([tc["query"]])
        d_embs = encode(tc["documents"])
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)
        all_query_embs.append(q_emb)
        all_doc_embs.append(d_embs)

    full_dim = all_query_embs[0].shape[1]
    print(f"  Full dimension: {full_dim}")
    print(f"  Avg encoding latency: {statistics.mean(latencies):.0f}ms")

    # Evaluate at each truncation dimension
    results_by_dim = {}

    for dim in TRUNCATION_DIMS:
        if dim > full_dim:
            continue

        precisions = []
        top1s = []
        spreads = []
        rel_scores_all = []
        irrel_scores_all = []
        case_details = []

        for i, tc in enumerate(TEST_CASES):
            q_trunc = all_query_embs[i][:, :dim]
            d_trunc = all_doc_embs[i][:, :dim]

            scores = cosine_similarity(q_trunc[0], d_trunc)
            ev = evaluate_case(scores, tc["relevant_indices"])

            precisions.append(ev["precision_at_k"])
            top1s.append(1 if ev["top1_correct"] else 0)
            spreads.append(ev["spread"])
            rel_scores_all.append(ev["avg_relevant_score"])
            irrel_scores_all.append(ev["avg_irrelevant_score"])
            case_details.append((tc["id"], ev))

        results_by_dim[dim] = {
            "avg_p3": statistics.mean(precisions),
            "top1_acc": statistics.mean(top1s),
            "avg_spread": statistics.mean(spreads),
            "avg_rel": statistics.mean(rel_scores_all),
            "avg_irrel": statistics.mean(irrel_scores_all),
            "details": case_details,
        }

    # ── Summary table ──
    print(f"\n{'=' * 90}")
    print(f"  RESULTS: pplx-embed-v1 Q4 truncated at each dimension")
    print(f"  Reference: bge-m3-onnx 1024D  P@3={BGE_M3_BASELINE['P@3']:.2f}  spread={BGE_M3_BASELINE['spread']:.3f}")
    print(f"{'=' * 90}\n")

    hdr = f"{'Dim':>6}  {'P@3':>6}  {'Top-1':>6}  {'Spread':>8}  {'Rel-Avg':>8}  {'Irrel-Avg':>10}  {'vs bge-m3':>10}  {'Storage':>10}"
    print(hdr)
    print("-" * len(hdr))

    for dim in TRUNCATION_DIMS:
        if dim not in results_by_dim:
            continue
        r = results_by_dim[dim]
        vs_bge = "BETTER" if r["avg_p3"] >= BGE_M3_BASELINE["P@3"] else "WORSE"
        if r["avg_p3"] == BGE_M3_BASELINE["P@3"]:
            vs_bge = "EQUAL"
        storage_ratio = f"{dim/1024:.0%} ({dim*1}B/vec)"
        print(
            f"{dim:>6}  "
            f"{r['avg_p3']:>6.2f}  "
            f"{r['top1_acc']:>6.2f}  "
            f"{r['avg_spread']:>8.4f}  "
            f"{r['avg_rel']:>8.4f}  "
            f"{r['avg_irrel']:>10.4f}  "
            f"{vs_bge:>10}  "
            f"{storage_ratio:>10}"
        )

    # ── Per-case breakdown ──
    print(f"\n{'=' * 90}")
    print(f"  PER-CASE BREAKDOWN")
    print(f"{'=' * 90}\n")

    for dim in TRUNCATION_DIMS:
        if dim not in results_by_dim:
            continue
        print(f"  --- {dim}D ---")
        for case_id, ev in results_by_dim[dim]["details"]:
            t1 = "OK" if ev["top1_correct"] else "MISS"
            print(
                f"    [{case_id:>10}] "
                f"P@3={ev['precision_at_k']:.2f}  "
                f"Top1={t1:<4}  "
                f"Spread={ev['spread']:.4f}  "
                f"Rank={ev['ranked_top_k']}"
            )
        print()

    # ── Recommendation ──
    print(f"{'=' * 90}")
    print(f"  RECOMMENDATION")
    print(f"{'=' * 90}\n")

    # Find smallest dim that still beats bge-m3
    best_small = None
    for dim in sorted(TRUNCATION_DIMS):
        if dim in results_by_dim and results_by_dim[dim]["avg_p3"] >= BGE_M3_BASELINE["P@3"]:
            best_small = dim

    if best_small and best_small < 1024:
        r = results_by_dim[best_small]
        savings = (1 - best_small / 1024) * 100
        print(f"  pplx-embed-v1 at {best_small}D still beats bge-m3 1024D!")
        print(f"  P@3={r['avg_p3']:.2f} (vs {BGE_M3_BASELINE['P@3']:.2f}), spread={r['avg_spread']:.4f} (vs {BGE_M3_BASELINE['spread']:.3f})")
        print(f"  Storage savings: {savings:.0f}% less per vector")
        print(f"  Cosine similarity speedup: ~{1024/best_small:.1f}x faster dot product")
    elif best_small == 1024:
        # Find the threshold
        threshold = None
        for dim in sorted(TRUNCATION_DIMS, reverse=True):
            if dim in results_by_dim and results_by_dim[dim]["avg_p3"] >= BGE_M3_BASELINE["P@3"]:
                threshold = dim
        if threshold and threshold < 1024:
            print(f"  Minimum dimension to match bge-m3: {threshold}D")
        else:
            print(f"  Only 1024D matches bge-m3. Truncation not recommended for this task.")
    else:
        print(f"  No truncated dimension matches bge-m3 quality. Use full 1024D.")

    print()


if __name__ == "__main__":
    main()
