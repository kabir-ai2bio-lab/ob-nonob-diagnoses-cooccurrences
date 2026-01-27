# GCN link prediction on pregnancy diagnosis co-occurrence graph
#
# New node features:
#   - age_mean: mean age of patients when the diagnosis appears in pregnancy visits
#   - age_std:  standard deviation of that age

import os
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for GPU nodes
import matplotlib.pyplot as plt

# Set environment variables before importing torch to avoid CUDA issues
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from config import (
    verify_data_files,
    DIAG_PATH,
    CONCEPT_PATH,
    PERSON_PATH,
    FIGURES_DIR,
    MIN_DIAG_COUNT,
    TOP_N_DIAG,
    EDGE_MIN_WEIGHT,
)


def check_cuda_compatibility():
    """Check CUDA availability and compatibility."""
    print("=" * 60)
    print("CUDA Compatibility Check")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        print("CUDA not available, will use CPU")
    print("=" * 60)

# ---------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def zscore_series(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mean = s.mean()
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        std = 1.0
    return (s - mean) / std


def it_combinations(iterable, r):
    """Simple replacement for itertools.combinations to keep dependencies local."""
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield tuple(pool[i] for i in indices)


# ---------------------------------------------------------------------
# build pregnancy graph and node features (including age)
# ---------------------------------------------------------------------

def plot_pregnancy_graph(G_preg, nodes_preg_df, comm_id_of_preg, out_name):
    """
    Draw a visualization of the full pregnancy graph.
    Nodes are colored by community and sized by betweenness.
    All nodes are labeled with their diagnosis names.
    """
    if G_preg.number_of_nodes() == 0:
        print("[Figure] Pregnancy graph is empty, skipping visualization.")
        return

    print("[Figure] Drawing full pregnancy graph...")
    pos = nx.spring_layout(G_preg, weight="weight", seed=42)

    # Node colors: community id
    node_color = [comm_id_of_preg.get(n, -1) for n in G_preg.nodes()]

    # Node sizes: betweenness_preg
    bet_map = {
        row["diagnosis"]: row["betweenness_preg"]
        for _, row in nodes_preg_df.iterrows()
    }
    max_b = max(bet_map.values()) if bet_map else 1.0
    node_size = [250 * (bet_map.get(n, 0) / max_b + 0.2) for n in G_preg.nodes()]

    plt.figure(figsize=(16, 12))
    nx.draw_networkx_edges(G_preg, pos, alpha=0.25, width=0.8)
    
    # Get unique communities and assign colors
    import matplotlib.cm as cm
    unique_communities = sorted(set(node_color))
    cmap = cm.get_cmap("tab20", len(unique_communities))
    
    # Draw nodes by community for legend
    from matplotlib.patches import Patch
    legend_elements = []
    for i, comm in enumerate(unique_communities):
        if comm == -1:
            continue  # Skip unassigned nodes
        nodes_in_comm = [n for n in G_preg.nodes() if comm_id_of_preg.get(n, -1) == comm]
        if not nodes_in_comm:
            continue
        
        # Get node sizes for this community
        comm_sizes = [250 * (bet_map.get(n, 0) / max_b + 0.2) for n in nodes_in_comm]
        comm_pos = {n: pos[n] for n in nodes_in_comm}
        
        nx.draw_networkx_nodes(
            G_preg,
            comm_pos,
            nodelist=nodes_in_comm,
            node_size=comm_sizes,
            node_color=[cmap(i)] * len(nodes_in_comm),
            alpha=0.9,
        )
        
        # Add to legend
        legend_elements.append(Patch(facecolor=cmap(i), label=f'Community {comm}'))

    # Label all nodes with their diagnosis names
    label_map = {
        n: nodes_preg_df.loc[nodes_preg_df["diagnosis"] == n, "label"].iloc[0]
        for n in G_preg.nodes()
        if n in nodes_preg_df["diagnosis"].values
    }
    nx.draw_networkx_labels(G_preg, pos, labels=label_map, font_size=6)

    plt.title("[Pregnancy] Full diagnosis co-occurrence graph (Project 2)", fontsize=14, pad=20)
    plt.axis("off")
    
    # Add legend
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.0, 1.0), 
               fontsize=10, title='Node Communities', title_fontsize=11)
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, out_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Figure] Saved pregnancy graph visualization to {out_path}")


def build_pregnancy_graph_and_features():
    """
    Rebuild the pregnancy co-occurrence graph, very close to Project 1,
    and construct node features for GNN training, including age_mean and age_std.

    Returns:
        nodes           list of diag_code strings in fixed index order
        X               np.ndarray [num_nodes, num_features]
        edge_pairs      np.ndarray [num_edges, 2] undirected (u_idx, v_idx)
        ob_mask         np.ndarray [num_nodes] bool, True if diagnosis is obstetric
    """
    verify_data_files()

    # 1) Load condition occurrences
    # Use full file so we can optionally grab condition_start_date if present
    df = pd.read_csv(DIAG_PATH, low_memory=False)

    required_cols = {"person_id", "visit_occurrence_id", "condition_concept_id"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DIAG_PATH is missing required columns: {required_cols}")

    df = df.dropna(
        subset=["person_id", "visit_occurrence_id", "condition_concept_id"]
    )
    df["person_id"] = df["person_id"].astype(int)
    df["patient_id"] = df["person_id"].astype(str)
    df["visit_id"] = df["visit_occurrence_id"].astype(str)
    df["diag_code"] = df["condition_concept_id"].astype(str)

    # Keep only relevant columns for now, but also condition_start_date if available
    date_col = None
    for cand in ["condition_start_date", "condition_start_datetime"]:
        if cand in df.columns:
            date_col = cand
            break

    base_cols = ["patient_id", "visit_id", "diag_code"]
    if date_col is not None:
        base_cols.append(date_col)
    df = df[base_cols].copy()

    # Filter out empty codes and rare diagnoses (global count)
    df = df[df["diag_code"].str.strip().ne("")]
    vc_global = df["diag_code"].value_counts()
    if MIN_DIAG_COUNT:
        df = df[df["diag_code"].isin(vc_global[vc_global >= MIN_DIAG_COUNT].index)]

    # 2) Load concept table and identify pregnancy codes
    concept = pd.read_csv(
        CONCEPT_PATH, usecols=["concept_id", "concept_name", "domain_id"]
    )
    concept_cond = concept[concept["domain_id"] == "Condition"].copy()
    concept_cond["name_low"] = concept_cond["concept_name"].str.lower()

    PREGNANCY_KEYWORDS = [
        "pregnan",
        "obstetric",
        "prenatal",
        "antepartum",
        "intrapartum",
        "postpartum",
        "puerper",
        "gestation",
        "gestational",
        "preeclampsia",
        "eclampsia",
        "hyperemesis",
        "placenta",
        "preterm",
        "premature rupture",
        "prom",
        "hellp",
        "chorioamnionitis",
        "obstetric hemorrhage",
        "labor",
        "delivery",
    ]

    preg_mask = False
    for kw in PREGNANCY_KEYWORDS:
        preg_mask = preg_mask | concept_cond["name_low"].str.contains(kw, na=False)
    pregnancy_codes = set(
        concept_cond.loc[preg_mask, "concept_id"].astype(str).tolist()
    )
    print(f"[Pregnancy] Found {len(pregnancy_codes)} obstetric concept_ids")

    # 3) Identify pregnancy visits (any visit with at least one pregnancy diagnosis)
    vis_has_preg = (
        df.assign(is_preg=df["diag_code"].isin(pregnancy_codes))
        .groupby(["patient_id", "visit_id"], sort=False)["is_preg"]
        .any()
        .reset_index()
    )
    preg_visits = set(
        map(
            tuple,
            vis_has_preg[vis_has_preg["is_preg"]][["patient_id", "visit_id"]].values,
        )
    )
    print(f"[Pregnancy] Identified {len(preg_visits)} pregnancy visits")

    df_preg = df[df[["patient_id", "visit_id"]].apply(tuple, axis=1).isin(preg_visits)]

    # 4) Attach year_of_birth from sampled_person.csv and compute age at visit
    person = pd.read_csv(PERSON_PATH, usecols=["person_id", "year_of_birth"])
    person["person_id"] = person["person_id"].astype(int)

    df_preg_age = df_preg.copy()
    # we still have patient_id as string but need person_id
    df_preg_age["person_id"] = df_preg_age["patient_id"].astype(int)

    df_preg_age = df_preg_age.merge(
        person, on="person_id", how="left"
    )  # adds year_of_birth

    # if we have a condition date, compute age at that date, else approximate
    if date_col is not None:
        df_preg_age[date_col] = pd.to_datetime(df_preg_age[date_col], errors="coerce")
        visit_year = df_preg_age[date_col].dt.year
        # fallback: if visit_year missing, use median visit year
        median_year = int(visit_year.dropna().median()) if visit_year.notna().any() else 2010
        visit_year = visit_year.fillna(median_year)
        df_preg_age["age_at_visit"] = visit_year - df_preg_age["year_of_birth"]
    else:
        # no date info, approximate with a study year (2010) minus year_of_birth
        df_preg_age["age_at_visit"] = 2010 - df_preg_age["year_of_birth"]

    # drop obvious nonsense ages (negative or over 120)
    df_preg_age.loc[
        (df_preg_age["age_at_visit"] < 0) | (df_preg_age["age_at_visit"] > 120),
        "age_at_visit",
    ] = np.nan

    # 5) More aggressive filter inside pregnancy cohort
    PREG_MIN_COUNT = max(10, MIN_DIAG_COUNT // 2)
    EDGE_MIN_WEIGHT_PREG = max(2, EDGE_MIN_WEIGHT - 1)

    vc_preg = df_preg_age["diag_code"].value_counts()
    df_preg_age = df_preg_age[
        df_preg_age["diag_code"].isin(vc_preg[vc_preg >= PREG_MIN_COUNT].index)
    ]

    # 6) Build pregnancy co-occurrence edges
    pairs_counter = Counter()
    for _, grp in df_preg_age.groupby(["patient_id", "visit_id"], sort=False):
        codes = sorted(set(grp["diag_code"]))
        for a, b in it_combinations(codes, 2):
            pairs_counter[(a, b)] += 1

    edges_preg = pd.DataFrame(
        [(a, b, w) for (a, b), w in pairs_counter.items()],
        columns=["diag_a", "diag_b", "weight"],
    )
    edges_preg = edges_preg[edges_preg["weight"] >= EDGE_MIN_WEIGHT_PREG]
    edges_preg = edges_preg.sort_values(
        ["diag_a", "diag_b"], kind="mergesort"
    ).reset_index(drop=True)
    print(
        f"[Pregnancy] Built {len(edges_preg)} edges over {df_preg_age['diag_code'].nunique()} diagnosis nodes."
    )

    # 7) Build NetworkX graph with distance attribute
    G_preg = nx.Graph()
    for a, b, w in edges_preg.itertuples(index=False):
        w = int(w)
        G_preg.add_edge(a, b, weight=w, distance=1.0 / max(1, w))

    print(
        f"[Pregnancy] Graph: {G_preg.number_of_nodes()} nodes, {G_preg.number_of_edges()} edges."
    )

    # 8) Louvain communities, betweenness, participation
    try:
        import community as community_louvain  # python-louvain

        part_p = community_louvain.best_partition(
            G_preg, weight="weight", random_state=42
        )
        comm_id_of_preg = part_p
        from collections import defaultdict as _dd2

        groups = _dd2(set)
        for n, c in part_p.items():
            groups[c].add(n)
        communities_preg = list(groups.values())
        from networkx.algorithms.community.quality import modularity as nx_mod

        mod_p = nx_mod(G_preg, communities_preg, weight="weight")
        print(
            f"[Pregnancy] Louvain: {len(communities_preg)} communities; modularity={mod_p:.3f}"
        )
    except Exception as e:
        print(
            f"[Pregnancy][WARN] Louvain failed ({e}); using greedy modularity for communities only."
        )
        mapping_p = {n: i for i, n in enumerate(sorted(G_preg.nodes()))}
        Gp_rel = nx.relabel_nodes(G_preg, mapping_p, copy=True)
        from networkx.algorithms.community import (
            greedy_modularity_communities,
            modularity,
        )

        comm_sets_p = list(
            greedy_modularity_communities(Gp_rel, weight="weight")
        )
        inv_mp = {i: n for n, i in mapping_p.items()}
        communities_preg = [{inv_mp[i] for i in s} for s in comm_sets_p]
        comm_id_of_preg = {}
        for cid, comm in enumerate(communities_preg):
            for n in comm:
                comm_id_of_preg[n] = cid
        mod_p = modularity(Gp_rel, comm_sets_p, weight="weight")
        print(
            f"[Pregnancy] Greedy: {len(communities_preg)} communities; modularity={mod_p:.3f}"
        )

    print("[Pregnancy] Computing betweenness centrality...")
    betw_p = nx.betweenness_centrality(
        G_preg, weight="distance", normalized=True
    )
    artic_p = set(nx.articulation_points(G_preg))

    def participation_coeff_local(Gx, comm_of):
        P = {}
        for u in Gx.nodes():
            k = Gx.degree(u)
            if k == 0:
                P[u] = 0.0
                continue
            counts = defaultdict(int)
            for v in Gx.neighbors(u):
                counts[comm_of.get(v, -1)] += 1
            P[u] = 1.0 - sum((cnt / k) ** 2 for cnt in counts.values())
        return P

    part_p = participation_coeff_local(G_preg, comm_id_of_preg)

    # 9) Obstetric flag
    ob_flag = {n: int(n in pregnancy_codes) for n in G_preg.nodes()}

    def cross_ob_share(Gx, ob_flag_dict, u):
        nbrs = list(Gx.neighbors(u))
        if not nbrs:
            return 0.0, False
        total = len(nbrs)
        cross = 0
        has_ob = False
        has_non = False
        for v in nbrs:
            if ob_flag_dict.get(v, 0) != ob_flag_dict.get(u, 0):
                cross += 1
            if ob_flag_dict.get(v, 0) == 1:
                has_ob = True
            else:
                has_non = True
        share = cross / total
        has_both = has_ob and has_non
        return share, has_both

    rows_p = []
    for n in G_preg.nodes():
        cross_share, has_both = cross_ob_share(G_preg, ob_flag, n)
        rows_p.append(
            {
                "diagnosis": str(n),
                "degree_preg": G_preg.degree(n),
                "weighted_degree_preg": int(
                    sum(G_preg[n][nbr]["weight"] for nbr in G_preg.neighbors(n))
                ),
                "betweenness_preg": betw_p.get(n, 0.0),
                "participation_coeff_preg": part_p.get(n, 0.0),
                "is_articulation_preg": n in artic_p,
                "ob_flag": ob_flag[n],
                "cross_share_to_other": cross_share,
                "has_ob_and_non_ob_neighbors": has_both,
            }
        )

    nodes_preg = pd.DataFrame(rows_p)
    nodes_preg["diagnosis"] = nodes_preg["diagnosis"].astype(str)
    
    # Add concept labels for visualization
    concept_labels = concept_cond[["concept_id", "concept_name"]].copy()
    concept_labels["concept_id"] = concept_labels["concept_id"].astype(str)
    concept_labels = concept_labels.rename(columns={"concept_id": "diagnosis", "concept_name": "label"})
    nodes_preg = nodes_preg.merge(concept_labels, on="diagnosis", how="left")
    nodes_preg["label"] = nodes_preg["label"].fillna(nodes_preg["diagnosis"])

    # 10) Age stats per diagnosis in pregnancy cohort
    #    Use df_preg_age, which already has age_at_visit
    age_stats = (
        df_preg_age.groupby("diag_code")["age_at_visit"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(
            columns={
                "diag_code": "diagnosis",
                "mean": "age_mean",
                "std": "age_std",
            }
        )
    )

    nodes_preg = nodes_preg.merge(age_stats, on="diagnosis", how="left")

    # handle missing ages: fill mean with global mean, std with 0
    global_age_mean = nodes_preg["age_mean"].mean()
    nodes_preg["age_mean"] = nodes_preg["age_mean"].fillna(global_age_mean)
    nodes_preg["age_std"] = nodes_preg["age_std"].fillna(0.0)

    # 11) Standardize numeric features and build X
    feature_cols = [
        "degree_preg",
        "weighted_degree_preg",
        "betweenness_preg",
        "participation_coeff_preg",
        "cross_share_to_other",
        "age_mean",
        "age_std",
    ]

    for col in feature_cols:
        nodes_preg[col] = zscore_series(nodes_preg[col])

    nodes_preg["ob_flag"] = nodes_preg["ob_flag"].astype(float)
    nodes_preg["has_ob_and_non_ob_neighbors"] = nodes_preg[
        "has_ob_and_non_ob_neighbors"
    ].astype(float)

    feature_cols_full = feature_cols + ["ob_flag", "has_ob_and_non_ob_neighbors"]

    # Fix a stable node index order
    nodes = sorted(G_preg.nodes())
    diag2idx = {d: i for i, d in enumerate(nodes)}

    X = (
        nodes_preg.set_index("diagnosis")
        .loc[nodes, feature_cols_full]
        .to_numpy(dtype=np.float32)
    )

    # 12) Edge pairs in index space
    edge_pairs = []
    for a, b, w in edges_preg.itertuples(index=False):
        if a in diag2idx and b in diag2idx:
            edge_pairs.append((diag2idx[a], diag2idx[b]))
    edge_pairs = np.array(edge_pairs, dtype=np.int64)

    ob_mask = np.array([ob_flag[d] == 1 for d in nodes], dtype=bool)

    print(
        f"[Pregnancy] Feature matrix X shape: {X.shape}, num edges: {edge_pairs.shape[0]}"
    )
    
    # Generate graph visualization (unique filename based on config)
    try:
        graph_viz_name = f"pregnancy_graph_mindiag{MIN_DIAG_COUNT}_topn{TOP_N_DIAG}.png"
        plot_pregnancy_graph(G_preg, nodes_preg, comm_id_of_preg, graph_viz_name)
    except Exception as e:
        import traceback
        print(f"[Warning] Failed to generate graph visualization: {e}")
        traceback.print_exc()
    
    return nodes, X, edge_pairs, ob_mask


# ---------------------------------------------------------------------
# splitting and negatives
# ---------------------------------------------------------------------


def split_edges_for_link_prediction(edge_pairs, ob_mask, train_ratio=0.7, val_ratio=0.15, seed=42):
    rng = np.random.RandomState(seed)

    # Positive edges: one obstetric, one non obstetric
    pos_idx = []
    for i, (u, v) in enumerate(edge_pairs):
        cross = (ob_mask[u] and not ob_mask[v]) or (ob_mask[v] and not ob_mask[u])
        if cross:
            pos_idx.append(i)
    pos_idx = np.array(pos_idx, dtype=np.int64)

    if len(pos_idx) < 10:
        print(
            f"[WARN] Only {len(pos_idx)} obstetric non obstetric edges found. Model may be unstable."
        )

    pos_edges = edge_pairs[pos_idx]
    num_pos = len(pos_edges)
    perm = rng.permutation(num_pos)

    train_end = int(train_ratio * num_pos)
    val_end = int((train_ratio + val_ratio) * num_pos)

    train_ids = perm[:train_end]
    val_ids = perm[train_end:val_end]
    test_ids = perm[val_end:]

    train_pos = pos_edges[train_ids]
    val_pos = pos_edges[val_ids]
    test_pos = pos_edges[test_ids]

    held_out_indices = np.concatenate([pos_idx[val_ids], pos_idx[test_ids]])
    mask_train_edges = np.ones(edge_pairs.shape[0], dtype=bool)
    mask_train_edges[held_out_indices] = False
    train_edge_pairs_for_gnn = edge_pairs[mask_train_edges]

    print(
        f"[Split] Positive edges - train: {train_pos.shape[0]}, val: {val_pos.shape[0]}, test: {test_pos.shape[0]}"
    )
    print(
        f"[Split] Training adjacency uses {train_edge_pairs_for_gnn.shape[0]} edges out of {edge_pairs.shape[0]} total."
    )

    return train_pos, val_pos, test_pos, train_edge_pairs_for_gnn


def sample_negative_edges(num_samples, ob_mask, existing_pairs, seed=42):
    rng = np.random.RandomState(seed)
    num_nodes = len(ob_mask)
    ob_indices = np.where(ob_mask)[0]
    non_ob_indices = np.where(~ob_mask)[0]

    neg_edges = set()
    while len(neg_edges) < num_samples:
        u = int(rng.choice(ob_indices))
        v = int(rng.choice(non_ob_indices))
        if u == v:
            continue
        key = (u, v) if u < v else (v, u)
        if key in existing_pairs or key in neg_edges:
            continue
        neg_edges.add(key)

    neg_edges = np.array(list(neg_edges), dtype=np.int64)
    return neg_edges


# ---------------------------------------------------------------------
# GCN link prediction model
# ---------------------------------------------------------------------


class GCNLinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden_channels, 1)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x

    def decode(self, z, edge_index):
        src, dst = edge_index
        h = z[src] * z[dst]  # elementwise product
        return self.lin(h).view(-1)

    def forward(self, x, edge_index, pos_edge_index, neg_edge_index):
        z = self.encode(x, edge_index)
        pos_scores = self.decode(z, pos_edge_index)
        neg_scores = self.decode(z, neg_edge_index)
        return pos_scores, neg_scores


def compute_bce_loss(pos_scores, neg_scores):
    scores = torch.cat([pos_scores, neg_scores], dim=0)
    labels = torch.cat(
        [torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0
    )
    return F.binary_cross_entropy_with_logits(scores, labels)


@torch.no_grad()
def evaluate_auc_ap(model, x, edge_index, pos_edge_index, neg_edge_index, device="cpu"):
    from sklearn.metrics import roc_auc_score, average_precision_score

    model.eval()
    x = x.to(device)
    edge_index = edge_index.to(device)
    pos_edge_index = pos_edge_index.to(device)
    neg_edge_index = neg_edge_index.to(device)

    pos_scores, neg_scores = model(x, edge_index, pos_edge_index, neg_edge_index)
    scores = torch.cat([pos_scores, neg_scores], dim=0).cpu().numpy()
    labels = np.concatenate(
        [
            np.ones(pos_scores.shape[0], dtype=np.int32),
            np.zeros(neg_scores.shape[0], dtype=np.int32),
        ]
    )

    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    return auc, ap


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


def main():
    check_cuda_compatibility()
    set_seed(42)
    
    # Use CPU for now to avoid segfault - GTX 1080 Ti with CUDA 12.1 may have compatibility issues
    device = torch.device("cpu")
    print(f"\nUsing device: {device}")
    print("Note: Running on CPU to avoid CUDA compatibility issues with GTX 1080 Ti")
    print("=" * 60)
    print()

    nodes, X, edge_pairs, ob_mask = build_pregnancy_graph_and_features()
    num_nodes, in_channels = X.shape

    (
        train_pos,
        val_pos,
        test_pos,
        train_edge_pairs_for_gnn,
    ) = split_edges_for_link_prediction(edge_pairs, ob_mask, seed=42)

    existing_pairs = {(min(u, v), max(u, v)) for (u, v) in edge_pairs.tolist()}

    train_neg = sample_negative_edges(
        num_samples=train_pos.shape[0],
        ob_mask=ob_mask,
        existing_pairs=existing_pairs,
        seed=1,
    )
    val_neg = sample_negative_edges(
        num_samples=val_pos.shape[0],
        ob_mask=ob_mask,
        existing_pairs=existing_pairs,
        seed=2,
    )
    test_neg = sample_negative_edges(
        num_samples=test_pos.shape[0],
        ob_mask=ob_mask,
        existing_pairs=existing_pairs,
        seed=3,
    )

    train_src = train_edge_pairs_for_gnn[:, 0]
    train_dst = train_edge_pairs_for_gnn[:, 1]
    edge_index_train = np.vstack(
        [np.concatenate([train_src, train_dst]), np.concatenate([train_dst, train_src])]
    )

    x = torch.tensor(X, dtype=torch.float)
    edge_index_train = torch.tensor(edge_index_train, dtype=torch.long)

    train_pos_edge_index = torch.tensor(train_pos.T, dtype=torch.long)
    val_pos_edge_index = torch.tensor(val_pos.T, dtype=torch.long)
    test_pos_edge_index = torch.tensor(test_pos.T, dtype=torch.long)

    train_neg_edge_index = torch.tensor(train_neg.T, dtype=torch.long)
    val_neg_edge_index = torch.tensor(val_neg.T, dtype=torch.long)
    test_neg_edge_index = torch.tensor(test_neg.T, dtype=torch.long)

    model = GCNLinkPredictor(in_channels=in_channels, hidden_channels=64, dropout=0.3)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    x = x.to(device)
    edge_index_train = edge_index_train.to(device)

    num_epochs = 100
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        pos_scores, neg_scores = model(
            x,
            edge_index_train,
            train_pos_edge_index.to(device),
            train_neg_edge_index.to(device),
        )
        loss = compute_bce_loss(pos_scores, neg_scores)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            val_auc, val_ap = evaluate_auc_ap(
                model,
                x,
                edge_index_train,
                val_pos_edge_index,
                val_neg_edge_index,
                device=device,
            )
            print(
                f"Epoch {epoch:03d} | Loss {loss.item():.4f} | Val AUC {val_auc:.4f} | Val AP {val_ap:.4f}"
            )

    test_auc, test_ap = evaluate_auc_ap(
        model,
        x,
        edge_index_train,
        test_pos_edge_index,
        test_neg_edge_index,
        device=device,
    )
    print(f"[Test] AUC: {test_auc:.4f}  AP: {test_ap:.4f}")


if __name__ == "__main__":
    main()
