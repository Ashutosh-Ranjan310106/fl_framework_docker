"""Streamlit web application for CIFAR-10 Dataset Manager in Federated Learning."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Python 3.14+ compatibility defense for protobuf
import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
try:
    import google.protobuf.internal.api_implementation as _api_impl
    if hasattr(_api_impl, "_CanImport"):
        _orig_can_import = _api_impl._CanImport
        def _safe_can_import(mod_name):
            try:
                return _orig_can_import(mod_name)
            except Exception:
                return False
        _api_impl._CanImport = _safe_can_import
except Exception:
    pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from dataset_manager.config import (
    CIFAR10_CLASSES,
    DATASET_NAME,
    DEFAULT_CONFIG_PATH,
    DEFAULT_SEED,
    IMAGE_SHAPE,
    NUM_CLASSES,
    ORIGINAL_CIFAR10_DIR,
    ORIGINAL_TEST_DIR,
    ORIGINAL_TRAIN_DIR,
    PARTITIONS_ROOT,
    TOTAL_TEST_SAMPLES,
    TOTAL_TRAIN_SAMPLES,
    load_yaml_config,
)
from dataset_manager.dataset_loader import (
    download_cifar10,
    get_client_partition_path,
    load_client_partition,
    verify_original_dataset,
)
from dataset_manager.partition_manager import PartitionManager
from dataset_manager.statistics import (
    compute_partition_statistics,
    plot_class_distribution_heatmap,
    plot_overall_class_balance,
    plot_samples_per_client,
    plot_stacked_class_distribution,
)
from dataset_manager.storage_manager import (
    StorageManager,
    format_size,
    get_dir_size_bytes,
)
from dataset_manager.validator import validate_partition_directory

st.set_page_config(
    page_title="CIFAR-10 Dataset Manager | FL Security Framework",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_managers():
    """Cache storage and partition managers."""
    storage_mgr = StorageManager()
    partition_mgr = PartitionManager()
    return storage_mgr, partition_mgr


def load_partition_df(partition_path: Path) -> pd.DataFrame:
    """Load or generate statistics dataframe for a partition."""
    stats_csv = partition_path / "statistics.csv"
    if stats_csv.exists():
        return pd.read_csv(stats_csv)

    manifest_json = partition_path / "manifest.json"
    if manifest_json.exists():
        with open(manifest_json, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        client_parts = manifest.get("client_partitions", {})
        rows = []
        for cid, info in client_parts.items():
            row = {
                "client_id": cid,
                "sample_count": info["sample_count"],
                "num_classes": len(info.get("class_distribution", {})),
                "dominant_class_name": info.get("dominant_class", "N/A"),
                "entropy": info.get("entropy", 0.0),
            }
            c_dist = info.get("class_distribution", {})
            for c_idx, c_name in enumerate(CIFAR10_CLASSES):
                row[f"class_{c_idx}"] = c_dist.get(c_name, 0)
            rows.append(row)
        return pd.DataFrame(rows)
    return pd.DataFrame()


def render_dashboard(storage_mgr: StorageManager):
    st.header("📊 Dataset & Partition Dashboard")
    st.markdown(
        "Overview of CIFAR-10 dataset status, storage consumption, and generated federated learning partitions."
    )

    registry = storage_mgr.load_registry()
    status = registry.get("status", "NOT_DOWNLOADED")
    train_samples = registry.get("train_samples", 0)
    test_samples = registry.get("test_samples", 0)

    partitions = storage_mgr.list_partitions()
    total_partitions = len(partitions)
    valid_partitions = sum(1 for p in partitions if p.get("is_valid"))

    orig_size = get_dir_size_bytes(storage_mgr.original_dir)
    part_size = get_dir_size_bytes(storage_mgr.partitions_dir)
    total_size = orig_size + part_size

    # KPI Metric Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Dataset Status", status, delta="Verified" if status == "DOWNLOADED" else "Attention")
    with col2:
        st.metric("Training Samples", f"{train_samples:,}", f"Target: {TOTAL_TRAIN_SAMPLES:,}")
    with col3:
        st.metric("Testing Samples", f"{test_samples:,}", f"Target: {TOTAL_TEST_SAMPLES:,}")
    with col4:
        st.metric("Partitions", f"{total_partitions}", f"{valid_partitions} Valid")
    with col5:
        st.metric("Total Storage", format_size(total_size), f"Partitions: {format_size(part_size)}")

    st.markdown("---")

    # Quick overview split
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("📁 Storage Breakdown")
        breakdown_df = pd.DataFrame([
            {"Component": "Original CIFAR-10 Dataset", "Path": str(storage_mgr.original_dir), "Size": format_size(orig_size)},
            {"Component": "Generated Partitions", "Path": str(storage_mgr.partitions_dir), "Size": format_size(part_size)},
            {"Component": "Metadata & Registry", "Path": str(storage_mgr.metadata_dir), "Size": format_size(get_dir_size_bytes(storage_mgr.metadata_dir))},
        ])
        st.dataframe(breakdown_df, hide_index=True, use_container_width=True)

        if status != "DOWNLOADED":
            st.warning("⚠️ CIFAR-10 dataset has not been downloaded or verified yet. Go to **Dataset Management**.")
        else:
            st.success("✅ CIFAR-10 is downloaded and ready for partitioning.")

    with right_col:
        st.subheader("📑 Recent Partitions")
        if partitions:
            recent_summary = []
            for p in partitions[:5]:
                m = p.get("manifest") or {}
                recent_summary.append({
                    "Name": p["name"],
                    "Strategy": p["strategy"].upper(),
                    "Clients": p.get("num_clients", "N/A"),
                    "Seed": p.get("random_seed", "N/A"),
                    "Size": p["size_formatted"],
                    "Status": "✅ Valid" if p["is_valid"] else "❌ Incomplete",
                })
            st.dataframe(pd.DataFrame(recent_summary), hide_index=True, use_container_width=True)
        else:
            st.info("No partition configurations generated yet. Use the **Create Partitions** tab.")


def render_dataset_management(storage_mgr: StorageManager):
    st.header("📥 CIFAR-10 Dataset Management")
    st.markdown(
        "Download, inspect, and verify the integrity of the original CIFAR-10 dataset. "
        "The original test set is maintained separately and is never distributed to clients."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Dataset Actions")
        if st.button("🚀 Download / Verify CIFAR-10", type="primary", use_container_width=True):
            with st.spinner("Downloading and verifying CIFAR-10 dataset..."):
                try:
                    res = download_cifar10(storage_mgr.original_dir)
                    if res["is_valid"]:
                        st.success("CIFAR-10 dataset downloaded and verified successfully!")
                    else:
                        st.error(f"Integrity check failed: {res['errors']}")
                except Exception as e:
                    st.error(f"Download failed: {e}")

        if st.button("🔍 Run Integrity Verification Only", use_container_width=True):
            with st.spinner("Checking hashes and sample shapes..."):
                res = verify_original_dataset(storage_mgr.original_dir)
                if res["is_valid"]:
                    st.success("Integrity verified! All 50,000 train and 10,000 test samples are sound.")
                    for c in res["checks"]:
                        st.markdown(f"- ✅ {c}")
                else:
                    st.error("Verification failed!")
                    for err in res["errors"]:
                        st.markdown(f"- ❌ {err}")

    with col2:
        st.subheader("Dataset Specifications")
        registry = storage_mgr.load_registry()
        st.json({
            "Dataset": DATASET_NAME,
            "Storage Location": str(storage_mgr.original_dir),
            "Train Split": f"{registry.get('train_samples', 0)} samples (3x32x32 uint8)",
            "Test Split": f"{registry.get('test_samples', 0)} samples (separate)",
            "Number of Classes": NUM_CLASSES,
            "Class Names": CIFAR10_CLASSES,
            "Status": registry.get("status", "NOT_DOWNLOADED"),
            "Last Verified": registry.get("last_verified_timestamp", "Never"),
        })


def render_create_partitions(partition_mgr: PartitionManager, storage_mgr: StorageManager):
    st.header("⚙️ Create Dataset Partitions")
    st.markdown(
        "Generate reproducible, isolated dataset partitions across clients using controlled seeds and robust validation."
    )

    # Strategy Selection
    strategies = {
        "IID": "Uniform random distribution of samples equally among clients.",
        "Dirichlet": "Non-IID label distribution parameterized by alpha (lower alpha = higher skew).",
        "Label Skew": "Non-IID partitioning where each client receives samples from a fixed number of classes.",
        "Quantity Skew": "Clients receive different quantities of samples while preserving the total 50,000 samples.",
        "Custom": "User-configured sample count allocation per client.",
    }

    col_strat, col_clients, col_seed = st.columns([1.5, 1, 1])

    with col_strat:
        strategy_label = st.selectbox("Partition Strategy", list(strategies.keys()), index=0)
        strategy_key = strategy_label.lower().replace(" ", "_")
        st.caption(strategies[strategy_label])

    with col_clients:
        client_presets = [2, 5, 10, 20, 50, 100, "Custom"]
        selected_client_preset = st.selectbox("Number of Clients", client_presets, index=2)
        if selected_client_preset == "Custom":
            num_clients = st.number_input("Custom Client Count", min_value=1, max_value=500, value=15, step=1)
        else:
            num_clients = int(selected_client_preset)

    with col_seed:
        random_seed = st.number_input(
            "Random Seed (Reproducibility)", min_value=0, max_value=9999999, value=DEFAULT_SEED, step=1
        )
        st.caption("Controlled random generator ensures 100% reproducible splitting.")

    # Strategy-specific parameter configuration
    strategy_params = {}
    st.markdown("#### Strategy Parameters")

    if strategy_key == "iid":
        st.info(f"Each of the {num_clients} clients will receive approximately {TOTAL_TRAIN_SAMPLES // num_clients} samples.")

    elif strategy_key == "dirichlet":
        col_alpha, col_info = st.columns([1, 1])
        with col_alpha:
            alpha_options = [0.1, 0.3, 0.5, 1.0, 10.0, "Custom"]
            chosen_alpha = st.selectbox("Dirichlet Alpha (α)", alpha_options, index=2)
            if chosen_alpha == "Custom":
                alpha_val = st.number_input("Custom Alpha", min_value=0.01, max_value=100.0, value=0.5, step=0.05)
            else:
                alpha_val = float(chosen_alpha)
            strategy_params["alpha"] = alpha_val
        with col_info:
            st.markdown(
                f"- **α = {alpha_val}**: {'Very high label imbalance' if alpha_val <= 0.3 else 'Moderate label skew' if alpha_val <= 1.0 else 'Near-uniform / IID'}.\n"
                f"- Samples are allocated deterministically using seed `{random_seed}`."
            )

    elif strategy_key == "label_skew":
        col_cpc, col_cpc_info = st.columns([1, 1])
        with col_cpc:
            cpc = st.slider("Classes per Client", min_value=1, max_value=NUM_CLASSES, value=2)
            strategy_params["classes_per_client"] = cpc
        with col_cpc_info:
            total_slots = num_clients * cpc
            if total_slots < NUM_CLASSES:
                st.error(
                    f"⚠️ Total slots ({total_slots}) < {NUM_CLASSES} classes. "
                    f"Increase clients or classes per client to allocate all training samples."
                )
            else:
                st.success(f"Valid: {total_slots} slots covers all {NUM_CLASSES} classes.")

    elif strategy_key == "quantity_skew":
        col_min, col_max = st.columns(2)
        with col_min:
            default_min = max(50, (TOTAL_TRAIN_SAMPLES // num_clients) // 3)
            min_samples = st.number_input("Min Samples per Client", min_value=10, max_value=TOTAL_TRAIN_SAMPLES, value=default_min, step=50)
            strategy_params["min_samples_per_client"] = int(min_samples)
        with col_max:
            default_max = min(TOTAL_TRAIN_SAMPLES, (TOTAL_TRAIN_SAMPLES // num_clients) * 2)
            max_samples = st.number_input("Max Samples per Client", min_value=int(min_samples), max_value=TOTAL_TRAIN_SAMPLES, value=default_max, step=50)
            strategy_params["max_samples_per_client"] = int(max_samples)

        min_total = num_clients * min_samples
        max_total = num_clients * max_samples
        if min_total > TOTAL_TRAIN_SAMPLES or max_total < TOTAL_TRAIN_SAMPLES:
            st.error(
                f"⚠️ Quantity constraints impossible: {num_clients} clients with range [{min_samples}, {max_samples}] "
                f"requires total sum between {min_total:,} and {max_total:,}, but dataset has {TOTAL_TRAIN_SAMPLES:,}."
            )
        else:
            st.success(f"Feasible: {TOTAL_TRAIN_SAMPLES:,} samples falls within range [{min_total:,}, {max_total:,}].")

    elif strategy_key == "custom":
        st.markdown("Define sample allocation for each client (must sum to 50,000):")
        custom_counts = {}
        cols = st.columns(min(5, num_clients))
        base_share = TOTAL_TRAIN_SAMPLES // num_clients
        for i in range(num_clients):
            cid = f"client_{i+1:03d}"
            with cols[i % len(cols)]:
                cnt = st.number_input(f"{cid}", min_value=0, max_value=TOTAL_TRAIN_SAMPLES, value=base_share, step=100, key=f"custom_{cid}")
                custom_counts[cid] = int(cnt)
        total_custom = sum(custom_counts.values())
        strategy_params["client_counts"] = custom_counts
        diff = total_custom - TOTAL_TRAIN_SAMPLES
        if diff != 0:
            st.error(f"Total custom samples: {total_custom:,} (Difference: {diff:+d}). Must sum to {TOTAL_TRAIN_SAMPLES:,}!")
        else:
            st.success(f"Total custom samples sum to {TOTAL_TRAIN_SAMPLES:,}.")

    st.markdown("---")

    # Configuration Preview & Confirmation
    st.subheader("📋 Configuration Summary")
    summary_box = {
        "Dataset": DATASET_NAME,
        "Partition Strategy": strategy_key,
        "Client Count": num_clients,
        "Random Seed": random_seed,
        "Total Training Samples": TOTAL_TRAIN_SAMPLES,
        "Parameters": strategy_params,
    }
    st.json(summary_box)

    overwrite = st.checkbox("Overwrite existing partition if directory already exists", value=False)
    confirm = st.checkbox("I confirm that I want to generate this partition configuration.", value=False)

    if st.button("🚀 Generate Dataset Partitions", type="primary", disabled=not confirm):
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(pct: float, msg: str):
            progress_bar.progress(pct)
            status_text.text(msg)

        try:
            result = partition_mgr.generate_partitions(
                strategy=strategy_key,
                num_clients=num_clients,
                seed=random_seed,
                strategy_params=strategy_params,
                overwrite=overwrite,
                progress_callback=update_progress,
            )
            st.success(f"Partitions ready! Storage path: `{result['path']}`")
            st.session_state["selected_partition_path"] = str(result["path"])
            st.info("Head to the **Partition Analysis** tab to view class distributions and heatmaps.")
        except Exception as e:
            st.error(f"Partition generation failed: {e}")


def render_partition_analysis(storage_mgr: StorageManager):
    st.header("📈 Partition Analysis & Visualizations")
    st.markdown("Examine sample counts, class distributions, and heterogeneity metrics for any generated partition.")

    partitions = storage_mgr.list_partitions()
    if not partitions:
        st.info("No generated partitions found. Please create one in the **Create Partitions** tab.")
        return

    partition_names = [p["name"] for p in partitions]
    selected_name = st.selectbox(
        "Select Partition to Analyze",
        partition_names,
        index=0,
    )
    selected_part = next((p for p in partitions if p["name"] == selected_name), None)

    if not selected_part:
        return

    part_dir = Path(selected_part["path"])
    df = load_partition_df(part_dir)
    manifest = selected_part.get("manifest") or {}

    # Header Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Strategy", selected_part["strategy"].upper())
    with col2:
        st.metric("Clients", manifest.get("num_clients", len(df)))
    with col3:
        st.metric("Seed", manifest.get("random_seed", "N/A"))
    with col4:
        st.metric("Min Samples", manifest.get("min_samples_per_client", int(df["sample_count"].min()) if not df.empty else 0))
    with col5:
        st.metric("Max Samples", manifest.get("max_samples_per_client", int(df["sample_count"].max()) if not df.empty else 0))

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Sample Distribution",
        "🔥 Class Heatmap",
        "🥞 Class Composition",
        "⚖️ Class Totals Check",
        "📋 Data Table",
    ])

    with tab1:
        st.subheader("Samples Allocated per Client")
        if not df.empty:
            fig = plot_samples_per_client(df, title=f"Sample Counts per Client ({selected_name})")
            st.pyplot(fig)
            plt.close(fig)

    with tab2:
        st.subheader("Class Distribution Heatmap")
        if not df.empty:
            fig = plot_class_distribution_heatmap(df, title=f"Class Distribution Heatmap ({selected_name})")
            st.pyplot(fig)
            plt.close(fig)

    with tab3:
        st.subheader("Stacked Class Composition per Client")
        if not df.empty:
            fig = plot_stacked_class_distribution(df, title=f"Class Composition Breakdown ({selected_name})")
            st.pyplot(fig)
            plt.close(fig)

    with tab4:
        st.subheader("Global Class Allocation Totals")
        if not df.empty:
            fig = plot_overall_class_balance(df, title=f"Global Class Allocation Verification ({selected_name})")
            st.pyplot(fig)
            plt.close(fig)

    with tab5:
        st.subheader("Partition Statistics Table")
        if not df.empty:
            st.dataframe(df, use_container_width=True)
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download statistics.csv",
                data=csv_data,
                file_name=f"{selected_name}_statistics.csv",
                mime="text/csv",
            )


def render_existing_partitions(storage_mgr: StorageManager):
    st.header("📁 Existing Partitions Manager")
    st.markdown(
        "Browse, inspect client tensors, re-validate integrity, export Docker volume configurations, and safely manage partitions."
    )

    partitions = storage_mgr.list_partitions()
    if not partitions:
        st.info("No generated partitions found.")
        return

    partition_names = [p["name"] for p in partitions]
    selected_name = st.selectbox("Choose Partition", partition_names, index=0)
    selected_part = next((p for p in partitions if p["name"] == selected_name), None)
    if not selected_part:
        return

    part_dir = Path(selected_part["path"])
    manifest = selected_part.get("manifest") or {}

    col_meta, col_actions = st.columns([2, 1])

    with col_meta:
        st.subheader("Partition Manifest & Metadata")
        st.json(manifest)

    with col_actions:
        st.subheader("Partition Operations")

        # 1. Re-validate
        if st.button("🔄 Re-Validate on Disk", use_container_width=True):
            with st.spinner("Exhaustively checking 10 validation criteria..."):
                val_res = validate_partition_directory(part_dir)
                if val_res["is_valid"]:
                    st.success("All 10 validation checks passed!")
                    for c in val_res["checks_passed"]:
                        st.markdown(f"- ✅ {c}")
                else:
                    st.error("Validation failed!")
                    for e in val_res["errors"]:
                        st.markdown(f"- ❌ {e}")

        # 2. Inspect individual client data
        st.markdown("#### Inspect Client Tensor")
        client_dirs = sorted([p.name for p in part_dir.iterdir() if p.is_dir() and p.name.startswith("client_")])
        if client_dirs:
            chosen_client = st.selectbox("Select Client", client_dirs)
            if st.button("🔍 Load Client data.pt"):
                try:
                    c_data = load_client_partition(part_dir / chosen_client)
                    st.write(f"**Images shape:** {tuple(c_data['images'].shape)} (dtype: {c_data['images'].dtype})")
                    st.write(f"**Labels shape:** {tuple(c_data['labels'].shape)}")
                    st.write(f"**Sample indices count:** {len(c_data['sample_indices'])}")
                    st.write(f"**Sample indices slice:** {c_data['sample_indices'][:10].tolist()}...")
                except Exception as e:
                    st.error(f"Failed to load: {e}")

        # 3. Docker & Flower code snippets
        st.markdown("#### Flower & Docker Integration")
        with st.expander("🐳 Docker Volume Mount Syntax"):
            st.code(
                f"""volumes:
  - {part_dir.resolve()}/client_001:/app/data:ro
""",
                language="yaml",
            )
        with st.expander("🌸 Flower Client Loading Example"):
            st.code(
                f"""from dataset_manager.dataset_loader import load_client_partition, CIFAR10ClientDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# In Flower ClientApp:
partition_path = r"{part_dir.resolve()}/client_001"
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

dataset = CIFAR10ClientDataset(partition_path=partition_path, transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
""",
                language="python",
            )

        # 4. Safe Delete with confirmation
        st.markdown("#### Delete Partition")
        st.caption("Permanently deletes this partition configuration. Original dataset is protected.")
        confirm_del = st.checkbox(f"Confirm deletion of {selected_name}", key=f"del_{selected_name}")
        if st.button("🗑️ Delete Partition", type="secondary", disabled=not confirm_del):
            try:
                storage_mgr.delete_partition(part_dir)
                st.success(f"Deleted {selected_name} successfully.")
                st.rerun()
            except Exception as e:
                st.error(f"Deletion failed: {e}")


def main():
    storage_mgr, partition_mgr = get_managers()

    st.sidebar.title("🛡️ FL Dataset Manager")
    st.sidebar.markdown("**CIFAR-10 Partitioning Suite**")

    menu = st.sidebar.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "📥 Dataset Management",
            "⚙️ Create Partitions",
            "📈 Partition Analysis",
            "📁 Existing Partitions",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Federated Learning Security & Benchmarking Framework")

    if menu == "📊 Dashboard":
        render_dashboard(storage_mgr)
    elif menu == "📥 Dataset Management":
        render_dataset_management(storage_mgr)
    elif menu == "⚙️ Create Partitions":
        render_create_partitions(partition_mgr, storage_mgr)
    elif menu == "📈 Partition Analysis":
        render_partition_analysis(storage_mgr)
    elif menu == "📁 Existing Partitions":
        render_existing_partitions(storage_mgr)


if __name__ == "__main__":
    main()
