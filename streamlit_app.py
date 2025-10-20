"""Streamlit-Anwendung für das BioCortex-Projekt."""
from __future__ import annotations

import glob
import io
import pathlib
from dataclasses import asdict
from typing import Iterable, Callable

import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

from biollm import (
    BioCortex,
    BioLLMConfig,
    BioBPEConfig,
    KNConfig,
    PlasticityConfig,
    ReplayConfig,
)

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit Setup
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="BioCortex Studio", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ──────────────────────────────────────────────────────────────────────────────
def _read_manifest(path: str = "manifest.md") -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return "# Manifest\n_Datei `manifest.md` nicht gefunden. Lege sie im Projektordner ab._"


def _display_logo(*, width: int = 96) -> None:
    logo_candidates = [
        "logo.png",
        "assets/logo.png",
        "/mnt/data/logo.png",
        "/mnt/data/12289b6d-cf34-47b8-8c02-07220117e6f1.png",
    ]
    for candidate in logo_candidates:
        try:
            st.image(candidate, width=width)
            break
        except Exception:
            continue


def _load_texts_from_upload(files: Iterable[io.BytesIO]) -> list[str]:
    """Lädt Upload-Dateien vollständig in den RAM (klassisch)."""
    texts: list[str] = []
    for fh in files:
        try:
            data = fh.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            data = fh.getvalue().decode("utf-8", errors="ignore")
        texts.append(data)
    return texts


def incremental_train_from_uploads(
    model: BioCortex,
    files: Iterable[io.BytesIO],
    *,
    lines_per_sample: int = 1_000,
    encoding: str = "utf-8",
    progress_cb: Callable[[str, int, int, str], None] | None = None,
) -> None:
    """
    RAM-schonendes, inkrementelles Training:
    - Liest jede Upload-Datei zeilenweise.
    - Baut Chunks von `lines_per_sample` zusammen.
    - Ruft für jeden Chunk `model.partial_fit([chunk])` auf.
    - Optional: Fortschritt via `progress_cb(stage, step, total, detail)`.
    """

    def _chunks_from_text(s: str) -> Iterable[str]:
        buf: list[str] = []
        for ln in s.splitlines():
            ln = ln.strip("\n")
            if ln:
                buf.append(ln)
            if len(buf) >= lines_per_sample:
                yield "\n".join(buf)
                buf.clear()
        if buf:
            yield "\n".join(buf)

    file_idx = 0
    for fh in files:
        file_idx += 1
        try:
            try:
                data = fh.getvalue().decode(encoding)
            except UnicodeDecodeError:
                data = fh.getvalue().decode(encoding, errors="ignore")

            for chunk_idx, chunk in enumerate(_chunks_from_text(data), start=1):
                if progress_cb:
                    progress_cb(
                        stage=f"Upload {file_idx}",
                        step=chunk_idx,
                        total=0,
                        detail=f"Chunk {chunk_idx} · ~{lines_per_sample} Zeilen",
                    )
                model.partial_fit([chunk])
        except Exception as exc:
            if progress_cb:
                progress_cb(stage=f"Upload {file_idx}", step=0, total=0, detail=f"Übersprungen: {exc}")
            continue


def _current_model() -> BioCortex | None:
    return st.session_state.get("biocortex_model")


def _set_model(model: BioCortex) -> None:
    st.session_state.pop("biollm_model", None)
    st.session_state.pop("biollm_config", None)
    st.session_state["biocortex_model"] = model
    st.session_state["biocortex_config"] = asdict(model.cfg)
    st.session_state["last_generation_trace"] = None
    st.session_state["biocortex_model_active"] = True


@st.cache_data(show_spinner=False)
def _spring_layout_positions(
    edges: tuple[tuple[str, str], ...],
    *,
    k: float = 0.8,
    seed: int = 42,
    iterations: int = 30,
) -> dict[str, tuple[float, float]]:
    if not edges:
        return {}
    graph = nx.Graph()
    graph.add_edges_from(edges)
    pos = nx.spring_layout(graph, k=k, seed=seed, iterations=iterations)
    return {node: (float(coord[0]), float(coord[1])) for node, coord in pos.items()}


def visualize_graph(bio_model: BioCortex, max_edges: int = 200) -> None:
    if not bio_model.graph.w:
        st.info("Noch keine Synapsengewichte vorhanden.")
        return
    G = nx.DiGraph()
    raw_edges = list(bio_model.graph.w.items())[:max_edges]
    layout_edges: list[tuple[str, str]] = []
    for (a, b), w in raw_edges:
        pher = bio_model.graph.pher.get((a, b), 0.0)
        label_a = bio_model.tokenizer.inv_vocab[a] if a >= 0 else str(a)
        label_b = bio_model.tokenizer.inv_vocab[b] if b >= 0 else str(b)
        G.add_edge(label_a, label_b, weight=w, pher=pher)
        layout_edges.append((label_a, label_b))

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = _spring_layout_positions(tuple(layout_edges))
    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    pheromones = [d.get("pher", 0.0) for _, _, d in G.edges(data=True)]
    max_pher = max(pheromones) if pheromones else 0.0
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=300,
        font_size=8,
        edge_color=weights,
        edge_cmap=plt.cm.plasma,
        arrowsize=8,
        width=[0.5 + 2.0 * (p / max_pher if max_pher else 0.0) for p in pheromones],
        ax=ax,
    )
    ax.set_title("Bio-Synapsen-Netzwerk")
    st.pyplot(fig)
    plt.close(fig)
    degrees = [deg for _, deg in G.degree()]
    clustering_graph = G.to_undirected() if len(G) else G
    stats = {
        "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
        "clustering_coeff": float(nx.average_clustering(clustering_graph)) if len(clustering_graph) > 0 else 0.0,
        "density": float(nx.density(G)) if len(G) > 1 else 0.0,
    }
    st.json(stats, expanded=False)


def plot_modulators(mod_history: list[dict[str, float]]) -> None:
    if not mod_history:
        st.info("Noch keine Neuromodulator-Aktivität vorhanden.")
        return
    df = pd.DataFrame(mod_history)
    df.index.name = "Schritt"
    fig, ax = plt.subplots()
    df.plot(ax=ax)
    ax.set_title("Neuromodulatorische Aktivität über Zeit")
    ax.set_xlabel("Zeit / Sequenzschritt")
    ax.set_ylabel("Signalstärke")
    ax.legend(loc="upper right")
    st.pyplot(fig)
    plt.close(fig)


def plot_replay_activity(replay_log: list[dict[str, float]]) -> None:
    if not replay_log:
        st.info("Noch keine Replay-Aktivität vorhanden.")
        return
    df = pd.DataFrame(replay_log)
    if "step" in df.columns:
        df = df.set_index("step")
    fig, ax = plt.subplots()
    df.plot(ax=ax, kind="line")
    ax.set_title("Replay-Aktivität über Zeit")
    ax.set_xlabel("Zeit / Sequenzschritt")
    ax.set_ylabel("Intensität")
    st.pyplot(fig)
    plt.close(fig)


def _config_from_sidebar() -> BioLLMConfig:
    st.sidebar.header("Konfiguration")
    st.sidebar.subheader("📜 Letzte Logdateien")
    log_files = sorted(glob.glob("logs/*.log"), reverse=True)[:5]
    if not log_files:
        st.sidebar.caption("Noch keine Logdateien vorhanden.")
    for lf in log_files:
        try:
            with open(lf, "r", encoding="utf-8") as f:
                lines = f.readlines()[-5:]
            st.sidebar.text(f"📄 {lf}\n" + "".join(lines))
        except OSError:
            st.sidebar.warning(f"Logdatei konnte nicht gelesen werden: {lf}")

    with st.sidebar.expander("Tokenizer (BioBPE)", expanded=False):
        vocab_size = st.number_input("Vokabulargröße", min_value=64, max_value=8192, value=4000, step=64)
        min_pair_freq = st.number_input("Min. Paarhäufigkeit", min_value=1, max_value=20, value=2)
        lowercase = st.checkbox("Kleinschreibung erzwingen", value=True)

    with st.sidebar.expander("Kneser-Ney LM", expanded=False):
        order = st.number_input("N-Gramm-Ordnung", min_value=2, max_value=7, value=5)
        discount = st.slider("Discount", min_value=0.1, max_value=1.0, value=0.75, step=0.05)

    with st.sidebar.expander("Plastizität", expanded=False):
        a_plus = st.slider("A+ (LTP)", min_value=0.0, max_value=0.2, value=0.05, step=0.01)
        a_minus = st.slider("A- (LTD)", min_value=0.0, max_value=0.2, value=0.02, step=0.01)
        tau = st.slider("Tau (Fenster)", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
        dopamine_gain = st.slider("Dopamin-Gewinn", 0.0, 1.0, 0.6, step=0.05)
        surprise_gain = st.slider("Überraschungs-Gewinn", 0.0, 1.0, 0.3, step=0.05)
        bond_gain = st.slider("Bindungs-Gewinn", 0.0, 1.0, 0.2, step=0.05)
        pher_evaporation = st.slider("Pheromon-Verdunstung", 0.0, 0.2, 0.05, step=0.01)
        pher_reinforce = st.slider("Pheromon-Verstärkung", 0.0, 0.5, 0.15, step=0.01)

    with st.sidebar.expander("Replay", expanded=False):
        buffer_size = st.number_input("Replay-Puffergröße", min_value=128, max_value=8192, value=2048, step=128)
        sample_len = st.number_input("Sample-Länge", min_value=8, max_value=256, value=64, step=8)
        nightly_samples = st.number_input("Replay-Samples", min_value=1, max_value=256, value=64)

    with st.sidebar.expander("Generierung", expanded=False):
        max_gen_len = st.slider("Max. neue Tokens", min_value=16, max_value=512, value=128, step=8)
        top_p = st.slider("Top-p", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
        gamma_bias = st.slider("Bio-Bias γ", min_value=0.0, max_value=3.0, value=1.4, step=0.1)

    return BioLLMConfig(
        bpe=BioBPEConfig(vocab_size=int(vocab_size), min_pair_freq=int(min_pair_freq), lowercase=lowercase),
        kn=KNConfig(order=int(order), discount=float(discount)),
        plastic=PlasticityConfig(
            a_plus=float(a_plus),
            a_minus=float(a_minus),
            tau=float(tau),
            dopamine_gain=float(dopamine_gain),
            surprise_gain=float(surprise_gain),
            bond_gain=float(bond_gain),
            pher_evaporation=float(pher_evaporation),
            pher_reinforce=float(pher_reinforce),
        ),
        replay=ReplayConfig(
            buffer_size=int(buffer_size),
            sample_len=int(sample_len),
            nightly_samples=int(nightly_samples),
        ),
        max_gen_len=int(max_gen_len),
        top_p=float(top_p),
        gamma_bias=float(gamma_bias),
    )

# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        _display_logo(width=96)
    with col_title:
        st.title("BioCortex Studio")
        st.caption("Erklärbares, bio-inspiriertes Lernsystem – STDP · Neuromodulation · Replay")

    cfg = _config_from_sidebar()

    tabs = st.tabs([
        "Training",
        "Generierung",
        "Synapsen-Graph",
        "Replay-Dynamik",
        "Modelle verwalten",
        "Manifest",
    ])

    # ── Tab: Training ─────────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Training & Feinjustierung")
        uploaded_files = st.file_uploader(
            "Trainingsdaten hochladen (TXT)",
            type=["txt"],
            accept_multiple_files=True,
        )
        extra_text = st.text_area("Zusätzlicher Trainingstext", height=200)

        if st.button("Neues Modell trainieren"):
            has_uploads = bool(uploaded_files)
            has_extra = bool(extra_text.strip())
            if not has_uploads and not has_extra:
                st.warning("Bitte mindestens eine Datei hochladen oder Zusatztext eingeben.")
            else:
                progress_bar = st.progress(0.0, text="Starte Training…")
                stage_placeholder = st.empty()
                phase_placeholder = st.empty()
                detail_placeholder = st.empty()
                history_placeholder = st.empty()
                events: list[str] = []

                def _update_history() -> None:
                    if events:
                        history_placeholder.markdown("\n".join(f"- {entry}" for entry in events[-8:]))
                    else:
                        history_placeholder.empty()

                def progress_callback(stage: str, step: int, total: int, detail: str) -> None:
                    fraction = 0.0 if total == 0 else step / total
                    progress_bar.progress(min(max(fraction, 0.0), 1.0), text=stage)
                    stage_placeholder.markdown(f"**{stage}** ({step}/{total})" if total else f"**{stage}**")
                    if stage.lower().startswith("replay"):
                        phase_placeholder.info("🟦 Schlaf/Replay-Phase aktiv")
                    else:
                        phase_placeholder.success("🟧 Wachzustand / aktives Lernen")
                    if detail:
                        detail_placeholder.write(detail)
                        events.append(f"{stage}: {detail}")
                    else:
                        detail_placeholder.empty()
                        events.append(stage)
                    _update_history()

                try:
                    model = BioCortex(cfg)

                    # Wichtig: Tokenizer/Lernen initialisieren mit kleinem Start
                    if has_uploads:
                        incremental_train_from_uploads(
                            model,
                            uploaded_files,
                            lines_per_sample=1_000,
                            progress_cb=progress_callback,
                        )
                    if has_extra:
                        model.partial_fit([extra_text], progress=progress_callback)

                except Exception as exc:
                    progress_bar.empty()
                    stage_placeholder.empty()
                    phase_placeholder.empty()
                    detail_placeholder.empty()
                    history_placeholder.empty()
                    st.error(f"Fehler beim Training: {exc}")
                else:
                    progress_bar.progress(1.0, text="Training abgeschlossen")
                    stage_placeholder.markdown("**Training abgeschlossen**")
                    detail_placeholder.empty()
                    phase_placeholder.empty()
                    _set_model(model)
                    st.success("Training abgeschlossen. Modell im Arbeitsspeicher verfügbar.")
                    st.markdown("### Trainings-Metadaten")
                    st.json(model.current_meta())

        if st.button("Weitertrainieren (aktuelles Modell)"):
            model = _current_model()
            if model is None:
                st.info("Kein Modell geladen. Bitte zuerst trainieren oder laden.")
            else:
                has_uploads = bool(uploaded_files)
                has_extra = bool(extra_text.strip())
                if not has_uploads and not has_extra:
                    st.warning("Für das Feintraining bitte Dateien hochladen oder Zusatztext eingeben.")
                else:
                    progress_bar = st.progress(0.0, text="Starte Feintraining…")
                    stage_placeholder = st.empty()
                    phase_placeholder = st.empty()
                    detail_placeholder = st.empty()
                    history_placeholder = st.empty()
                    events: list[str] = []

                    def _update_history() -> None:
                        if events:
                            history_placeholder.markdown("\n".join(f"- {entry}" for entry in events[-8:]))
                        else:
                            history_placeholder.empty()

                    def progress_callback(stage: str, step: int, total: int, detail: str) -> None:
                        fraction = 0.0 if total == 0 else step / total
                        progress_bar.progress(min(max(fraction, 0.0), 1.0), text=stage)
                        stage_placeholder.markdown(f"**{stage}** ({step}/{total})" if total else f"**{stage}**")
                        if stage.lower().startswith("replay"):
                            phase_placeholder.info("🟦 Schlaf/Replay-Phase aktiv")
                        else:
                            phase_placeholder.success("🟧 Wachzustand / aktives Lernen")
                        if detail:
                            detail_placeholder.write(detail)
                            events.append(f"{stage}: {detail}")
                        else:
                            detail_placeholder.empty()
                            events.append(stage)
                        _update_history()

                    try:
                        if has_uploads:
                            incremental_train_from_uploads(
                                model,
                                uploaded_files,
                                lines_per_sample=1_000,
                                progress_cb=progress_callback,
                            )
                        if has_extra:
                            model.partial_fit([extra_text], progress=progress_callback)

                    except Exception as exc:
                        progress_bar.empty()
                        stage_placeholder.empty()
                        phase_placeholder.empty()
                        detail_placeholder.empty()
                        history_placeholder.empty()
                        st.error(f"Fehler beim Feintraining: {exc}")
                    else:
                        progress_bar.progress(1.0, text="Feintraining abgeschlossen")
                        stage_placeholder.markdown("**Feintraining abgeschlossen**")
                        detail_placeholder.empty()
                        phase_placeholder.empty()
                        st.success("Feintraining abgeschlossen.")
                        st.markdown("### Aktualisierte Metadaten")
                        st.json(model.current_meta())

    # ── Tab: Generierung ───────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Textgenerierung")
        model = _current_model()
        if model is None:
            st.info("Noch kein Modell aktiv. Bitte trainieren oder laden.")
        else:
            prompt = st.text_area("Prompt", height=150)
            gen_max_new = st.slider(
                "Maximale neue Tokens",
                min_value=8,
                max_value=512,
                value=model.cfg.max_gen_len,
                step=8,
            )
            gen_top_p = st.slider("Top-p", min_value=0.1, max_value=1.0, value=model.cfg.top_p, step=0.05)
            gen_seed = st.number_input(
                "Seed (optional)",
                min_value=0,
                max_value=10_000_000,
                value=17,
                step=1,
            )
            if st.button("Text generieren"):
                if not prompt.strip():
                    st.warning("Bitte einen Prompt eingeben.")
                else:
                    with st.spinner("Generiere …"):
                        random.seed(int(gen_seed))
                        np.random.seed(int(gen_seed))
                        trace = model.generate_with_trace(
                            prompt,
                            max_new_tokens=int(gen_max_new),
                            top_p=float(gen_top_p),
                        )
                    st.session_state["last_generation_trace"] = trace
                    st.text_area("Generierter Text", value=trace.text, height=250)
            last_trace = st.session_state.get("last_generation_trace")
            if last_trace is not None:
                st.markdown("### Neuromodulatoren-Protokoll")
                plot_modulators(last_trace.mod_history)

    # ── Tab: Synapsen-Graph ───────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Synapsen-Graph")
        model = _current_model()
        if model is None:
            st.info("Kein Modell geladen. Bitte trainieren oder laden.")
        else:
            max_edges = st.slider("Maximale Kanten", min_value=20, max_value=500, value=200, step=20)
            visualize_graph(model, max_edges=max_edges)
            st.markdown("### Plastizitäts-Kennzahlen")
            st.json(model.current_meta())

    # ── Tab: Replay ───────────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Replay-Dynamik")
        model = _current_model()
        if model is None:
            st.info("Kein Modell geladen. Bitte trainieren oder laden.")
        else:
            meta = model.current_meta()
            replay_log = meta.get("replay_activity", [])
            if replay_log:
                plot_replay_activity(replay_log)
                st.markdown("### Replay-Kennzahlen")
                st.json(
                    {
                        "replay_efficiency": meta.get("replay_efficiency", 0.0),
                        "plasticity_balance": meta.get("plasticity_balance", {}),
                        "dopamine_avg": meta.get("dopamine_avg", 0.0),
                    },
                    expanded=False,
                )
            else:
                st.info("Noch keine Replay-Logs vorhanden.")

    # ── Tab: Modelle verwalten ────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("Modelle verwalten")
        model = _current_model()
        last_loaded = st.session_state.pop("biocortex_last_loaded", None)
        if last_loaded:
            st.success(f"Modell geladen und aktiviert: {last_loaded}.*")
        st.markdown("ℹ️ Lies die **Philosophie hinter BioCortex** im Tab **„Manifest“**.")
        col1, col2 = st.columns(2)
        with col1:
            save_path = st.text_input("Speicherpfad (ohne Endung)", key="save_path")
            if st.button("Modell speichern"):
                if model is None:
                    st.info("Kein Modell im Speicher.")
                elif not save_path.strip():
                    st.warning("Bitte einen Speicherpfad angeben.")
                else:
                    path = pathlib.Path(save_path).expanduser().resolve()
                    with st.spinner(f"Speichere Modell nach {path} …"):
                        model.save(str(path))
                    st.success(f"Modell gespeichert unter {path}.*")
        with col2:
            load_path = st.text_input("Ladepfad (ohne Endung)", key="load_path")
            if st.button("Modell laden"):
                if not load_path.strip():
                    st.warning("Bitte einen Pfad angeben.")
                else:
                    path = pathlib.Path(load_path).expanduser().resolve()
                    if not path.with_suffix(".cfg.json").exists():
                        st.error("Kein Modell an diesem Pfad gefunden.")
                    else:
                        with st.spinner(f"Lade Modell von {path} …"):
                            model = BioCortex.load(str(path))
                        _set_model(model)
                        st.session_state["biocortex_last_loaded"] = str(path)
                        st.rerun()

        if model is not None:
            st.markdown("### Aktuelle Modellkonfiguration")
            st.json(st.session_state.get("biocortex_config", {}))
            if model.training_meta:
                st.markdown("### Gespeicherte Metadaten")
                st.json(model.current_meta())

    # ── Tab: Manifest ─────────────────────────────────────────────────────────
    with tabs[5]:
        st.subheader("Manifest des Lebendigen Denkens")
        _display_logo(width=96)
        st.markdown(_read_manifest(), unsafe_allow_html=False)


if __name__ == "__main__":
    main()
