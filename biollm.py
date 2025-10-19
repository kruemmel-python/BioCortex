#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BioCortex – Trainierbares, rein bio‑inspiriertes Lernsystem (ohne Tensoren)
===========================================================================

• Ziel: Ein eigenständiges, *trainierbares* Lernsystem ohne TensorFlow/PyTorch.
• Prinzip: Rein *biologisch inspirierte* Mechaniken – keine Gradientenrückpropagation –
  sondern lokale Lernregeln: STDP‑ähnliche Updates, Pheromon‑/Neuromodulator‑Bias,
  Mycel‑Graph‑Plastizität, Hippocampus‑Replays.
• Fokus: Lehr‑/Forschungs‑Prototyp, transparent & erklärbar, erweiterbar.

Kernelemente
------------
1) BioBPE‑Tokenizer (Gen‑Fusion / k‑mer‑Analogon) – erklärbare Subword‑Tokenisierung.
2) Synapsen‑Graph über Tokens (gerichtete Übergänge):
   - **STDP‑ähnlich**: Δw = A_plus·pre·post − A_minus·decay; zeitliche Nachbarschaft zählt.
   - **Dopamin‑/Neuromodulator‑Gate** (Belohnung/Überraschung/Bindung) verstärkt/hemmt Updates.
   - **Pheromon‑Dynamik** (Ameisen): Verdunstung + Verstärkung häufig genutzter Pfade.
3) Sprachmodell‑Basis via **N‑Gramm Kneser‑Ney** (statistisch robust, tensorfrei).
4) **Kontext‑Mycel**: lokale Sequenz‑Nachbarschaft + inhaltliche Nähe verbinden Knoten.
5) **Hippocampus‑Replay**: stichprobenartige „Wiederholungen“ früherer Sequenzen zur
   Konsolidierung (Offline‑Training), analog zu Schlaf/Replay‑Theorien.

Training = Zählen (KN), STDP‑Updates (Graph), Pheromon‑Anpassungen, Replay‑Konsolidierung.
Generierung = KN‑Wahrscheinlichkeiten * Bio‑Bias (Pheromon/Neuromodulator) → Nucleus‑Sampling.

Persistenz: Tokenizer/Counts/Graph als JSON+NPZ.
CLI: train | finetune | generate | save | load (siehe __main__).

Abhängigkeiten: nur numpy
    pip install numpy

Lizenz: MIT – 2025‑10‑03 – Ralf Krümmel & Contributors
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Optional, Sequence, Iterable, Iterator, Any, Mapping
from collections import Counter, defaultdict, deque
import argparse
import csv
import json
import math
import random
import re
import time
import heapq
import os
import pathlib

from datetime import datetime

import numpy as np

from biollm_logging import setup_logging

logger = setup_logging("biocortex-train")

# ===============================================================
# 0) Utilities & Normalisierung
# ===============================================================

def normalize(text: str, *, lowercase: bool = True) -> str:
    if lowercase:
        text = text.lower()
    text = text.replace("\r", "\n").replace("\t", " ")
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ===============================================================
# 1) BioBPE – transparente Subword‑Tokenisierung (Gen‑Fusion)
# ===============================================================

@dataclass(slots=True)
class BioBPEConfig:
    vocab_size: int = 4000
    min_pair_freq: int = 2
    lowercase: bool = True


class BioBPE:
    def __init__(self, cfg: BioBPEConfig | None = None):
        self.cfg = cfg or BioBPEConfig()
        self.vocab: dict[str, int] = {}
        self.inv_vocab: list[str] = []
        self._sorted_vocab: list[str] = []
        self._sorted_vocab_dirty = True
        self.token_freq: Counter[int] = Counter()
        self._max_token_freq = 0

    def _refresh_sorted_vocab(self) -> None:
        self._sorted_vocab = sorted(self.vocab.keys(), key=len, reverse=True)
        self._sorted_vocab_dirty = False

    def reset_frequency(self) -> None:
        self.token_freq = Counter()
        self._max_token_freq = 0

    def _update_frequency(self, token_id: int) -> None:
        self.token_freq[token_id] += 1
        if self.token_freq[token_id] > self._max_token_freq:
            self._max_token_freq = self.token_freq[token_id]

    def observe_token(self, token_id: int) -> None:
        self._update_frequency(token_id)

    def fit(self, texts: Sequence[str]) -> None:
        corpus = "\n".join(normalize(t, lowercase=self.cfg.lowercase) for t in texts)
        symbols = sorted(set(corpus)) or [" "]
        self.inv_vocab = symbols.copy()
        self.vocab = {s: i for i, s in enumerate(self.inv_vocab)}
        self.reset_frequency()

        tokens = list(corpus)
        while len(self.inv_vocab) < self.cfg.vocab_size:
            # Häufigstes Paar finden
            pair_freq: Counter[tuple[str, str]] = Counter()
            for i in range(len(tokens) - 1):
                pair_freq[(tokens[i], tokens[i + 1])] += 1
            if not pair_freq:
                break
            (a, b), freq = pair_freq.most_common(1)[0]
            if freq < self.cfg.min_pair_freq:
                break
            merged = a + b
            if merged in self.vocab:
                break
            # Merge anwenden
            self.vocab[merged] = len(self.inv_vocab)
            self.inv_vocab.append(merged)
            i = 0
            new_tokens: list[str] = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        self._sorted_vocab_dirty = True
        self._refresh_sorted_vocab()

    def encode(self, text: str) -> list[int]:
        text = normalize(text, lowercase=self.cfg.lowercase)
        if not text:
            return []
        ids: list[int] = []
        if self._sorted_vocab_dirty or len(self._sorted_vocab) != len(self.vocab):
            self._refresh_sorted_vocab()
        vocab_sorted = self._sorted_vocab
        i = 0
        while i < len(text):
            for sym in vocab_sorted:
                if text.startswith(sym, i):
                    tok_id = self.vocab[sym]
                    ids.append(tok_id)
                    self._update_frequency(tok_id)
                    i += len(sym)
                    break
            else:
                ch = text[i]
                if ch not in self.vocab:
                    self.vocab[ch] = len(self.inv_vocab)
                    self.inv_vocab.append(ch)
                    self._sorted_vocab_dirty = True
                tok_id = self.vocab[ch]
                ids.append(tok_id)
                self._update_frequency(tok_id)
                i += 1
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.inv_vocab[i] for i in ids)

    def to_json(self) -> dict[str, Any]:
        return {"cfg": asdict(self.cfg), "inv_vocab": self.inv_vocab}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "BioBPE":
        obj = cls(BioBPEConfig(**data["cfg"]))
        obj.inv_vocab = list(data["inv_vocab"])
        obj.vocab = {s: i for i, s in enumerate(obj.inv_vocab)}
        obj._sorted_vocab_dirty = True
        obj._refresh_sorted_vocab()
        obj.reset_frequency()
        return obj

# ===============================================================
# 2) Kneser‑Ney N‑Gramm LM (Basis)
# ===============================================================

@dataclass(slots=True)
class KNConfig:
    order: int = 5
    discount: float = 0.75


class KneserNeyLM:
    def __init__(self, cfg: KNConfig):
        self.cfg = cfg
        self.ngram_counts: list[Counter[tuple[int, ...]]] = [Counter() for _ in range(cfg.order)]
        self.context_counts: list[Counter[tuple[int, ...]]] = [Counter() for _ in range(cfg.order)]
        self.continuation_counts: Counter[int] = Counter()
        # Beschleunigung: speichere Folgetoken pro Kontext und globales Vokabular
        self.successors: list[dict[tuple[int, ...], tuple[int, ...]]] = [dict() for _ in range(cfg.order)]
        self.vocab: set[int] = set()
        self.BOS = -1
        self.EOS = -2

    def train_sequences(self, sequences: Sequence[list[int]]) -> None:
        follower_sets: list[defaultdict[tuple[int, ...], set[int]]] = [defaultdict(set) for _ in range(self.cfg.order)]
        for seq in sequences:
            padded = [self.BOS] * (self.cfg.order - 1) + seq + [self.EOS]
            self.vocab.update(seq)
            for n in range(1, self.cfg.order + 1):
                for i in range(len(padded) - n + 1):
                    ng = tuple(padded[i : i + n])
                    self.ngram_counts[n - 1][ng] += 1
                    if n > 1:
                        ctx = ng[:-1]
                        tok = ng[-1]
                        self.context_counts[n - 1][ctx] += 1
                        follower_sets[n - 1][ctx].add(tok)
        # Continuation‑Basis
        seen_pred: dict[int, set[tuple[int, ...]]] = defaultdict(set)
        for n in range(2, self.cfg.order + 1):
            for ng in self.ngram_counts[n - 1].keys():
                w = ng[-1]
                seen_pred[w].add(ng[:-1])
        for w, preds in seen_pred.items():
            self.continuation_counts[w] = len(preds)
        self._assign_successors(follower_sets)
        self._ensure_vocab()

    def _assign_successors(self, follower_sets: Sequence[defaultdict[tuple[int, ...], set[int]]]) -> None:
        self.successors = [dict() for _ in range(self.cfg.order)]
        for n in range(1, self.cfg.order):
            if follower_sets[n]:
                self.successors[n] = {
                    ctx: tuple(sorted(tokens)) for ctx, tokens in follower_sets[n].items()
                }

    def _ensure_vocab(self) -> None:
        if not self.vocab:
            vocab = set()
            for counts in self.ngram_counts:
                for ng in counts:
                    vocab.update(ng)
            vocab.discard(self.BOS)
            self.vocab = vocab
        self.vocab.add(self.EOS)

    def p_cont(self, w: int) -> float:
        Z = sum(self.continuation_counts.values()) or 1.0
        return self.continuation_counts[w] / Z

    def prob_next(self, context: Sequence[int], candidates: Optional[Sequence[int]] = None) -> dict[int, float]:
        h = tuple(context[-(self.cfg.order - 1):])
        if candidates is None:
            candidates = list(self._candidates_for(h))
        D = self.cfg.discount
        counts = self.ngram_counts
        ctx_counts = self.context_counts

        def p_kn(w: int, h: tuple[int, ...], n: int) -> float:
            if n == 1:
                return self.p_cont(w)
            c_hw = counts[n - 1].get(h + (w,), 0)
            c_h  = ctx_counts[n - 1].get(h, 0)
            if c_h > 0:
                types_after = len(self.successors[n - 1].get(h, ()))
                p_ml = max(c_hw - D, 0.0) / c_h
                lam = (D * types_after) / c_h if c_h else 0.0
                return p_ml + lam * p_kn(w, h[1:], n - 1)
            else:
                return p_kn(w, h[1:], n - 1)

        raw = {w: p_kn(w, h, self.cfg.order) for w in candidates}
        Z = sum(raw.values()) or 1e-12
        return {w: p / Z for w, p in raw.items()}

    def _candidates_for(self, context: tuple[int, ...]) -> set[int]:
        for order_minus1 in range(len(context), 0, -1):
            ctx = context[-order_minus1:]
            succ = self.successors[order_minus1].get(ctx)
            if succ:
                return set(succ)
        return set(self.vocab) or {self.EOS}

    def to_npz(self, path: str) -> None:
        # Speichere nur kleine Projektionen (zählstarke Strukturen sind dicts → JSON)
        data = {
            "order": self.cfg.order,
            "discount": self.cfg.discount,
        }
        with open(path + ".kn.json", "w", encoding="utf-8") as f:
            json.dump({
                "cfg": data,
                "ngram_counts": [{"|".join(map(str, k)): v for k, v in c.items()} for c in self.ngram_counts],
                "context_counts": [{"|".join(map(str, k)): v for k, v in c.items()} for c in self.context_counts],
                "continuation_counts": {str(k): v for k, v in self.continuation_counts.items()},
                "successors": [
                    {"|".join(map(str, k)): list(map(int, vals)) for k, vals in succ.items()}
                    for succ in self.successors
                ],
                "vocab": sorted(int(x) for x in self.vocab),
            }, f)

    @classmethod
    def from_npz(cls, path: str) -> "KneserNeyLM":
        with open(path + ".kn.json", "r", encoding="utf-8") as f:
            raw = json.load(f)
        obj = cls(KNConfig(order=raw["cfg"]["order"], discount=raw["cfg"]["discount"]))
        def parse_key(s: str) -> tuple[int, ...]:
            return tuple(int(x) for x in s.split("|")) if s else tuple()
        obj.ngram_counts = [Counter({parse_key(k): v for k, v in c.items()}) for c in raw["ngram_counts"]]
        obj.context_counts = [Counter({parse_key(k): v for k, v in c.items()}) for c in raw["context_counts"]]
        obj.continuation_counts = Counter({int(k): v for k, v in raw["continuation_counts"].items()})
        successors_raw = raw.get("successors")
        if successors_raw:
            obj.successors = [
                {parse_key(k): tuple(int(x) for x in vals) for k, vals in succ.items()}
                for succ in successors_raw
            ]
            if len(obj.successors) > obj.cfg.order:
                obj.successors = obj.successors[: obj.cfg.order]
            while len(obj.successors) < obj.cfg.order:
                obj.successors.append({})
        else:
            obj._recompute_successors()
        obj.vocab = set(raw.get("vocab", []))
        obj._ensure_vocab()
        return obj

    def _recompute_successors(self) -> None:
        follower_sets: list[defaultdict[tuple[int, ...], set[int]]] = [defaultdict(set) for _ in range(self.cfg.order)]
        for n in range(2, self.cfg.order + 1):
            for ng in self.ngram_counts[n - 1].keys():
                ctx = ng[:-1]
                follower_sets[n - 1][ctx].add(ng[-1])
        self._assign_successors(follower_sets)

# ===============================================================
# 3) Bio‑Synapsen‑Graph: STDP + Pheromon + Neuromodulatoren
# ===============================================================

@dataclass(slots=True)
class PlasticityConfig:
    a_plus: float = 0.05     # LTP‑Anteil (pre→post zeitnah)
    a_minus: float = 0.02    # LTD‑Anteil (Decay/Strafterm)
    tau: float = 3.0         # Zeitkonstante (Steps) für Nachbarschaftswirkung
    dopamine_gain: float = 0.6  # Verstärkung bei Belohnung
    surprise_gain: float = 0.3  # Verstärkung bei Überraschung
    bond_gain: float = 0.2      # Verstärkung bei Kohärenz/Bindung (z. B. Klammern/Quotes)
    pher_evaporation: float = 0.05
    pher_reinforce: float = 0.15


class BioGraph:
    """Gerichteter Übergangsgraph über Token‑IDs mit Gewichten und Pheromonen."""
    def __init__(self, vocab_size_hint: int = 4096) -> None:
        self.w: dict[tuple[int, int], float] = defaultdict(float)   # Synapsengewicht
        self.pher: dict[tuple[int, int], float] = defaultdict(float) # Pheromonspur
        self.out_sum: Counter[int] = Counter()

    def evaporate(self, rho: float) -> None:
        for k in list(self.pher.keys()):
            self.pher[k] *= (1.0 - rho)
            if self.pher[k] < 1e-12:
                del self.pher[k]

    def reinforce_pher(self, seq: Sequence[int], amount: float) -> None:
        for a, b in zip(seq, seq[1:]):
            if a < 0 or b < 0:
                continue
            self.pher[(a, b)] += amount

    def stdp_update(
        self, seq: Sequence[int], pcfg: PlasticityConfig, modulators: dict[str, float]
    ) -> tuple[float, float]:
        """Ein einfacher STDP‑ähnlicher Local‑Learner über ein Sequenzfenster.
        pre = seq[i], post in {i+1 .. i+tau} → LTP; ferner LTD‑Leak.
        Neuromodulatoren (dopamine/surprise/bond) skalieren den LTP‑Term.
        """
        A_plus = pcfg.a_plus * (1.0 + pcfg.dopamine_gain*modulators.get("dopamine",0.0)
                                       + pcfg.surprise_gain*modulators.get("surprise",0.0)
                                       + pcfg.bond_gain*modulators.get("bond",0.0))
        A_minus = pcfg.a_minus
        tau = pcfg.tau
        # LTD: globaler kleiner Leak
        ltd_total = 0.0
        decay_rate = A_minus * 0.01
        if decay_rate > 0.0:
            for k in list(self.w.keys()):
                decay = self.w[k] * decay_rate
                self.w[k] -= decay
                ltd_total += abs(decay)
                if abs(self.w[k]) < 1e-12:
                    del self.w[k]
        # LTP: lokale Nachbarschaft
        ltp_total = 0.0
        for i, pre in enumerate(seq):
            for j in range(i+1, min(len(seq), i + 1 + int(tau))):
                post = seq[j]
                dt = j - i
                gain = A_plus * math.exp(- dt / tau)
                self.w[(pre, post)] += gain
                self.out_sum[pre] += gain
                ltp_total += gain
        return ltp_total, ltd_total

    def bias_next(self, last_token: Optional[int]) -> dict[int, float]:
        if last_token is None:
            return {}
        # Kombiniere synaptische Gewichte und Pheromonspuren als Bias
        cand: dict[int, float] = defaultdict(float)
        # Synaptik
        for (a, b), v in self.w.items():
            if a == last_token and v > 0:
                cand[b] += v
        # Pheromon
        ph_total = sum(v for (a, _), v in self.pher.items() if a == last_token)
        if ph_total > 0:
            for (a, b), v in self.pher.items():
                if a == last_token:
                    cand[b] += v / ph_total
        # Norm
        Z = sum(cand.values())
        if Z <= 0:
            return {}
        return {k: v / Z for k, v in cand.items()}

# ===============================================================
# 4) Hippocampus‑Replay & Modulatoren‑Heuristik
# ===============================================================

@dataclass(slots=True)
class ReplayConfig:
    buffer_size: int = 2048
    sample_len: int = 64
    nightly_samples: int = 64


class ReplayBuffer:
    def __init__(self, cfg: ReplayConfig):
        self.cfg = cfg
        self.buf: deque[list[int]] = deque(maxlen=cfg.buffer_size)

    def add(self, seq: list[int]) -> None:
        if seq:
            self.buf.append(seq[: self.cfg.sample_len])

    def samples(self) -> Iterable[list[int]]:
        if not self.buf:
            return []
        k = min(len(self.buf), self.cfg.nightly_samples)
        return [s[: self.cfg.sample_len] for s in random.sample(list(self.buf), k)]


def modulator_signals(
    tokens: Sequence[int],
    inv_vocab: list[str],
    freq_map: Mapping[int, int] | None = None,
) -> dict[str, float]:
    """Grobe Heuristiken:
    - dopamine: End‑of‑sentence/Punktuation → Belohnung
    - surprise: seltene Symbole (niedrige Frequenz) → Überraschung
    - bond: Klammern/Quotes/Paarigkeit → Bindung/Kohärenz
    """
    if not tokens:
        return {"dopamine": 0.0, "surprise": 0.0, "bond": 0.0}
    txt = "".join(inv_vocab[t] for t in tokens if t >= 0)
    dopamine = 1.0 if re.search(r"[.!?]", txt[-3:]) else 0.0
    freq_map = freq_map or {}
    if freq_map:
        max_freq = max(freq_map.values()) or 1
    else:
        max_freq = 1
    surprise_window = tokens[-5:]
    surprise_scores: list[float] = []
    for tok in surprise_window:
        freq = freq_map.get(tok, 0)
        surprise_scores.append(1.0 - min(freq / max_freq, 1.0))
    surprise = float(np.mean(surprise_scores)) if surprise_scores else 0.0
    # Paarigkeit
    opens = txt.count("(") + txt.count("[") + txt.count("{")
    closes = txt.count(")") + txt.count("]") + txt.count("}")
    quotes = txt.count("\"")
    bond = 1.0 if abs(opens - closes) <= 1 and quotes % 2 == 0 else 0.0
    return {"dopamine": dopamine, "surprise": surprise, "bond": bond}

# ===============================================================
# 5) Orchestrierung: BioCortex – Training / Inferenz / Persistenz
# ===============================================================

@dataclass(slots=True)
class BioLLMConfig:
    bpe: BioBPEConfig = field(default_factory=BioBPEConfig)
    kn: KNConfig = field(default_factory=KNConfig)
    plastic: PlasticityConfig = field(default_factory=PlasticityConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    max_gen_len: int = 128
    top_p: float = 0.9
    gamma_bias: float = 1.4  # Stärke des Bio‑Bias in p' = p_kn * (1 + gamma * bias)


ProgressCallback = Callable[[str, int, int, str], None]


@dataclass(slots=True)
class GenerationTrace:
    text: str
    tokens: list[int]
    mod_history: list[dict[str, float]]


class BioCortex:
    def __init__(self, cfg: BioLLMConfig | None = None, *, rng_seed: Optional[int] = 17):
        self.cfg = cfg or BioLLMConfig()
        if rng_seed is not None:
            random.seed(rng_seed)
            np.random.seed(rng_seed)
        self.tokenizer = BioBPE(self.cfg.bpe)
        self.kn = KneserNeyLM(self.cfg.kn)
        self.graph = BioGraph()
        self.replay = ReplayBuffer(self.cfg.replay)
        self.training_meta: dict[str, Any] = {}
        self.mod_history: deque[dict[str, float]] = deque(maxlen=512)
        self.plasticity_totals = {"ltp": 0.0, "ltd": 0.0}
        self.replay_activity: list[dict[str, float]] = []
        self._activity_step = 0
        self.run_id = getattr(logger, "run_id", "training")
        self._metrics_path = pathlib.Path("logs") / "metrics.csv"

    # -------------------------- Training -------------------------
    def fit(self, texts: Sequence[str], progress: ProgressCallback | None = None) -> None:
        def notify(stage: str, step: int, total: int, detail: str = "") -> None:
            if progress is not None:
                progress(stage, step, total, detail)

        start_ts = time.time()
        mod_totals = Counter()
        mod_count = 0
        # 1) Tokenizer/Sequences
        buffer_len_before = len(self.replay.buf)
        seq_count = len(texts)
        est_buf_after = min(self.cfg.replay.buffer_size, buffer_len_before + seq_count)
        replay_steps = min(est_buf_after, self.cfg.replay.nightly_samples)
        total_steps = max(1, 2 + seq_count + replay_steps)

        step = 0
        notify("Tokenizer", step, total_steps, "Initialisiere Tokenizer")
        logger.info("Training gestartet: %d Texte", len(texts))
        self.tokenizer.fit(texts)
        step += 1
        notify("Tokenizer", step, total_steps, "Tokenizer abgeschlossen")
        logger.info("Tokenisierung abgeschlossen – Vokabulargröße: %d", len(self.tokenizer.vocab))
        sequences = [self.tokenizer.encode(t) for t in texts]
        # 2) KN zählen
        notify("Kneser-Ney", step, total_steps, "Trainiere N-Gramm-Statistiken")
        self.kn.train_sequences(sequences)
        step += 1
        notify("Kneser-Ney", step, total_steps, "N-Gramm-Training abgeschlossen")
        logger.info(
            "Kneser-Ney trainiert (Order=%d, Discount=%.3f)",
            self.cfg.kn.order,
            self.cfg.kn.discount,
        )
        # 3) Bio‑Graph lernen (STDP + Pheromon), inkl. Replay
        for idx, seq in enumerate(sequences, start=1):
            mods, ltp_total, ltd_total = self._learn_sequence(seq)
            if mods:
                mod_totals.update(mods)
                mod_count += 1
            self.replay.add(seq)
            step += 1
            notify(
                "Bio-Graph",
                step,
                total_steps,
                f"Sequenz {step - 2}/{seq_count} verarbeitet",
            )
            if idx % 100 == 0 or idx == len(sequences):
                logger.info("Sequenz %d/%d trainiert", idx, len(sequences))
            self._log_activity("sequence", mods, ltp_total, ltd_total)
        # Konsolidierung (Replay‑Durchgänge)
        replays = list(self.replay.samples())
        if replays:
            logger.info("Replay gestartet (%d Sequenzen)", len(replays))
        else:
            logger.info("Replay übersprungen – keine Samples verfügbar")
        for idx, rep in enumerate(replays, start=1):
            mods, ltp_total, ltd_total = self._learn_sequence(rep)
            if mods:
                mod_totals.update(mods)
                mod_count += 1
            step += 1
            notify("Replay", step, total_steps, f"Replay {idx}/{len(replays)} konsolidiert")
            if idx % 10 == 0 or idx == len(replays):
                logger.info("Replay %d/%d konsolidiert", idx, len(replays))
            self._log_activity("replay", mods, ltp_total, ltd_total)
        duration = time.time() - start_ts
        self._update_training_meta(
            num_sequences=len(sequences),
            replay_cycles=len(replays),
            duration=duration,
            mod_totals=mod_totals,
            mod_count=mod_count,
            incremental=False,
        )
        logger.info(
            "Training abgeschlossen – Dauer %.2fs, Sequenzen=%d, Replays=%d",
            duration,
            len(sequences),
            len(replays),
        )

    def partial_fit(self, texts: Sequence[str], progress: ProgressCallback | None = None) -> None:
        def notify(stage: str, step: int, total: int, detail: str = "") -> None:
            if progress is not None:
                progress(stage, step, total, detail)

        sequences = [self.tokenizer.encode(t) for t in texts]
        buffer_len_before = len(self.replay.buf)
        seq_count = len(sequences)
        est_buf_after = min(self.cfg.replay.buffer_size, buffer_len_before + seq_count)
        replay_steps = min(est_buf_after, self.cfg.replay.nightly_samples)
        total_steps = max(1, 1 + seq_count + replay_steps)
        step = 0
        start_ts = time.time()
        mod_totals = Counter()
        mod_count = 0
        logger.info("Feintraining gestartet: %d Sequenzen", len(sequences))
        notify("Kneser-Ney", step, total_steps, "Aktualisiere N-Gramm-Statistiken")
        self.kn.train_sequences(sequences)
        step += 1
        notify("Kneser-Ney", step, total_steps, "N-Gramm-Update abgeschlossen")
        logger.info(
            "Kneser-Ney aktualisiert (Order=%d, Discount=%.3f)",
            self.cfg.kn.order,
            self.cfg.kn.discount,
        )
        for idx, seq in enumerate(sequences, start=1):
            mods, ltp_total, ltd_total = self._learn_sequence(seq)
            if mods:
                mod_totals.update(mods)
                mod_count += 1
            self.replay.add(seq)
            step += 1
            notify(
                "Bio-Graph",
                step,
                total_steps,
                f"Sequenz {step - 1}/{seq_count} verarbeitet",
            )
            if idx % 100 == 0 or idx == len(sequences):
                logger.info("Feintraining: Sequenz %d/%d verarbeitet", idx, len(sequences))
            self._log_activity("sequence", mods, ltp_total, ltd_total)
        replays = list(self.replay.samples())
        if replays:
            logger.info("Feintraining: Replay gestartet (%d Sequenzen)", len(replays))
        else:
            logger.info("Feintraining: Kein Replay ausgeführt")
        for idx, rep in enumerate(replays, start=1):
            mods, ltp_total, ltd_total = self._learn_sequence(rep)
            if mods:
                mod_totals.update(mods)
                mod_count += 1
            step += 1
            notify("Replay", step, total_steps, f"Replay {idx}/{len(replays)} konsolidiert")
            if idx % 10 == 0 or idx == len(replays):
                logger.info("Feintraining: Replay %d/%d konsolidiert", idx, len(replays))
            self._log_activity("replay", mods, ltp_total, ltd_total)
        duration = time.time() - start_ts
        self._update_training_meta(
            num_sequences=len(sequences),
            replay_cycles=len(replays),
            duration=duration,
            mod_totals=mod_totals,
            mod_count=mod_count,
            incremental=True,
        )
        logger.info(
            "Feintraining abgeschlossen – Dauer %.2fs, Sequenzen=%d, Replays=%d",
            duration,
            len(sequences),
            len(replays),
        )

    def _learn_sequence(self, seq: list[int]) -> tuple[dict[str, float], float, float]:
        # Pheromon: verstärke real gelaufene Pfade
        self.graph.reinforce_pher(seq, amount=self.cfg.plastic.pher_reinforce)
        self.graph.evaporate(self.cfg.plastic.pher_evaporation)
        # STDP mit Modulatoren
        mods = modulator_signals(seq, self.tokenizer.inv_vocab, self.tokenizer.token_freq)
        ltp_total, ltd_total = self.graph.stdp_update(seq, self.cfg.plastic, mods)
        self.mod_history.append(mods)
        self.plasticity_totals["ltp"] += float(ltp_total)
        self.plasticity_totals["ltd"] += float(ltd_total)
        self._adapt_parameters()
        return mods, float(ltp_total), float(ltd_total)

    # ------------------------- Inferenz --------------------------
    def next_distribution(self, context_ids: Sequence[int]) -> dict[int, float]:
        base = self.kn.prob_next(context_ids)
        bias = self.graph.bias_next(context_ids[-1] if context_ids else None)
        if not bias:
            return base
        gamma = self.cfg.gamma_bias
        fused: dict[int, float] = {}
        for w, p in base.items():
            fused[w] = p * (1.0 + gamma * bias.get(w, 0.0))
        Z = sum(fused.values()) or 1e-12
        for w in fused:
            fused[w] /= Z
        return fused

    def _sample_top_p(self, probs: dict[int, float], top_p: float) -> int:
        items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        cum = 0.0
        nucleus: list[tuple[int, float]] = []
        for w, p in items:
            nucleus.append((w, p))
            cum += p
            if cum >= top_p:
                break
        r = random.random() * cum
        s = 0.0
        for w, p in nucleus:
            s += p
            if s >= r:
                return w
        return nucleus[-1][0]

    def generate(self, prompt: str, *, max_new_tokens: Optional[int] = None, top_p: Optional[float] = None) -> str:
        trace = self.generate_with_trace(prompt, max_new_tokens=max_new_tokens, top_p=top_p)
        return trace.text

    def generate_with_trace(
        self,
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> "GenerationTrace":
        max_new = max_new_tokens or self.cfg.max_gen_len
        snippet = prompt.replace("\n", " ")
        if len(snippet) > 40:
            snippet = snippet[:37] + "..."
        logger.info(
            "Generierung gestartet: prompt='%s' (max_tokens=%d, top_p=%.2f)",
            snippet,
            max_new,
            top_p if top_p is not None else self.cfg.top_p,
        )
        ctx = self.tokenizer.encode(prompt)
        out = ctx.copy()
        mod_history: list[dict[str, float]] = []
        self._adapt_parameters()
        for _ in range(max_new):
            effective_top_p = float(top_p) if top_p is not None else self.cfg.top_p
            probs = self.next_distribution(out)
            w = self._sample_top_p(probs, effective_top_p)
            if w == self.kn.EOS:
                break
            out.append(w)
            self.tokenizer.observe_token(w)
            # Online‑Mikro‑Plastizität beim Erzeugen
            self.graph.reinforce_pher(out[-2:], amount=self.cfg.plastic.pher_reinforce * 0.05)
            self.graph.evaporate(self.cfg.plastic.pher_evaporation * 0.02)
            mods = modulator_signals(out, self.tokenizer.inv_vocab, self.tokenizer.token_freq)
            mod_history.append(mods)
            self.mod_history.append(mods)
            self._adapt_parameters()
        generated = out[len(ctx) :]
        dopamine_avg = float(np.mean([m.get("dopamine", 0.0) for m in mod_history])) if mod_history else 0.0
        logger.info(
            "Generierung beendet – %d neue Tokens, mean_dopamine=%.4f", len(generated), dopamine_avg
        )
        return GenerationTrace(text=self.tokenizer.decode(out), tokens=out, mod_history=mod_history)

    # ------------------------ Persistenz -------------------------
    def save(self, path: str) -> None:
        path = str(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Tokenizer
        with open(path + ".bpe.json", "w", encoding="utf-8") as f:
            json.dump(self.tokenizer.to_json(), f)
        # KN
        self.kn.to_npz(path)
        # Graph
        with open(path + ".graph.json", "w", encoding="utf-8") as f:
            json.dump({
                "w": {f"{a}|{b}": v for (a, b), v in self.graph.w.items()},
                "pher": {f"{a}|{b}": v for (a, b), v in self.graph.pher.items()},
            }, f)
        meta = self.current_meta()
        with open(path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)
        # Konfig
        with open(path + ".cfg.json", "w", encoding="utf-8") as f:
            json.dump({
                "bpe": asdict(self.cfg.bpe),
                "kn": asdict(self.cfg.kn),
                "plastic": asdict(self.cfg.plastic),
                "replay": asdict(self.cfg.replay),
                "max_gen_len": self.cfg.max_gen_len,
                "top_p": self.cfg.top_p,
                "gamma_bias": self.cfg.gamma_bias,
            }, f)
        logger.info("Modell gespeichert unter: %s.*", path)

    @classmethod
    def load(cls, path: str) -> "BioCortex":
        with open(path + ".cfg.json", "r", encoding="utf-8") as f:
            raw = json.load(f)
        cfg = BioLLMConfig(
            bpe=BioBPEConfig(**raw["bpe"]),
            kn=KNConfig(**raw["kn"]),
            plastic=PlasticityConfig(**raw["plastic"]),
            replay=ReplayConfig(**raw["replay"]),
            max_gen_len=raw["max_gen_len"],
            top_p=raw["top_p"],
            gamma_bias=raw["gamma_bias"],
        )
        obj = cls(cfg)
        # Tokenizer
        with open(path + ".bpe.json", "r", encoding="utf-8") as f:
            obj.tokenizer = BioBPE.from_json(json.load(f))
        # KN
        obj.kn = KneserNeyLM.from_npz(path)
        # Graph
        with open(path + ".graph.json", "r", encoding="utf-8") as f:
            g = json.load(f)
        def parse_k(k: str) -> tuple[int, int]:
            a, b = k.split("|")
            return (int(a), int(b))
        obj.graph.w = defaultdict(float, {parse_k(k): v for k, v in g["w"].items()})
        obj.graph.pher = defaultdict(float, {parse_k(k): v for k, v in g["pher"].items()})
        # Replay leer lassen – wird im weiteren Training gefüllt
        obj.replay = ReplayBuffer(obj.cfg.replay)
        meta_path = pathlib.Path(path + ".meta.json")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                obj.training_meta = json.load(f)
        else:
            obj.training_meta = obj.current_meta()
        obj.replay_activity = list(obj.training_meta.get("replay_activity", []))
        obj._activity_step = len(obj.replay_activity)
        logger.info("Modell geladen von: %s.*", path)
        return obj

    def current_meta(self) -> dict[str, Any]:
        meta = dict(self.training_meta)
        pher_values = list(self.graph.pher.values())
        weight_values = list(self.graph.w.values())
        meta.setdefault("mean_pher", float(np.mean(pher_values)) if pher_values else 0.0)
        meta.setdefault("mean_weight", float(np.mean(weight_values)) if weight_values else 0.0)
        if "training_time" not in meta:
            meta["training_time"] = datetime.utcnow().isoformat()
        meta.setdefault("engine", "BioCortex")
        meta.setdefault("version", "1.0")
        meta.setdefault("num_sequences", 0)
        meta.setdefault("replay_cycles", 0)
        meta.setdefault("dopamine_level", 0.0)
        meta.setdefault("elapsed_seconds", 0.0)
        meta.setdefault("replay_efficiency", 0.0)
        meta.setdefault("plasticity_balance", {"ltp": 0.0, "ltd": 0.0})
        meta.setdefault("graph_density", 0.0)
        meta.setdefault("dopamine_avg", meta.get("dopamine_level", 0.0))
        meta.setdefault("replay_activity", [])
        return meta

    def _update_training_meta(
        self,
        *,
        num_sequences: int,
        replay_cycles: int,
        duration: float,
        mod_totals: Counter,
        mod_count: int,
        incremental: bool,
    ) -> None:
        base = self.training_meta.copy() if incremental else {}
        timestamp = datetime.utcnow().isoformat()
        base["training_time"] = timestamp
        base["elapsed_seconds"] = round((base.get("elapsed_seconds", 0.0) if incremental else 0.0) + duration, 3)
        base["num_sequences"] = (base.get("num_sequences", 0) if incremental else 0) + num_sequences
        base["replay_cycles"] = (base.get("replay_cycles", 0) if incremental else 0) + replay_cycles
        mean_dopa = 0.0
        if mod_count > 0:
            mean_dopa = float(mod_totals.get("dopamine", 0.0)) / mod_count
        base["dopamine_level"] = round(mean_dopa, 4)
        pher_values = list(self.graph.pher.values())
        weight_values = list(self.graph.w.values())
        base["mean_pher"] = float(np.mean(pher_values)) if pher_values else 0.0
        base["mean_weight"] = float(np.mean(weight_values)) if weight_values else 0.0
        nightly = max(1, self.cfg.replay.nightly_samples)
        base["replay_efficiency"] = round(replay_cycles / nightly, 4) if nightly else 0.0
        total_plastic = self.plasticity_totals["ltp"] + self.plasticity_totals["ltd"]
        if total_plastic > 0:
            base["plasticity_balance"] = {
                "ltp": round(self.plasticity_totals["ltp"] / total_plastic, 4),
                "ltd": round(self.plasticity_totals["ltd"] / total_plastic, 4),
            }
        else:
            base["plasticity_balance"] = {"ltp": 0.0, "ltd": 0.0}
        base["graph_density"] = round(self._graph_density(), 6)
        base["dopamine_avg"] = base.get("dopamine_level", 0.0)
        if self.replay_activity:
            base["replay_activity"] = self.replay_activity[-500:]
        self.training_meta = base
        logger.info(
            "Training-Metriken aktualisiert (%s): sequences=%d, replays=%d, mean_pher=%.4f, mean_weight=%.4f, dopamine=%.4f",
            "inkrementell" if incremental else "voll",
            base.get("num_sequences", 0),
            base.get("replay_cycles", 0),
            base.get("mean_pher", 0.0),
            base.get("mean_weight", 0.0),
            base.get("dopamine_level", 0.0),
        )

    def _log_activity(self, kind: str, mods: dict[str, float] | None, ltp: float, ltd: float) -> None:
        mods = mods or {}
        entry = {
            "step": self._activity_step,
            "replay": 1.0 if kind == "replay" else 0.0,
            "ltp": float(ltp),
            "ltd": float(ltd),
            "dopamine": float(mods.get("dopamine", 0.0)),
            "surprise": float(mods.get("surprise", 0.0)),
            "bond": float(mods.get("bond", 0.0)),
        }
        self._activity_step += 1
        self.replay_activity.append(entry)
        if len(self.replay_activity) > 2048:
            self.replay_activity = self.replay_activity[-2048:]
        self._write_metrics_row(entry)

    def _write_metrics_row(self, entry: dict[str, float]) -> None:
        try:
            self._metrics_path.parent.mkdir(exist_ok=True)
            if self._metrics_path.exists() and self._metrics_path.stat().st_size > 5 * 1024 * 1024:
                rotated = self._metrics_path.with_name(
                    f"metrics_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
                )
                self._metrics_path.rename(rotated)
            pher_values = list(self.graph.pher.values())
            weight_values = list(self.graph.w.values())
            mean_pher = float(np.mean(pher_values)) if pher_values else 0.0
            mean_weight = float(np.mean(weight_values)) if weight_values else 0.0
            row = [
                self.run_id,
                int(entry["step"]),
                f"{mean_pher:.6f}",
                f"{mean_weight:.6f}",
                f"{entry.get('dopamine', 0.0):.6f}",
                f"{entry.get('surprise', 0.0):.6f}",
                f"{entry.get('bond', 0.0):.6f}",
                f"{entry.get('replay', 0.0):.0f}",
                f"{entry.get('ltp', 0.0):.6f}",
                f"{entry.get('ltd', 0.0):.6f}",
            ]
            with open(self._metrics_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(
                        [
                            "run",
                            "step",
                            "mean_pher",
                            "mean_weight",
                            "dopamine",
                            "surprise",
                            "bond",
                            "replay_flag",
                            "ltp",
                            "ltd",
                        ]
                    )
                writer.writerow(row)
        except (OSError, csv.Error):
            logger.warning("Konnte Metrics-CSV nicht schreiben", exc_info=True)

    def _graph_density(self) -> float:
        edges = set(self.graph.w.keys()) | set(self.graph.pher.keys())
        if not edges:
            return 0.0
        nodes: set[int] = set()
        for a, b in edges:
            nodes.add(a)
            nodes.add(b)
        n = len(nodes)
        if n <= 1:
            return 0.0
        return float(len(edges) / (n * (n - 1)))

    def _adapt_parameters(self) -> None:
        pher_values = list(self.graph.pher.values())
        pher_mean = float(np.mean(pher_values)) if pher_values else 0.0
        window = list(self.mod_history)[-5:]
        if window:
            dopamine = float(np.mean([m.get("dopamine", 0.0) for m in window]))
        else:
            dopamine = 0.0
        self.cfg.top_p = float(min(1.0, 0.7 + dopamine * 0.3))
        discount = float(max(0.5, 0.8 - pher_mean * 0.2))
        self.cfg.kn.discount = discount
        self.kn.cfg.discount = discount


# Alias für Rückwärtskompatibilität
BioLLM = BioCortex

# ===============================================================
# 6) CLI
# ===============================================================

def _read_all_texts(paths: Sequence[str]) -> list[str]:
    acc: list[str] = []
    for p in paths:
        pth = pathlib.Path(p)
        if pth.is_dir():
            for fp in pth.rglob("*.txt"):
                acc.append(fp.read_text(encoding="utf-8", errors="ignore"))
        else:
            acc.append(pth.read_text(encoding="utf-8", errors="ignore"))
    return acc


def main() -> None:
    ap = argparse.ArgumentParser(description="BioCortex – bio‑inspiriertes Lernsystem ohne Tensoren")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_train = sub.add_parser("train", help="Neues Modell trainieren")
    ap_train.add_argument("--data", nargs="+", required=True, help="TXT‑Dateien oder Ordner")
    ap_train.add_argument("--out", required=True, help="Basispfad zum Speichern (ohne Endung)")

    ap_ft = sub.add_parser("finetune", help="Weitertrainieren eines Modells")
    ap_ft.add_argument("--model", required=True, help="Basispfad eines gespeicherten Modells")
    ap_ft.add_argument("--data", nargs="+", required=True)

    ap_gen = sub.add_parser("generate", help="Text generieren")
    ap_gen.add_argument("--model", required=True)
    ap_gen.add_argument("--prompt", required=True)
    ap_gen.add_argument("--max_new", type=int, default=128)
    ap_gen.add_argument("--top_p", type=float, default=0.9)

    args = ap.parse_args()

    match args.cmd:
        case "train":
            texts = _read_all_texts(args.data)
            logger.info("CLI: Starte Training mit %d Texten (Ausgabe: %s)", len(texts), args.out)
            model = BioCortex()
            model.fit(texts)
            model.save(args.out)
            print(f"[OK] Modell gespeichert unter {args.out}.*")
        case "finetune":
            texts = _read_all_texts(args.data)
            logger.info(
                "CLI: Starte Feintraining für %s mit %d Texten", args.model, len(texts)
            )
            model = BioCortex.load(args.model)
            model.partial_fit(texts)
            model.save(args.model)
            print(f"[OK] Modell aktualisiert: {args.model}.*")
        case "generate":
            logger.info("CLI: Generierung für Modell %s", args.model)
            model = BioCortex.load(args.model)
            txt = model.generate(args.prompt, max_new_tokens=args.max_new, top_p=args.top_p)
            print(txt)
        case _:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
