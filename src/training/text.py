from __future__ import annotations

import re

import regex


WS_RE = re.compile(r"\s+")
TIE_BARS = {"\u035c", "\u0361"}


def graphemes(text: str) -> list[str]:
    # Unicode extended grapheme segmentation groups the tie bar with the first phone but leaves the second phone separate ("d͡", "ʒ").  
    # In IPA the complete tied sequence is one unit and must never receive an internal word boundary, so join a following cluster when the previous one ends in COMBINING DOUBLE BREVE BELOW or COMBINING DOUBLE INVERTED BREVE.
    raw = regex.findall(r"\X", text)
    merged: list[str] = []
    index = 0
    while index < len(raw):
        cluster = raw[index]
        if cluster and cluster[-1] in TIE_BARS and index + 1 < len(raw):
            cluster += raw[index + 1]
            index += 1
        merged.append(cluster)
        index += 1
    return merged


def space_free(text: str) -> str:
    return WS_RE.sub("", text)


def gold_clusters_and_labels(gold: str) -> tuple[list[str], list[int]]:
    words = [word for word in WS_RE.split(gold.strip()) if word]
    clusters: list[str] = []
    labels: list[int] = []
    for word_index, word in enumerate(words):
        word_clusters = graphemes(word)
        for cluster_index, cluster in enumerate(word_clusters):
            clusters.append(cluster)
            is_word_end = cluster_index == len(word_clusters) - 1
            labels.append(int(is_word_end and word_index < len(words) - 1))
    return clusters, labels


def reconstruct(clusters: list[str], boundaries: list[bool | int]) -> str:
    if len(clusters) != len(boundaries):
        raise ValueError("clusters and boundaries must have equal lengths")
    pieces: list[str] = []
    for index, (cluster, boundary) in enumerate(zip(clusters, boundaries)):
        pieces.append(cluster)
        if boundary and index < len(clusters) - 1:
            pieces.append(" ")
    return "".join(pieces)
