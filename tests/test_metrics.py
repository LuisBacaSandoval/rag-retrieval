"""Pruebas de las métricas con casos calculados a mano."""

from __future__ import annotations

import math

import pytest

from rag_retrieval.evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RETRIEVED = ["a", "b", "c", "d", "e"]
RELEVANT = {"b", "d", "x"}


def test_precision_ranking_perfecto():
    assert precision_at_k(["b", "d", "a"], RELEVANT, k=2) == 1.0


def test_precision_parcial():
    # 2 relevantes (b, d) en el top-4 -> 2/4.
    assert precision_at_k(RETRIEVED, RELEVANT, k=4) == 0.5


def test_precision_sin_relevantes_recuperados():
    assert precision_at_k(["a", "c", "e"], RELEVANT, k=3) == 0.0


def test_precision_divide_entre_k_fijo():
    # 1 recuperado relevante pero k=5: las posiciones vacías cuentan -> 1/5.
    assert precision_at_k(["b"], RELEVANT, k=5) == pytest.approx(0.2)


def test_recall_cobertura_total():
    assert recall_at_k(["b", "d", "x"], RELEVANT, k=3) == 1.0


def test_recall_parcial():
    # En el top-5 aparecen b y d, pero no x -> 2/3.
    assert recall_at_k(RETRIEVED, RELEVANT, k=5) == pytest.approx(2 / 3)


def test_recall_corta_en_k():
    # x está en la posición 4, fuera de k=3 -> solo b -> 1/3.
    assert recall_at_k(["b", "a", "c", "x"], RELEVANT, k=3) == pytest.approx(1 / 3)


def test_recall_sin_relevantes_definidos():
    assert recall_at_k(RETRIEVED, set(), k=5) == 0.0


def test_recall_no_cuenta_duplicados():
    assert recall_at_k(["b", "b", "b"], RELEVANT, k=3) == pytest.approx(1 / 3)


def test_rr_primero():
    assert reciprocal_rank(["b", "a"], RELEVANT) == 1.0


def test_rr_segundo():
    assert reciprocal_rank(RETRIEVED, RELEVANT) == 0.5


def test_rr_sin_aciertos():
    assert reciprocal_rank(["a", "c", "e"], RELEVANT) == 0.0


def test_ndcg_ranking_ideal():
    assert ndcg_at_k(["b", "d", "x", "a"], RELEVANT, k=4) == pytest.approx(1.0)


def test_ndcg_un_relevante_en_posicion_2():
    # DCG = 1/log2(3); IDCG = 1/log2(2) = 1.
    assert ndcg_at_k(["a", "b"], {"b"}, k=2) == pytest.approx(1 / math.log2(3))


def test_ndcg_penaliza_relevantes_abajo():
    # Mismos aciertos en el top-4, peor colocados: nDCG los distingue, P@4 no.
    arriba = ndcg_at_k(["b", "d", "a", "c"], {"b", "d"}, k=4)
    abajo = ndcg_at_k(["a", "c", "b", "d"], {"b", "d"}, k=4)
    assert arriba == pytest.approx(1.0)
    esperado = (1 / math.log2(4) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    assert abajo == pytest.approx(esperado)


def test_ndcg_ideal_capado_por_k():
    # 3 relevantes pero k=2: colocar 2 en el top-2 ya es perfecto.
    assert ndcg_at_k(["b", "d"], RELEVANT, k=2) == pytest.approx(1.0)


def test_ndcg_sin_relevantes_definidos():
    assert ndcg_at_k(RETRIEVED, set(), k=5) == 0.0


@pytest.mark.parametrize("funcion", [precision_at_k, recall_at_k, ndcg_at_k])
def test_k_invalido_lanza_error(funcion):
    with pytest.raises(ValueError):
        funcion(RETRIEVED, RELEVANT, 0)


@pytest.mark.parametrize("funcion", [precision_at_k, recall_at_k, ndcg_at_k])
def test_lista_vacia_da_cero(funcion):
    assert funcion([], RELEVANT, 5) == 0.0
