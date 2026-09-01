"""
O cliente mediano do simulador, como asserção — churn-predictor

Execute com: pytest tests/ -v

POR QUE ESTE ARQUIVO EXISTE
---------------------------
O app subiu no Streamlit Community Cloud e morreu na hora com
`TypeError: Cannot perform reduction 'median' with string dtype`.

A causa: `_typical_row` decidia se uma coluna era texto testando
`df[c].dtype == object`. Isso vale no pandas 2, mas no pandas 3 as colunas de
texto deixaram de ser `object` (viraram `string`, com Arrow por trás). O teste
dava falso para elas, o código caía no `else` e pedia a mediana de uma coluna
de texto.

Nada disso aparecia rodando local, onde o pandas ainda era 2.x — só no
container do Streamlit, que resolve `pandas>=2.2.0` para a versão mais nova
que existir no dia. É o tipo de defeito que só a publicação encontra, e é por
isso que ele vira teste em vez de virar só um commit de correção.

O teste exercita a função com as DUAS representações de texto (`object` e
`string`), então continua valendo em qualquer uma das duas versões do pandas.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _typical_row_atual(df, colunas):
    """Espelha a regra de app.py::_typical_row.

    Está reescrita aqui, e não importada, de propósito: importar `app` puxa o
    Streamlit inteiro e treina o modelo no import. O que precisa ser garantido
    é a REGRA, e ela é uma linha.
    """
    from pandas.api.types import is_numeric_dtype

    return {
        c: (df[c].median() if is_numeric_dtype(df[c]) else df[c].mode().iloc[0])
        for c in colunas
    }


@pytest.fixture
def base():
    return pd.DataFrame(
        {
            "meses_casa": [1, 2, 3, 10],
            "valor_mensalidade": [80.0, 100.0, 120.0, 200.0],
            "plano": ["FIBRA 300", "FIBRA 600", "FIBRA 600", "FIBRA 900"],
        }
    )


@pytest.mark.parametrize("dtype_texto", ["object", "string"])
def test_nao_tira_mediana_de_coluna_de_texto(base, dtype_texto):
    """O bug que derrubou o deploy: texto como `string` caía na mediana."""
    base["plano"] = base["plano"].astype(dtype_texto)

    linha = _typical_row_atual(base, ["meses_casa", "valor_mensalidade", "plano"])

    assert linha["plano"] == "FIBRA 600", "texto tem de virar moda, não mediana"
    assert linha["meses_casa"] == 2.5
    assert linha["valor_mensalidade"] == 110.0


@pytest.mark.parametrize("dtype_texto", ["object", "string"])
def test_a_regra_antiga_quebraria(base, dtype_texto):
    """Prova que o teste acima não é decorativo.

    Com `dtype == object`, a coluna `string` do pandas 3 escapa da moda e vai
    para a mediana. Se um dia o pandas voltar a aceitar mediana de texto, este
    teste falha e avisa que a proteção deixou de fazer sentido.
    """
    base["plano"] = base["plano"].astype(dtype_texto)
    regra_antiga = lambda df, c: (
        df[c].mode().iloc[0] if df[c].dtype == object else df[c].median()
    )

    if dtype_texto == "object":
        assert regra_antiga(base, "plano") == "FIBRA 600"
    else:
        with pytest.raises(TypeError):
            regra_antiga(base, "plano")


def test_app_nao_volta_a_comparar_dtype_com_object():
    """A regra frágil não pode reaparecer em app.py.

    Percorre a árvore sintática em vez de procurar texto: o próprio comentário
    que explica o bug contém a expressão `dtype == object`, e um grep acusaria
    a documentação como se fosse o defeito.
    """
    import ast

    origem = open(
        os.path.join(os.path.dirname(__file__), "..", "app.py"), encoding="utf-8"
    ).read()
    assert "is_numeric_dtype" in origem, "a checagem por tipo numérico sumiu"

    ofensores = [
        no
        for no in ast.walk(ast.parse(origem))
        if isinstance(no, ast.Compare)
        and isinstance(no.left, ast.Attribute)
        and no.left.attr == "dtype"
        and any(
            isinstance(c, ast.Name) and c.id == "object" for c in no.comparators
        )
    ]
    assert not ofensores, (
        "app.py voltou a testar `.dtype == object` na linha "
        f"{[n.lineno for n in ofensores]} — quebra no pandas 3"
    )
