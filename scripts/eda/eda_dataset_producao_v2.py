"""EDA simples do dataset de producao v2 (output/dataset_producao_v2/, 270
arquivos HDF5). Le uma amostra deterministica de arquivos cobrindo extremos
e meio da grade fatorial (pi x delta_t x recompensa), separa em tres grupos
via metadados_janela.contrato_ativado e reporta describe() de Fonte A/B/C
por grupo, mais a quebra de ativacao por recompensa. So leitura -- nao
altera nenhum arquivo do dataset.

Uso:
    python -m scripts.eda.eda_dataset_producao_v2
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 140)

DIRETORIO_DATASET = Path("output/dataset_producao_v2")

_PADRAO_NOME = re.compile(
    r"beta-(?P<beta>[\d.]+)_delta_t-(?P<dt>[\d.]+)_g-(?P<g>\w+)_pi-(?P<pi>[\d.]+)_"
    r"recompensa-(?P<rec>[\d.]+)_rho-(?P<rho>[\d.]+)_seed-(?P<seed>\d+)\.h5"
)

# Amostra determinística: os 3 níveis de recompensa (extremos + implícito
# "meio" já é o próprio 1.0) cruzados com pi extremo/meio e delta_t
# extremo/meio, em rho=0.0 e rho=1.0 (extremos de rho), seed 1 e 2 — cobre
# os cantos e o centro da grade sem precisar dos 270 arquivos.
_COMBINACOES_AMOSTRA = [
    # (pi, dt, rec, rho, seed)
    (0.0, 0.0, 0.5, 0.0, 1),
    (0.0, 0.0, 0.5, 0.0, 2),
    (0.0, 0.0, 1.0, 0.0, 1),
    (0.0, 0.0, 1.5, 0.0, 1),
    (0.95, 24.0, 0.5, 1.0, 1),
    (0.95, 24.0, 1.0, 1.0, 1),
    (0.95, 24.0, 1.5, 1.0, 1),
    (0.5, 2.0, 0.5, 0.5, 1),
    (0.5, 2.0, 1.0, 0.5, 1),
    (0.5, 2.0, 1.5, 0.5, 2),
    (0.75, 0.0, 0.5, 1.0, 1),
    (0.75, 24.0, 1.0, 0.0, 2),
    (0.9, 2.0, 1.5, 0.0, 1),
    (0.9, 24.0, 0.5, 0.5, 2),
    (0.0, 24.0, 1.0, 1.0, 2),
    (0.95, 0.0, 1.5, 0.5, 1),
    (0.5, 0.0, 0.5, 1.0, 2),
    (0.75, 2.0, 1.0, 1.0, 1),
]


def _caminho_para_combinacao(pi: float, dt: float, rec: float, rho: float, seed: int) -> Path:
    nome = (
        f"beta-1_delta_t-{dt:.4f}_g-pool_pi-{pi:.4f}_"
        f"recompensa-{rec:.4f}_rho-{rho:.4f}_seed-{seed}.h5"
    )
    return DIRETORIO_DATASET / nome


def _selecionar_amostra() -> list[Path]:
    caminhos = []
    for pi, dt, rec, rho, seed in _COMBINACOES_AMOSTRA:
        caminho = _caminho_para_combinacao(pi, dt, rec, rho, seed)
        if not caminho.exists():
            raise FileNotFoundError(f"Combinacao da amostra nao encontrada: {caminho}")
        caminhos.append(caminho)
    return caminhos


def _extrair_params(caminho: Path) -> dict[str, float | int | str]:
    m = _PADRAO_NOME.search(caminho.name)
    if not m:
        raise ValueError(f"Nome de arquivo fora do padrao esperado: {caminho.name}")
    d = m.groupdict()
    return {
        "pi": float(d["pi"]),
        "dt": float(d["dt"]),
        "rec": float(d["rec"]),
        "rho": float(d["rho"]),
        "seed": int(d["seed"]),
    }


def _carregar_grupos(caminhos: list[Path]) -> dict[str, dict[str, pd.DataFrame]]:
    """Concatena fonte_a/fonte_b/fonte_c_secao/metadados de todos os arquivos
    da amostra, já separados nos três grupos (negativa / positiva_ativa /
    positiva_inativa), via merge com metadados_janela.contrato_ativado."""
    partes = {
        "negativa": {"fonte_a": [], "fonte_b": [], "fonte_c": []},
        "positiva_ativa": {"fonte_a": [], "fonte_b": [], "fonte_c": []},
        "positiva_inativa": {"fonte_a": [], "fonte_b": [], "fonte_c": []},
    }
    registros_ativacao = []

    for caminho in caminhos:
        params = _extrair_params(caminho)
        md = pd.read_hdf(caminho, "metadados_janela")
        fa = pd.read_hdf(caminho, "fonte_a")
        fb = pd.read_hdf(caminho, "fonte_b")
        fc = pd.read_hdf(caminho, "fonte_c_secao")

        chave = md[["window_id", "classe", "contrato_ativado"]].drop_duplicates()

        fa_m = fa.merge(chave, on=["window_id", "classe"], how="left")
        fb_m = fb.merge(chave, on=["window_id", "classe"], how="left")
        fc_m = fc.merge(chave, on=["window_id", "classe"], how="left")

        neg = chave["classe"] == "negativa"
        pos_ativa = (chave["classe"] == "positiva") & (chave["contrato_ativado"] == 1.0)
        pos_inativa = (chave["classe"] == "positiva") & (chave["contrato_ativado"] == 0.0)

        pos = md[md["classe"] == "positiva"]
        n_ativado = int((pos["contrato_ativado"] == 1.0).sum())
        registros_ativacao.append({**params, "n_janelas": len(pos), "n_ativado": n_ativado})

        for nome_grupo, mascara_janelas in (
            ("negativa", neg),
            ("positiva_ativa", pos_ativa),
            ("positiva_inativa", pos_inativa),
        ):
            janelas_alvo = chave.loc[mascara_janelas, ["window_id", "classe"]]
            fa_g = fa_m.merge(janelas_alvo, on=["window_id", "classe"], how="inner")
            fb_g = fb_m.merge(janelas_alvo, on=["window_id", "classe"], how="inner")
            fc_g = fc_m.merge(janelas_alvo, on=["window_id", "classe"], how="inner")
            partes[nome_grupo]["fonte_a"].append(fa_g)
            partes[nome_grupo]["fonte_b"].append(fb_g)
            partes[nome_grupo]["fonte_c"].append(fc_g)

    grupos = {
        nome: {
            "fonte_a": pd.concat(fontes["fonte_a"], ignore_index=True),
            "fonte_b": pd.concat(fontes["fonte_b"], ignore_index=True),
            "fonte_c": pd.concat(fontes["fonte_c"], ignore_index=True),
        }
        for nome, fontes in partes.items()
    }
    df_ativacao = pd.DataFrame(registros_ativacao)
    return grupos, df_ativacao


def _reportar_grupo(nome: str, dados: dict[str, pd.DataFrame]) -> None:
    fa, fb, fc = dados["fonte_a"], dados["fonte_b"], dados["fonte_c"]

    print()
    print("#" * 100)
    print(f"GRUPO: {nome}")
    print("#" * 100)

    print(f"\n-- FONTE A (n_linhas={len(fa)}) --")
    if len(fa) > 0:
        colunas_dados_a = ["n_eventos", "volume", "timestamp_medio", "dispersao_timestamp"]
        print(fa[["n_eventos", "volume", "dispersao_timestamp"]].describe().to_string())
        nan_a = int(fa[colunas_dados_a].isna().sum().sum())
        print(f"NaN em fonte_a (colunas de dados, exclui contrato_ativado do merge): {nan_a}")
    else:
        print("(vazio)")

    print(f"\n-- FONTE B (n_linhas={len(fb)}) --")
    if len(fb) > 0:
        print(fb[["timestamp"]].describe().to_string())
        nan_b = int(fb["timestamp"].isna().sum())
        print(f"NaN em fonte_b: {nan_b}")
    else:
        print("(vazio)")

    print(f"\n-- FONTE C (n_linhas={len(fc)}) --")
    if len(fc) > 0:
        print(fc["fracao_candidato_alvo"].describe().to_string())
        nan_c = int(fc["fracao_candidato_alvo"].isna().sum())
        print(f"NaN em fonte_c.fracao_candidato_alvo: {nan_c}")
    else:
        print("(vazio)")
    print()


def main() -> None:
    caminhos = _selecionar_amostra()
    print(f"amostra: {len(caminhos)} arquivos de {len(list(DIRETORIO_DATASET.glob('*.h5')))} totais")
    for c in caminhos:
        print(f"  {c.name}")

    grupos, df_ativacao = _carregar_grupos(caminhos)

    for nome in ("negativa", "positiva_ativa", "positiva_inativa"):
        _reportar_grupo(nome, grupos[nome])

    print()
    print("=" * 100)
    print("QUEBRA DE ATIVACAO POR RECOMPENSA (arquivos da amostra)")
    print("=" * 100)
    for rec in sorted(df_ativacao["rec"].unique()):
        s = df_ativacao[df_ativacao["rec"] == rec]
        total_j = int(s["n_janelas"].sum())
        total_a = int(s["n_ativado"].sum())
        frac = total_a / total_j if total_j else float("nan")
        print(f"recompensa={rec}: {total_a}/{total_j} ativadas = {frac:.4f}  (arquivos: {len(s)})")


if __name__ == "__main__":
    main()
