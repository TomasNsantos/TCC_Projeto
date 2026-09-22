r"""Relatorio visual autocontido (HTML unico, imagens embutidas em base64,
sem CDN/dependencia de rede) do M1 (CUSUM) rodado sobre o grid completo do
dataset_producao_v3/. Cobre: tabela completa de metricas por estrato,
FPR vs. pi, F1 vs. recompensa, series de z-score de janelas de exemplo
(com marcacao de h e do timestep de disparo), e histograma de
timestep_disparo.

So leitura -- nao altera nenhum arquivo do dataset nem do manifesto.
Reusavel (script oficial, nao throwaway). Roda o pipeline completo
(loader -> binning -> referencia -> zscore -> cusum -> metricas) sobre os
270 arquivos do v3 (~7min, mesma ordem de grandeza ja medida em
scripts/m1_cusum_grid_completo.py) -- nao ha resultado intermediario
salvo em disco para reusar.

Uso (a partir da raiz do projeto, para `import src....` resolver):
    python -m scripts.eda.relatorio_visual_m1_cusum
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pipeline.binning import bin_janelas, calcular_referencia_negativa_treino, calcular_zscore
from src.pipeline.cusum import aplicar_cusum_por_janela, calcular_metricas_m1
from src.pipeline.loader import carregar_arquivo

DIR_V3 = Path("output/dataset_producao_v3")
CAMINHO_SAIDA = Path("output/relatorio_visual_m1_cusum.html")

K_CUSUM: float = 0.5
H_CUSUM: float = 5.0
"""Mesmos valores de scripts/m1_cusum_grid_completo.py -- ver esse script
para a justificativa completa (valores classicos de CUSUM, nao calibrados
a este problema)."""


def _fig_para_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _img_tag(b64: str, alt: str = "") -> str:
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:100%;">'


def _rodar_pipeline_completo() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Roda loader -> bin_janelas (por arquivo) -> referencia -> zscore ->
    cusum sobre os 270 arquivos do v3. Retorna (com_zscore, previsoes,
    metricas) -- com_zscore e previsoes sao usados para as series de
    exemplo (secao 4) e o histograma (secao 5); metricas para as secoes
    1-3. Mesmo padrao de scripts/m1_cusum_grid_completo.py (processamento
    arquivo por arquivo, nao carregar_diretorio + bin_janelas de uma vez
    -- ver CLAUDE.md/relatorio anterior para o motivo: window_id nao e
    global, estoura memoria se concatenado antes de binar)."""
    caminhos = sorted(DIR_V3.glob("*.h5"))
    print(f"Processando {len(caminhos)} arquivos...")

    blocos_binados = []
    for i, caminho in enumerate(caminhos):
        carregado = carregar_arquivo(caminho)
        binado = bin_janelas(carregado["fonte_a"], carregado["fonte_b"], carregado["metadados_janela"])
        blocos_binados.append(binado)
        if (i + 1) % 60 == 0:
            print(f"  {i + 1}/{len(caminhos)} arquivos binados")

    binado_completo = pd.concat(blocos_binados, ignore_index=True)
    del blocos_binados

    print("Calculando referência...")
    referencia = calcular_referencia_negativa_treino(binado_completo)

    print("Calculando z-score...")
    com_zscore = calcular_zscore(binado_completo, referencia)
    del binado_completo

    print("Aplicando CUSUM...")
    previsoes = aplicar_cusum_por_janela(com_zscore, k=K_CUSUM, h=H_CUSUM)

    print("Calculando métricas...")
    metricas = calcular_metricas_m1(previsoes)

    return com_zscore, previsoes, metricas


# ---------------------------------------------------------------------------
# Secao 1: tabela completa de metricas por estrato
# ---------------------------------------------------------------------------


def _secao_1_tabela_metricas(metricas: pd.DataFrame) -> str:
    partes = ["<h2>1. Tabela completa de métricas por estrato</h2>"]
    partes.append(
        "<p>Uma linha por (subgrupo_positivo, delta_t, recompensa, pi) -- positiva_ativa e "
        "positiva_inativa comparadas separadamente contra as mesmas negativas do estrato "
        "(ver src/pipeline/cusum.py::calcular_metricas_m1). Ordenado por delta_t, recompensa, pi.</p>"
    )

    ordenado = metricas.sort_values(["delta_t", "recompensa", "pi", "subgrupo_positivo"])
    colunas = ["subgrupo_positivo", "delta_t", "recompensa", "pi", "n_positivas", "n_negativas", "vp", "fp", "vn", "fn", "precisao", "recall", "f1", "fpr"]

    linhas_html = ["<table border='1' cellpadding='4' cellspacing='0'>"]
    linhas_html.append("<tr>" + "".join(f"<th>{c}</th>" for c in colunas) + "</tr>")
    for _, linha in ordenado.iterrows():
        celulas = []
        for c in colunas:
            valor = linha[c]
            if isinstance(valor, float):
                celulas.append(f"<td>{valor:.4f}</td>" if not pd.isna(valor) else "<td>NaN</td>")
            else:
                celulas.append(f"<td>{valor}</td>")
        linhas_html.append("<tr>" + "".join(celulas) + "</tr>")
    linhas_html.append("</table>")

    partes.append("\n".join(linhas_html))
    return "\n".join(partes)


# ---------------------------------------------------------------------------
# Secao 2: FPR vs pi, uma linha por (delta_t, recompensa)
# ---------------------------------------------------------------------------


def _secao_2_fpr_vs_pi(metricas: pd.DataFrame) -> str:
    partes = ["<h2>2. FPR vs. π, por (delta_t, recompensa)</h2>"]
    partes.append(
        "<p>Efeito de mascaramento de Fonte A: π degrada só a Fonte A (Bernoulli), não a Fonte B "
        "(estruturalmente irredutível) -- espera-se FPR crescente com π. Usa só a linha "
        "positiva_ativa (fpr é idêntico entre positiva_ativa/positiva_inativa do mesmo estrato, "
        "já que ambas comparam contra as mesmas negativas).</p>"
    )

    dados = metricas[metricas["subgrupo_positivo"] == "positiva_ativa"].copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    for (delta_t, recompensa), grupo in dados.groupby(["delta_t", "recompensa"]):
        grupo = grupo.sort_values("pi")
        ax.plot(grupo["pi"], grupo["fpr"], marker="o", label=f"delta_t={delta_t}, recompensa={recompensa}")

    ax.set_xlabel("π")
    ax.set_ylabel("FPR")
    ax.set_title("FPR vs. π")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    b64 = _fig_para_base64(fig)
    partes.append(f"<div>{_img_tag(b64, 'FPR vs pi')}</div>")
    return "\n".join(partes)


# ---------------------------------------------------------------------------
# Secao 3: F1 vs recompensa, separado por delta_t
# ---------------------------------------------------------------------------


def _secao_3_f1_vs_recompensa(metricas: pd.DataFrame) -> str:
    partes = ["<h2>3. F1 vs. recompensa, por delta_t</h2>"]
    partes.append(
        "<p>Onde o detector é mais fraco (perto de C_min) -- só positiva_ativa (positiva_inativa "
        "tem recall=0 sempre, F1 indefinido/NaN, ver relatório de texto). Uma linha por π.</p>"
    )

    dados = metricas[metricas["subgrupo_positivo"] == "positiva_ativa"].copy()
    delta_ts = sorted(dados["delta_t"].unique())
    fig, axes = plt.subplots(1, len(delta_ts), figsize=(6 * len(delta_ts), 5), sharey=True)
    if len(delta_ts) == 1:
        axes = [axes]

    for ax, delta_t in zip(axes, delta_ts):
        subset = dados[dados["delta_t"] == delta_t]
        for pi, grupo in subset.groupby("pi"):
            grupo = grupo.sort_values("recompensa")
            ax.plot(grupo["recompensa"], grupo["f1"], marker="o", label=f"π={pi}")
        ax.set_xlabel("recompensa")
        ax.set_title(f"delta_t={delta_t}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    axes[0].set_ylabel("F1")
    fig.suptitle("F1 vs. recompensa (positiva_ativa)", fontsize=12)
    fig.tight_layout()
    b64 = _fig_para_base64(fig)
    partes.append(f"<div>{_img_tag(b64, 'F1 vs recompensa')}</div>")
    return "\n".join(partes)


# ---------------------------------------------------------------------------
# Secao 4: series de z-score de janelas de exemplo
# ---------------------------------------------------------------------------


def _plot_serie_zscore(com_zscore: pd.DataFrame, chave_janela: dict, titulo: str) -> str:
    """chave_janela: dict com classe, window_id, delta_t, recompensa, pi, rho, beta, g, seed."""
    mascara = pd.Series(True, index=com_zscore.index)
    for k, v in chave_janela.items():
        mascara &= com_zscore[k] == v
    serie = com_zscore[mascara].sort_values("timestep")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(serie["timestep"], serie["zscore_a"], marker="o", markersize=3, label="zscore_a", color="#1f77b4")
    ax.plot(serie["timestep"], serie["zscore_b"], marker="s", markersize=3, label="zscore_b", color="#d62728")
    ax.axhline(H_CUSUM, color="gray", linestyle="--", linewidth=1, label=f"h={H_CUSUM}")

    impossiveis = serie[serie["timestep_estruturalmente_impossivel"]]
    if len(impossiveis) > 0:
        ax.axvspan(impossiveis["timestep"].min(), impossiveis["timestep"].max(), alpha=0.15, color="gray", label="impossível")

    ax.set_xlabel("timestep")
    ax.set_ylabel("z-score")
    ax.set_title(titulo, fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return _fig_para_base64(fig)


def _secao_4_series_exemplo(com_zscore: pd.DataFrame, previsoes: pd.DataFrame) -> str:
    partes = ["<h2>4. Séries de z-score -- janelas de exemplo</h2>"]
    partes.append(
        "<p>zscore_a (azul) e zscore_b (vermelho) por timestep; linha tracejada = h=5.0; "
        "área cinza = timesteps estruturalmente impossíveis. Marcação vertical no timestep de disparo, quando houver.</p>"
    )

    colunas_chave = ["classe", "window_id", "delta_t", "recompensa", "pi", "rho", "beta", "g", "seed"]

    # 2 positiva_ativa que dispararam
    ativas_disparadas = previsoes[(previsoes["grupo"] == "positiva_ativa") & (previsoes["disparo_m1"])].head(2)
    # 1 positiva_inativa que não disparou
    inativa_nao_disparada = previsoes[(previsoes["grupo"] == "positiva_inativa") & (~previsoes["disparo_m1"])].head(1)
    # 1 negativa que disparou (falso positivo real, se existir)
    negativa_disparada = previsoes[(previsoes["grupo"] == "negativa") & (previsoes["disparo_m1"])].head(1)

    exemplos = []
    for _, linha in ativas_disparadas.iterrows():
        exemplos.append((linha, "positiva_ativa (disparou)"))
    for _, linha in inativa_nao_disparada.iterrows():
        exemplos.append((linha, "positiva_inativa (não disparou)"))
    if len(negativa_disparada) > 0:
        for _, linha in negativa_disparada.iterrows():
            exemplos.append((linha, "negativa (disparou -- FALSO POSITIVO real)"))
    else:
        partes.append("<p><i>Nenhuma janela negativa com disparo encontrada nas primeiras linhas consultadas -- "
                       "ver nota abaixo sobre a busca.</i></p>")

    if len(exemplos) == 0:
        partes.append("<p><b>Nenhum exemplo disponível.</b></p>")
        return "\n".join(partes)

    for linha, rotulo in exemplos:
        chave_janela = {c: linha[c] for c in colunas_chave}
        titulo = (
            f"{rotulo} -- classe={linha['classe']}, window_id={linha['window_id']}, "
            f"delta_t={linha['delta_t']}, recompensa={linha['recompensa']}, pi={linha['pi']}\n"
            f"disparo_a={linha['disparo_a']} (t={linha['timestep_disparo_a']}), "
            f"disparo_b={linha['disparo_b']} (t={linha['timestep_disparo_b']})"
        )
        b64 = _plot_serie_zscore(com_zscore, chave_janela, titulo)
        partes.append(f"<div>{_img_tag(b64, rotulo)}</div>")

    return "\n".join(partes)


# ---------------------------------------------------------------------------
# Secao 5: histograma de timestep_disparo, positiva_ativa, por delta_t
# ---------------------------------------------------------------------------


def _secao_5_histograma_timestep_disparo(previsoes: pd.DataFrame) -> str:
    partes = ["<h2>5. Histograma de timestep_disparo -- positiva_ativa, por delta_t</h2>"]
    partes.append(
        "<p>Quando houve disparo (disparo_a OU disparo_b), o PRIMEIRO timestep de cruzamento de h "
        "(min de timestep_disparo_a/timestep_disparo_b quando ambos dispararam) -- detecção cedo vs. tarde na janela.</p>"
    )

    dados = previsoes[(previsoes["grupo"] == "positiva_ativa") & (previsoes["disparo_m1"])].copy()
    dados["timestep_disparo_efetivo"] = dados[["timestep_disparo_a", "timestep_disparo_b"]].min(axis=1, skipna=True)

    delta_ts = sorted(dados["delta_t"].unique())
    fig, axes = plt.subplots(1, len(delta_ts), figsize=(5 * len(delta_ts), 4))
    if len(delta_ts) == 1:
        axes = [axes]

    for ax, delta_t in zip(axes, delta_ts):
        subset = dados[dados["delta_t"] == delta_t]
        ax.hist(subset["timestep_disparo_efetivo"], bins=range(0, 26), color="#2ca02c", alpha=0.7)
        ax.set_xlabel("timestep de disparo")
        ax.set_title(f"delta_t={delta_t} (n={len(subset)})", fontsize=9)

    axes[0].set_ylabel("n janelas")
    fig.suptitle("Distribuição do timestep de disparo -- positiva_ativa", fontsize=12)
    fig.tight_layout()
    b64 = _fig_para_base64(fig)
    partes.append(f"<div>{_img_tag(b64, 'histograma timestep_disparo')}</div>")
    return "\n".join(partes)


def main() -> None:
    com_zscore, previsoes, metricas = _rodar_pipeline_completo()

    print("Gerando relatório visual...")
    secoes = [
        _secao_1_tabela_metricas(metricas),
        _secao_2_fpr_vs_pi(metricas),
        _secao_3_f1_vs_recompensa(metricas),
        _secao_4_series_exemplo(com_zscore, previsoes),
        _secao_5_histograma_timestep_disparo(previsoes),
    ]

    html = f"""<!doctype html>
<html lang="pt-br">
<head>
<meta charset="utf-8">
<title>Relatorio visual -- M1 (CUSUM)</title>
<style>
  body {{ font-family: sans-serif; max-width: 1400px; margin: 20px auto; padding: 0 16px; }}
  h1 {{ border-bottom: 2px solid #333; }}
  h2 {{ margin-top: 40px; border-bottom: 1px solid #999; }}
  table {{ border-collapse: collapse; margin: 10px 0; font-size: 12px; }}
  th, td {{ text-align: left; padding: 2px 6px; }}
  th {{ background: #eee; position: sticky; top: 0; }}
  img {{ margin: 10px 0; border: 1px solid #ddd; }}
</style>
</head>
<body>
<h1>Relatorio visual -- M1 (CUSUM) sobre output/dataset_producao_v3/</h1>
<p>Gerado por scripts/eda/relatorio_visual_m1_cusum.py -- so leitura, sem alterar dataset/manifesto.
k={K_CUSUM}, h={H_CUSUM} (valores classicos de CUSUM, nao calibrados -- ver scripts/m1_cusum_grid_completo.py).</p>
{"".join(f"<div>{s}</div>" for s in secoes)}
</body>
</html>
"""

    CAMINHO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    CAMINHO_SAIDA.write_text(html, encoding="utf-8")
    print(f"Relatorio salvo em {CAMINHO_SAIDA} ({CAMINHO_SAIDA.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
