"""Relatorio visual autocontido (HTML unico, imagens embutidas em base64,
sem CDN/dependencia de rede) do dataset de producao v3
(output/dataset_producao_v3/). Cobre: timelines de Fonte A/B por grupo x
delta_t, histogramas comparativos, distribuicao de Fonte C, tabela de
janelas vazias por grupo x delta_t, e contagem de eventos por timestep
para diferentes niveis de recompensa (positiva_ativa).

So leitura -- nao altera nenhum arquivo do dataset. Reusavel (script
oficial, nao throwaway).

Uso (a partir da raiz do projeto, para `import src....` resolver):
    python -m scripts.eda.relatorio_visual_dataset_v3
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

DIR_V3 = Path("output/dataset_producao_v3")
CAMINHO_SAIDA = Path("output/relatorio_visual_dataset_v3.html")

_DELTA_TS = [0.0, 2.0, 24.0]
_SEED_PADRAO = 1
_PI_PADRAO = 0.0
_RHO_PADRAO = 0.0
_RECOMPENSA_PADRAO = 0.5
"""recompensa=0.5 (nao 1.0) -- unico nivel onde o grupo positiva_inativa
existe em volume real (~90% das janelas positivas ficam inativas nesse
nivel, ver CLAUDE.md/achado de resultado_alvo). Com recompensa=1.0 a
ativacao e 100%, entao positiva_inativa fica vazio e as secoes 1/2/3 nao
teriam nada para mostrar desse grupo."""

_CORES = {"fonte_a": "#1f77b4", "fonte_b": "#d62728"}


def _caminho(delta_t: float, recompensa: float = _RECOMPENSA_PADRAO, rho: float = _RHO_PADRAO,
             pi: float = _PI_PADRAO, seed: int = _SEED_PADRAO) -> Path:
    nome = (
        f"beta-1_delta_t-{delta_t:.4f}_g-pool_pi-{pi:.4f}_"
        f"recompensa-{recompensa:.4f}_rho-{rho:.4f}_seed-{seed}.h5"
    )
    caminho = DIR_V3 / nome
    if not caminho.exists():
        raise FileNotFoundError(f"Combinacao esperada nao encontrada: {caminho}")
    return caminho


def _fig_para_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _img_tag(b64: str, alt: str = "") -> str:
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:100%;">'


def _carregar_grupos_de_arquivo(caminho: Path) -> dict[str, dict[str, pd.DataFrame]]:
    """Le um .h5 e separa fonte_a/fonte_b/fonte_c_secao/metadados nos tres
    grupos (negativa / positiva_ativa / positiva_inativa), via merge com
    metadados_janela.contrato_ativado -- mesmo padrao de
    scripts/eda/eda_dataset_producao_v2.py."""
    md = pd.read_hdf(caminho, "metadados_janela")
    fa = pd.read_hdf(caminho, "fonte_a")
    fb = pd.read_hdf(caminho, "fonte_b")
    fc = pd.read_hdf(caminho, "fonte_c_secao")
    fc_mun = pd.read_hdf(caminho, "fonte_c_municipio")
    fc_est = pd.read_hdf(caminho, "fonte_c_estado")

    chave = md[["window_id", "classe", "contrato_ativado"]].drop_duplicates()

    neg = chave["classe"] == "negativa"
    pos_ativa = (chave["classe"] == "positiva") & (chave["contrato_ativado"] == 1.0)
    pos_inativa = (chave["classe"] == "positiva") & (chave["contrato_ativado"] == 0.0)

    grupos = {}
    for nome_grupo, mascara in (("negativa", neg), ("positiva_ativa", pos_ativa), ("positiva_inativa", pos_inativa)):
        janelas_alvo = chave.loc[mascara, ["window_id", "classe"]]
        grupos[nome_grupo] = {
            "fonte_a": fa.merge(janelas_alvo, on=["window_id", "classe"], how="inner"),
            "fonte_b": fb.merge(janelas_alvo, on=["window_id", "classe"], how="inner"),
            "fonte_c_secao": fc.merge(janelas_alvo, on=["window_id", "classe"], how="inner"),
            "fonte_c_municipio": fc_mun.merge(janelas_alvo, on=["window_id", "classe"], how="inner"),
            "fonte_c_estado": fc_est.merge(janelas_alvo, on=["window_id", "classe"], how="inner"),
            "metadados": md.merge(janelas_alvo, on=["window_id", "classe"], how="inner"),
        }
    return grupos


# ---------------------------------------------------------------------------
# Secao 1: timelines de exemplo (2 janelas por grupo x delta_t)
# ---------------------------------------------------------------------------


def _secao_1_timelines() -> str:
    partes = ["<h2>1. Timelines de exemplo -- Fonte A e Fonte B (2 janelas por grupo x delta_t)</h2>"]
    partes.append(
        "<p>Amostra reprodutivel: pi=0.0, recompensa=0.5, rho=0.0, seed=1 "
        "(mesma combinacao base nos tres niveis de delta_t, so delta_t varia). "
        "Cada subplot mostra timestamps continuos de Fonte A (fonte_a_eventos_fronteira -- "
        "aqui reconstruidos via timestamp_medio por linha, ja que o timestamp bruto por evento "
        "nao e persistido individualmente) e Fonte B (timestamp por evento, persistido diretamente).</p>"
    )

    for delta_t in _DELTA_TS:
        caminho = _caminho(delta_t)
        grupos = _carregar_grupos_de_arquivo(caminho)
        partes.append(f"<h3>delta_t = {delta_t}</h3>")

        for nome_grupo in ("negativa", "positiva_ativa", "positiva_inativa"):
            dados = grupos[nome_grupo]
            window_ids_disponiveis = sorted(dados["metadados"]["window_id"].unique())
            if not window_ids_disponiveis:
                partes.append(f"<p><b>{nome_grupo}</b>: nenhuma janela deste grupo nesta combinacao.</p>")
                continue

            amostra_windows = window_ids_disponiveis[:2]
            fig, axes = plt.subplots(1, len(amostra_windows), figsize=(6 * len(amostra_windows), 2.2), squeeze=False)
            axes = axes[0]

            for ax, window_id in zip(axes, amostra_windows):
                fa_janela = dados["fonte_a"][dados["fonte_a"]["window_id"] == window_id]
                fb_janela = dados["fonte_b"][dados["fonte_b"]["window_id"] == window_id]

                if len(fa_janela) > 0:
                    ax.scatter(
                        fa_janela["timestamp_medio"], np.ones(len(fa_janela)) * 1.0,
                        s=np.clip(fa_janela["n_eventos"] * 8, 10, 200), color=_CORES["fonte_a"],
                        alpha=0.6, label="Fonte A (timestamp_medio, tamanho~n_eventos)",
                    )
                if len(fb_janela) > 0:
                    ax.scatter(
                        fb_janela["timestamp"], np.ones(len(fb_janela)) * 0.0,
                        s=15, color=_CORES["fonte_b"], alpha=0.5, marker="|", label="Fonte B",
                    )

                ax.set_yticks([0, 1])
                ax.set_yticklabels(["Fonte B", "Fonte A"])
                ax.set_xlabel("timestamp")
                ax.set_title(f"window_id={window_id} (nA={len(fa_janela)}, nB={len(fb_janela)})", fontsize=9)
                ax.set_ylim(-0.5, 1.5)

            fig.suptitle(f"{nome_grupo} -- delta_t={delta_t}", fontsize=11)
            fig.tight_layout()
            b64 = _fig_para_base64(fig)
            partes.append(f"<div>{_img_tag(b64, f'{nome_grupo} delta_t={delta_t}')}</div>")

    return "\n".join(partes)


# ---------------------------------------------------------------------------
# Secao 2: histogramas comparativos por grupo, separados por delta_t
# ---------------------------------------------------------------------------


def _secao_2_histogramas() -> str:
    partes = ["<h2>2. Histogramas comparativos entre grupos (por delta_t)</h2>"]
    partes.append(
        "<p>n_eventos e volume agregados por janela (soma sobre todos os timesteps da janela, Fonte A); "
        "contagem de eventos por janela, Fonte B. Mesma combinacao base da secao 1 (pi=0.0, recompensa=0.5, rho=0.0, seed=1).</p>"
    )

    cores_grupo = {"negativa": "#7f7f7f", "positiva_ativa": "#2ca02c", "positiva_inativa": "#ff7f0e"}

    for delta_t in _DELTA_TS:
        caminho = _caminho(delta_t)
        grupos = _carregar_grupos_de_arquivo(caminho)
        partes.append(f"<h3>delta_t = {delta_t}</h3>")

        fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))

        for nome_grupo in ("negativa", "positiva_ativa", "positiva_inativa"):
            fa = grupos[nome_grupo]["fonte_a"]
            fb = grupos[nome_grupo]["fonte_b"]
            cor = cores_grupo[nome_grupo]

            if len(fa) > 0:
                n_eventos_por_janela = fa.groupby("window_id")["n_eventos"].sum()
                volume_por_janela = fa.groupby("window_id")["volume"].sum()
                axes[0].hist(n_eventos_por_janela, bins=30, alpha=0.5, label=nome_grupo, color=cor)
                axes[1].hist(volume_por_janela, bins=30, alpha=0.5, label=nome_grupo, color=cor)

            if len(fb) > 0:
                contagem_b_por_janela = fb.groupby("window_id").size()
                axes[2].hist(contagem_b_por_janela, bins=30, alpha=0.5, label=nome_grupo, color=cor)

        axes[0].set_title("n_eventos por janela (Fonte A, somado)")
        axes[1].set_title("volume por janela (Fonte A, somado)")
        axes[2].set_title("contagem de eventos (Fonte B)")
        for ax in axes:
            ax.legend(fontsize=8)
            ax.set_ylabel("n janelas")

        fig.suptitle(f"delta_t={delta_t}", fontsize=11)
        fig.tight_layout()
        b64 = _fig_para_base64(fig)
        partes.append(f"<div>{_img_tag(b64, f'histogramas delta_t={delta_t}')}</div>")

    return "\n".join(partes)


# ---------------------------------------------------------------------------
# Secao 3: Fonte C -- distribuicao por secao + checagem municipio/estado
# ---------------------------------------------------------------------------


def _secao_3_fonte_c() -> str:
    partes = ["<h2>3. Fonte C -- distribuicao de fracao_candidato_alvo por secao</h2>"]
    partes.append(
        "<p>Amostra de ate 30 janelas por grupo (delta_t=2.0, mesma combinacao base). "
        "Boxplot das 5 secoes (fonte_c_secao); nota textual comparando a MEDIA das 5 secoes "
        "contra o valor unico persistido em fonte_c_municipio/fonte_c_estado (hierarquia "
        "default colapsa tudo num unico municipio/estado -- ver CLAUDE.md).</p>"
    )

    caminho = _caminho(2.0)
    grupos = _carregar_grupos_de_arquivo(caminho)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, nome_grupo in zip(axes, ("negativa", "positiva_ativa", "positiva_inativa")):
        fc = grupos[nome_grupo]["fonte_c_secao"]
        window_ids_amostra = sorted(fc["window_id"].unique())[:30]
        fc_amostra = fc[fc["window_id"].isin(window_ids_amostra)]

        dados_por_secao = [
            fc_amostra.loc[fc_amostra["unidade"] == u, "fracao_candidato_alvo"].to_numpy()
            for u in range(5)
        ]
        ax.boxplot(dados_por_secao, tick_labels=[f"secao {u}" for u in range(5)])
        ax.set_title(f"{nome_grupo}\n(n_janelas amostra={len(window_ids_amostra)})", fontsize=9)
        ax.set_ylabel("fracao_candidato_alvo")
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Fonte C por secao -- delta_t=2.0", fontsize=11)
    fig.tight_layout()
    b64 = _fig_para_base64(fig)
    partes.append(f"<div>{_img_tag(b64, 'fonte C por secao')}</div>")

    # Checagem numerica: media das 5 secoes vs valor unico de municipio/estado,
    # por window_id, para o grupo positiva_ativa (grupo com mais janelas
    # tipicamente nao-degeneradas).
    partes.append("<h3>Checagem numerica -- média(fonte_c_secao) vs. fonte_c_municipio/estado</h3>")
    linhas_tabela = ["<table border='1' cellpadding='4' cellspacing='0'>",
                      "<tr><th>grupo</th><th>window_id (amostra)</th>"
                      "<th>média das 5 seções</th><th>fonte_c_municipio</th>"
                      "<th>fonte_c_estado</th><th>diferença (média_secao - município)</th></tr>"]
    for nome_grupo in ("negativa", "positiva_ativa", "positiva_inativa"):
        fc_secao = grupos[nome_grupo]["fonte_c_secao"]
        fc_mun = grupos[nome_grupo]["fonte_c_municipio"]
        fc_est = grupos[nome_grupo]["fonte_c_estado"]
        window_ids_amostra = sorted(fc_secao["window_id"].unique())[:3]
        for window_id in window_ids_amostra:
            media_secao = fc_secao.loc[fc_secao["window_id"] == window_id, "fracao_candidato_alvo"].mean()
            valor_mun = fc_mun.loc[fc_mun["window_id"] == window_id, "fracao_candidato_alvo"].iloc[0]
            valor_est = fc_est.loc[fc_est["window_id"] == window_id, "fracao_candidato_alvo"].iloc[0]
            diferenca = media_secao - valor_mun
            linhas_tabela.append(
                f"<tr><td>{nome_grupo}</td><td>{window_id}</td>"
                f"<td>{media_secao:.6f}</td><td>{valor_mun:.6f}</td>"
                f"<td>{valor_est:.6f}</td><td>{diferenca:+.6f}</td></tr>"
            )
    linhas_tabela.append("</table>")
    partes.append("\n".join(linhas_tabela))
    partes.append(
        "<p><i>Nota: média das 5 seções não é necessariamente idêntica ao valor de "
        "município/estado -- fonte_c_municipio/estado é calculado sobre a hierarquia de "
        "agentes (município/estado, colapsados no default), não como média simples das "
        "frações por seção; as duas só coincidem exatamente se cada seção tiver o mesmo "
        "número de agentes, o que não é garantido pela atribuição uniforme aleatória "
        "(ver ElectionModel.resultado_eleitoral_por_secao/municipio, model.py). A diferença "
        "numérica mostrada na tabela é o valor real medido, não assumido.</i></p>"
    )

    return "\n".join(partes)


# ---------------------------------------------------------------------------
# Secao 4: tabela resumo -- fracao de janelas com fonte_a/fonte_b vazia
# ---------------------------------------------------------------------------


def _secao_4_tabela_vazias() -> str:
    partes = ["<h2>4. Tabela resumo -- fração de janelas com Fonte A / Fonte B vazia (grupo x delta_t)</h2>"]
    partes.append(
        "<p>Agregado sobre TODOS os arquivos do v3 com aquele delta_t (18 arquivos por nível: "
        "5 pi x 3 recompensa x 3 rho x 2 seeds / mas cada arquivo já cobre 1000 janelas por classe -- "
        "agregação feita sobre os 90 arquivos de cada delta_t).</p>"
    )

    linhas_tabela = [
        "<table border='1' cellpadding='4' cellspacing='0'>",
        "<tr><th>delta_t</th><th>grupo</th><th>n_janelas</th>"
        "<th>fração Fonte A vazia</th><th>fração Fonte B vazia</th></tr>",
    ]

    for delta_t in _DELTA_TS:
        arquivos_delta_t = sorted(DIR_V3.glob(f"*delta_t-{delta_t:.4f}*.h5"))
        contadores = {
            "negativa": {"total": 0, "a_vazia": 0, "b_vazia": 0},
            "positiva_ativa": {"total": 0, "a_vazia": 0, "b_vazia": 0},
            "positiva_inativa": {"total": 0, "a_vazia": 0, "b_vazia": 0},
        }

        for caminho in arquivos_delta_t:
            md = pd.read_hdf(caminho, "metadados_janela")
            fa = pd.read_hdf(caminho, "fonte_a")
            fb = pd.read_hdf(caminho, "fonte_b")

            janelas_com_a = set(zip(fa["classe"], fa["window_id"]))
            janelas_com_b = set(zip(fb["classe"], fb["window_id"]))

            chave = md[["window_id", "classe", "contrato_ativado"]].drop_duplicates()
            neg = chave["classe"] == "negativa"
            pos_ativa = (chave["classe"] == "positiva") & (chave["contrato_ativado"] == 1.0)
            pos_inativa = (chave["classe"] == "positiva") & (chave["contrato_ativado"] == 0.0)

            for nome_grupo, mascara in (("negativa", neg), ("positiva_ativa", pos_ativa), ("positiva_inativa", pos_inativa)):
                for _, row in chave.loc[mascara].iterrows():
                    chave_janela = (row["classe"], row["window_id"])
                    contadores[nome_grupo]["total"] += 1
                    if chave_janela not in janelas_com_a:
                        contadores[nome_grupo]["a_vazia"] += 1
                    if chave_janela not in janelas_com_b:
                        contadores[nome_grupo]["b_vazia"] += 1

        for nome_grupo, c in contadores.items():
            if c["total"] == 0:
                continue
            frac_a = c["a_vazia"] / c["total"]
            frac_b = c["b_vazia"] / c["total"]
            linhas_tabela.append(
                f"<tr><td>{delta_t}</td><td>{nome_grupo}</td><td>{c['total']}</td>"
                f"<td>{frac_a:.4f} ({frac_a*100:.2f}%)</td><td>{frac_b:.4f} ({frac_b*100:.2f}%)</td></tr>"
            )

    linhas_tabela.append("</table>")
    partes.append("\n".join(linhas_tabela))
    return "\n".join(partes)


# ---------------------------------------------------------------------------
# Secao 5: contagem de eventos por timestep, positiva_ativa, por recompensa
# ---------------------------------------------------------------------------


def _secao_5_binning_recompensa() -> str:
    partes = ["<h2>5. Contagem de eventos por timestep -- positiva_ativa, por nível de recompensa</h2>"]
    partes.append(
        "<p>Binning que alimenta features tipo CUSUM/M1 (contagem de Fonte A por timestep inteiro). "
        "delta_t=24.0, pi=0.0, rho=0.0, seed=1 -- mesma combinação base, variando só recompensa "
        "(0.5 / 1.0 / 1.5). 2 janelas de exemplo por nível.</p>"
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 3.5), sharey=True)

    for ax, recompensa in zip(axes, (0.5, 1.0, 1.5)):
        caminho = _caminho(24.0, recompensa=recompensa)
        grupos = _carregar_grupos_de_arquivo(caminho)
        fa = grupos["positiva_ativa"]["fonte_a"]
        window_ids = sorted(fa["window_id"].unique())[:4]

        for window_id in window_ids:
            fa_janela = fa[fa["window_id"] == window_id].sort_values("timestep")
            ax.plot(fa_janela["timestep"], fa_janela["n_eventos"], marker="o", markersize=3,
                     alpha=0.7, label=f"window_id={window_id}")

        ax.set_title(f"recompensa={recompensa}", fontsize=10)
        ax.set_xlabel("timestep")
        ax.legend(fontsize=7)

    axes[0].set_ylabel("n_eventos (Fonte A)")
    fig.suptitle("positiva_ativa -- delta_t=24.0", fontsize=11)
    fig.tight_layout()
    b64 = _fig_para_base64(fig)
    partes.append(f"<div>{_img_tag(b64, 'binning por recompensa')}</div>")

    return "\n".join(partes)


def main() -> None:
    print("Gerando relatorio visual do dataset v3...")
    secoes = [
        _secao_1_timelines(),
        _secao_2_histogramas(),
        _secao_3_fonte_c(),
        _secao_4_tabela_vazias(),
        _secao_5_binning_recompensa(),
    ]

    html = f"""<!doctype html>
<html lang="pt-br">
<head>
<meta charset="utf-8">
<title>Relatorio visual -- dataset_producao_v3</title>
<style>
  body {{ font-family: sans-serif; max-width: 1400px; margin: 20px auto; padding: 0 16px; }}
  h1 {{ border-bottom: 2px solid #333; }}
  h2 {{ margin-top: 40px; border-bottom: 1px solid #999; }}
  h3 {{ margin-top: 20px; color: #444; }}
  table {{ border-collapse: collapse; margin: 10px 0; }}
  th, td {{ text-align: left; }}
  img {{ margin: 10px 0; border: 1px solid #ddd; }}
</style>
</head>
<body>
<h1>Relatorio visual -- output/dataset_producao_v3/</h1>
<p>Gerado por scripts/eda/relatorio_visual_dataset_v3.py -- so leitura, sem alterar o dataset.
Amostra reprodutivel (seed fixa) usada como base para as secoes 1, 2, 3 e 5: pi=0.0,
recompensa=0.5 (exceto secao 5, que varia recompensa), rho=0.0, seed=1 -- recompensa=0.5
é o único nível com volume real de janelas positiva_inativa (~90%% das positivas ficam
inativas nesse nível, ver CLAUDE.md).</p>
{"".join(f"<div>{s}</div>" for s in secoes)}
</body>
</html>
"""

    CAMINHO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    CAMINHO_SAIDA.write_text(html, encoding="utf-8")
    print(f"Relatorio salvo em {CAMINHO_SAIDA} ({CAMINHO_SAIDA.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
