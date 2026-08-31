"""Testes do runner resumível (`orquestrar`), com um fake local -- nunca
importa ElectionModel nem os geradores reais.
"""

from __future__ import annotations

import os

import mlflow
import pytest

from src.pipeline.config import (
    GradeFatorial,
    ParametrosPopulacionaisStub,
    ParametrosStubGeracao,
    expandir_grade,
    run_id,
)
from src.pipeline.manifest import Manifesto
from src.pipeline.runner import orquestrar, orquestrar_paralelo
from src.pipeline.tracking import configurar_mlflow


@pytest.fixture(autouse=True)
def _tracking_uri_isolado(tmp_path) -> None:
    """`orquestrar`/`orquestrar_paralelo` agora chamam `registrar_run_mlflow`
    incondicionalmente -- sem isto, o tracking URI default do MLflow
    (`sqlite:///mlflow.db`, relativo ao cwd) poluiria a raiz do repositório
    a cada execução da suíte. `autouse=True` cobre todos os testes deste
    arquivo, inclusive os que não mencionam MLflow explicitamente."""
    configurar_mlflow(tmp_path / "mlflow.db")


class FakeGeradorParDeClasses:
    """Fake injetável: sucede sempre, exceto para as seeds em ``falha_seeds``,
    onde levanta ``ValueError``. Registra toda chamada em ``chamadas`` para
    os testes verificarem quantas vezes (e com quais argumentos) foi usada.

    ``falha_seeds`` seleciona por valor de seed sozinho, não por combinação
    -- só é preciso quando a grade tem uma única seed por linha (senão o
    mesmo valor de seed aparece em várias combinações, ver
    ``falha_predicado`` para selecionar por combinação exata).

    **Nota sobre uso sob `orquestrar_paralelo` (backend `loky`):** cada
    worker roda numa cópia do fake, picklada e enviada a um processo
    separado -- mutações em `self.chamadas` dentro de um worker NÃO se
    propagam de volta para a instância vista pelo teste no processo
    principal (memória separada). Por isso `caminho_output` inclui
    `os.getpid()` -- é o único canal para o teste no processo principal
    observar de qual processo o resultado veio, sem depender de estado
    mutado remotamente."""

    def __init__(
        self,
        falha_seeds: frozenset[int] = frozenset(),
        falha_predicado=None,
    ) -> None:
        self.chamadas: list[tuple[dict, int, int]] = []
        self.falha_seeds = falha_seeds
        self.falha_predicado = falha_predicado

    def __call__(self, params: dict, seed: int, n_janelas: int) -> dict:
        self.chamadas.append((params, seed, n_janelas))
        deve_falhar = seed in self.falha_seeds or (self.falha_predicado and self.falha_predicado(params, seed))
        if deve_falhar:
            raise ValueError(f"falha simulada para seed={seed}")
        return {
            "n_janelas_ok": n_janelas,
            "n_janelas_falha": 0,
            "n_contrato_nao_ativado": 1,
            "caminho_output": f"fake_{seed}_pid{os.getpid()}.h5",
        }


@pytest.fixture
def caminho_manifesto(tmp_path) -> str:
    return str(tmp_path / "manifesto.sqlite")


@pytest.fixture
def grade_pequena() -> GradeFatorial:
    return GradeFatorial(g=["secao", "municipio"], delta_t=[2.0], recompensa=[5.0], rho=[0.0, 0.5], beta=[1], seeds=[1, 2])


@pytest.fixture
def stub_geracao() -> ParametrosStubGeracao:
    return ParametrosStubGeracao(tau_kendall=0.5, taxa_fonte_a=1.0, volume_medio_fonte_a=1000.0, taxa_fonte_b=1.0)


@pytest.fixture
def populacionais() -> ParametrosPopulacionaisStub:
    return ParametrosPopulacionaisStub()


def _run_ids_da_grade(grade: GradeFatorial) -> list[str]:
    combinacoes = expandir_grade(grade)
    return [run_id({k: v for k, v in c.items() if k != "seed"}, c["seed"]) for c in combinacoes]


def test_execucao_completa_marca_tudo_success(
    grade_pequena: GradeFatorial,
    stub_geracao: ParametrosStubGeracao,
    populacionais: ParametrosPopulacionaisStub,
    caminho_manifesto: str,
) -> None:
    manifesto = Manifesto(caminho_manifesto)
    fake = FakeGeradorParDeClasses()

    orquestrar(grade_pequena, stub_geracao, populacionais, n_janelas_por_classe=10, manifesto=manifesto, gerar_par_de_classes=fake)

    run_ids_esperados = _run_ids_da_grade(grade_pequena)
    assert len(fake.chamadas) == len(run_ids_esperados)
    assert manifesto.pendentes() == []
    for rid in run_ids_esperados:
        assert manifesto.obter(rid)["status"] == "success"
    manifesto.close()


def test_reexecucao_apos_sucesso_nao_chama_gerador_de_novo(
    grade_pequena: GradeFatorial,
    stub_geracao: ParametrosStubGeracao,
    populacionais: ParametrosPopulacionaisStub,
    caminho_manifesto: str,
) -> None:
    manifesto = Manifesto(caminho_manifesto)
    orquestrar(grade_pequena, stub_geracao, populacionais, 10, manifesto, FakeGeradorParDeClasses())

    fake_segunda_chamada = FakeGeradorParDeClasses()
    orquestrar(grade_pequena, stub_geracao, populacionais, 10, manifesto, fake_segunda_chamada)

    assert fake_segunda_chamada.chamadas == []
    manifesto.close()


def test_falha_marca_failed_outras_ficam_success_e_retry_so_a_que_falhou(
    grade_pequena: GradeFatorial,
    stub_geracao: ParametrosStubGeracao,
    populacionais: ParametrosPopulacionaisStub,
    caminho_manifesto: str,
) -> None:
    manifesto = Manifesto(caminho_manifesto)
    combinacoes = expandir_grade(grade_pequena)
    combinacao_que_falha = combinacoes[1]
    seed_que_falha = combinacao_que_falha["seed"]
    params_que_falha = {k: v for k, v in combinacao_que_falha.items() if k != "seed"}
    rid_que_falha = run_id(params_que_falha, seed_que_falha)

    fake_com_falha = FakeGeradorParDeClasses(
        falha_predicado=lambda params, seed: all(params.get(k) == v for k, v in params_que_falha.items()) and seed == seed_que_falha
    )
    orquestrar(grade_pequena, stub_geracao, populacionais, 10, manifesto, fake_com_falha)

    linha_falha = manifesto.obter(rid_que_falha)
    assert linha_falha["status"] == "failed"
    assert "falha simulada" in linha_falha["erro"]

    outros_run_ids = [rid for rid in _run_ids_da_grade(grade_pequena) if rid != rid_que_falha]
    for rid in outros_run_ids:
        assert manifesto.obter(rid)["status"] == "success"

    fake_retry = FakeGeradorParDeClasses()
    orquestrar(grade_pequena, stub_geracao, populacionais, 10, manifesto, fake_retry)

    assert len(fake_retry.chamadas) == 1
    assert fake_retry.chamadas[0][1] == seed_que_falha
    assert manifesto.obter(rid_que_falha)["status"] == "success"
    manifesto.close()


def test_running_orfao_de_execucao_interrompida_e_retentado(
    grade_pequena: GradeFatorial,
    stub_geracao: ParametrosStubGeracao,
    populacionais: ParametrosPopulacionaisStub,
    caminho_manifesto: str,
) -> None:
    manifesto = Manifesto(caminho_manifesto)
    combinacoes = expandir_grade(grade_pequena)

    for combinacao in combinacoes:
        rid = run_id({k: v for k, v in combinacao.items() if k != "seed"}, combinacao["seed"])
        manifesto.registrar_pending(rid, combinacao, combinacao["seed"])

    rid_orfao = run_id({k: v for k, v in combinacoes[0].items() if k != "seed"}, combinacoes[0]["seed"])
    manifesto.marcar_running(rid_orfao)

    assert rid_orfao in manifesto.pendentes()

    fake = FakeGeradorParDeClasses()
    orquestrar(grade_pequena, stub_geracao, populacionais, 10, manifesto, fake)

    assert manifesto.pendentes() == []
    assert manifesto.obter(rid_orfao)["status"] == "success"

    params_esperados = {k: v for k, v in combinacoes[0].items() if k != "seed"}
    chamou_a_orfa = any(
        seed == combinacoes[0]["seed"] and all(params.get(k) == v for k, v in params_esperados.items())
        for params, seed, _ in fake.chamadas
    )
    assert chamou_a_orfa
    manifesto.close()


def _linhas_finais(manifesto: Manifesto, run_ids: list[str]) -> dict[str, tuple]:
    """Snapshot comparável de cada `run_id`: status + contagens, ignorando
    campos que legitimamente variam entre execuções (timestamp, e
    `caminho_output`, que agora carrega o PID do worker -- ver
    `FakeGeradorParDeClasses`)."""
    linhas = {}
    for rid in run_ids:
        linha = manifesto.obter(rid)
        linhas[rid] = (linha["status"], linha["n_janelas_ok"], linha["n_janelas_falha"], linha["n_contrato_nao_ativado"])
    return linhas


def test_orquestrar_paralelo_produz_mesmo_resultado_que_serial(
    grade_pequena: GradeFatorial,
    stub_geracao: ParametrosStubGeracao,
    populacionais: ParametrosPopulacionaisStub,
    tmp_path,
) -> None:
    manifesto_serial = Manifesto(str(tmp_path / "serial.sqlite"))
    orquestrar(grade_pequena, stub_geracao, populacionais, 10, manifesto_serial, FakeGeradorParDeClasses())

    manifesto_paralelo = Manifesto(str(tmp_path / "paralelo.sqlite"))
    orquestrar_paralelo(
        grade_pequena, stub_geracao, populacionais, 10, manifesto_paralelo, FakeGeradorParDeClasses(), n_jobs=2
    )

    run_ids_esperados = _run_ids_da_grade(grade_pequena)
    assert manifesto_serial.pendentes() == []
    assert manifesto_paralelo.pendentes() == []
    assert _linhas_finais(manifesto_serial, run_ids_esperados) == _linhas_finais(manifesto_paralelo, run_ids_esperados)

    manifesto_serial.close()
    manifesto_paralelo.close()


def test_orquestrar_paralelo_falha_isolada_nao_derruba_outras(
    grade_pequena: GradeFatorial,
    stub_geracao: ParametrosStubGeracao,
    populacionais: ParametrosPopulacionaisStub,
    caminho_manifesto: str,
) -> None:
    manifesto = Manifesto(caminho_manifesto)
    combinacoes = expandir_grade(grade_pequena)
    combinacao_que_falha = combinacoes[1]
    seed_que_falha = combinacao_que_falha["seed"]
    params_que_falha = {k: v for k, v in combinacao_que_falha.items() if k != "seed"}
    rid_que_falha = run_id(params_que_falha, seed_que_falha)

    fake_com_falha = FakeGeradorParDeClasses(
        falha_predicado=lambda params, seed: all(params.get(k) == v for k, v in params_que_falha.items()) and seed == seed_que_falha
    )
    orquestrar_paralelo(grade_pequena, stub_geracao, populacionais, 10, manifesto, fake_com_falha, n_jobs=2)

    linha_falha = manifesto.obter(rid_que_falha)
    assert linha_falha["status"] == "failed"
    assert "falha simulada" in linha_falha["erro"]

    outros_run_ids = [rid for rid in _run_ids_da_grade(grade_pequena) if rid != rid_que_falha]
    for rid in outros_run_ids:
        assert manifesto.obter(rid)["status"] == "success"
    manifesto.close()


def test_orquestrar_paralelo_usa_processos_separados_de_verdade(
    stub_geracao: ParametrosStubGeracao,
    populacionais: ParametrosPopulacionaisStub,
    caminho_manifesto: str,
) -> None:
    """Prova empírica de que `backend="loky"` roda em processos reais, não
    numa regressão silenciosa para execução serial disfarçada: extrai o PID
    de cada worker via `caminho_output` (o único canal picklable de volta
    ao processo principal -- ver nota em `FakeGeradorParDeClasses`) e
    confirma (a) pelo menos um PID de worker difere do processo de teste, e
    (b) aparece mais de um PID distinto entre as chamadas, provando que
    n_jobs=2 de fato distribuiu o trabalho em vez de colapsar num único
    worker."""
    grade_com_varias_combinacoes = GradeFatorial(
        g=["secao", "municipio", "estado"], delta_t=[2.0], recompensa=[5.0], rho=[0.0, 0.5], beta=[1], seeds=[1]
    )
    manifesto = Manifesto(caminho_manifesto)
    fake = FakeGeradorParDeClasses()

    orquestrar_paralelo(grade_com_varias_combinacoes, stub_geracao, populacionais, 10, manifesto, fake, n_jobs=2)

    run_ids = _run_ids_da_grade(grade_com_varias_combinacoes)
    assert len(run_ids) >= 3

    pid_processo_teste = os.getpid()
    pids_dos_workers = []
    for rid in run_ids:
        linha = manifesto.obter(rid)
        assert linha["status"] == "success"
        pid = int(linha["caminho_output"].rsplit("_pid", 1)[1].removesuffix(".h5"))
        pids_dos_workers.append(pid)

    print(f"\nPID do processo de teste: {pid_processo_teste}")
    print(f"PIDs dos workers (um por combinação): {pids_dos_workers}")

    assert any(pid != pid_processo_teste for pid in pids_dos_workers), (
        "nenhum worker rodou fora do processo de teste -- orquestrar_paralelo pode ter regredido para execução serial"
    )
    assert len(set(pids_dos_workers)) > 1, (
        f"todas as combinações rodaram no mesmo processo ({set(pids_dos_workers)}) -- n_jobs=2 não paralelizou de verdade"
    )
    manifesto.close()


@pytest.mark.parametrize("funcao_orquestracao", [orquestrar, orquestrar_paralelo])
def test_orquestrar_registra_todo_run_id_no_mlflow_sucesso_e_falha(
    funcao_orquestracao,
    grade_pequena: GradeFatorial,
    stub_geracao: ParametrosStubGeracao,
    populacionais: ParametrosPopulacionaisStub,
    caminho_manifesto: str,
) -> None:
    """Verificação ponta-a-ponta pedida no plano: não basta
    `registrar_run_mlflow` funcionar isolada (`test_pipeline_tracking.py`)
    -- confirma que `orquestrar`/`orquestrar_paralelo` de fato a chamam no
    lugar certo, para todo `run_id`, incluindo o que falha. `_tracking_uri_isolado`
    (autouse) já aponta o MLflow para um sqlite em `tmp_path`."""
    manifesto = Manifesto(caminho_manifesto)
    combinacoes = expandir_grade(grade_pequena)
    combinacao_que_falha = combinacoes[1]
    seed_que_falha = combinacao_que_falha["seed"]
    params_que_falha = {k: v for k, v in combinacao_que_falha.items() if k != "seed"}
    rid_que_falha = run_id(params_que_falha, seed_que_falha)

    fake = FakeGeradorParDeClasses(
        falha_predicado=lambda params, seed: all(params.get(k) == v for k, v in params_que_falha.items()) and seed == seed_que_falha
    )
    kwargs = {"n_jobs": 2} if funcao_orquestracao is orquestrar_paralelo else {}
    funcao_orquestracao(grade_pequena, stub_geracao, populacionais, 10, manifesto, fake, **kwargs)

    run_ids_esperados = _run_ids_da_grade(grade_pequena)
    df = mlflow.search_runs()

    assert len(df) == len(run_ids_esperados)
    nomes_registrados = set(df["tags.mlflow.runName"])
    assert nomes_registrados == set(run_ids_esperados)

    status_por_nome = dict(zip(df["tags.mlflow.runName"], df["tags.status"]))
    assert status_por_nome[rid_que_falha] == "failed"
    for rid in run_ids_esperados:
        if rid != rid_que_falha:
            assert status_por_nome[rid] == "success"

    manifesto.close()
