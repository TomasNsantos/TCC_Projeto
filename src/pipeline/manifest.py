"""Manifesto de execução do pipeline: registra progresso em SQLite para permitir
retomar uma execução interrompida sem repetir combinações já concluídas.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

_STATUS_VALIDOS = ("pending", "running", "success", "failed")
_STATUS_IN_SQL = ", ".join(f"'{status}'" for status in _STATUS_VALIDOS)

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS run_id (
    run_id TEXT PRIMARY KEY,
    params_json TEXT NOT NULL,
    seed INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ({_STATUS_IN_SQL})),
    n_janelas_ok INTEGER,
    n_janelas_falha INTEGER,
    n_contrato_nao_ativado INTEGER,
    caminho_output TEXT,
    timestamp TEXT NOT NULL,
    erro TEXT
)
"""


def _agora() -> str:
    return datetime.now(timezone.utc).isoformat()


class Manifesto:
    """Registro de progresso de uma execução do pipeline, sobre um arquivo SQLite.

    NÃO é thread-safe nem process-safe para escrita concorrente: assume
    escrita sempre serial, de um único processo. A paralelização (tarefa
    futura) agrega os resultados de cada worker no processo principal antes
    de escrever aqui — os workers não escrevem no manifesto diretamente.

    O arquivo é reaberto de forma idempotente: construir um ``Manifesto``
    sobre um arquivo já existente preserva as linhas gravadas por uma
    execução anterior (é isso que permite retomar depois de uma
    interrupção).

    Parameters
    ----------
    caminho : str
        Caminho do arquivo SQLite (criado se não existir).
    """

    def __init__(self, caminho: str) -> None:
        self._conexao = sqlite3.connect(caminho)
        self._conexao.execute(_SCHEMA)
        self._conexao.commit()

    def registrar_pending(self, run_id: str, params: dict, seed: int) -> None:
        """Insere ``run_id`` como ``"pending"`` — não-op se já existir.

        ``INSERT OR IGNORE``: idempotente para qualquer status existente,
        não só ``"success"`` — uma linha ``"running"`` órfã (de um processo
        anterior interrompido) também não é tocada, preservando o registro
        de que ela já tinha sido iniciada antes.
        """
        self._conexao.execute(
            "INSERT OR IGNORE INTO run_id (run_id, params_json, seed, status, timestamp) "
            "VALUES (?, ?, ?, 'pending', ?)",
            (run_id, json.dumps(params, sort_keys=True), seed, _agora()),
        )
        self._conexao.commit()

    def marcar_running(self, run_id: str) -> None:
        self._conexao.execute(
            "UPDATE run_id SET status = 'running', timestamp = ? WHERE run_id = ?",
            (_agora(), run_id),
        )
        self._conexao.commit()

    def marcar_success(
        self,
        run_id: str,
        n_janelas_ok: int,
        n_janelas_falha: int,
        n_contrato_nao_ativado: int,
        caminho_output: str,
    ) -> None:
        self._conexao.execute(
            "UPDATE run_id SET status = 'success', n_janelas_ok = ?, n_janelas_falha = ?, "
            "n_contrato_nao_ativado = ?, caminho_output = ?, erro = NULL, timestamp = ? "
            "WHERE run_id = ?",
            (n_janelas_ok, n_janelas_falha, n_contrato_nao_ativado, caminho_output, _agora(), run_id),
        )
        self._conexao.commit()

    def marcar_failed(self, run_id: str, erro: str) -> None:
        self._conexao.execute(
            "UPDATE run_id SET status = 'failed', erro = ?, timestamp = ? WHERE run_id = ?",
            (erro, _agora(), run_id),
        )
        self._conexao.commit()

    def pendentes(self) -> list[str]:
        """``run_id``s com status diferente de ``"success"``.

        Inclui ``"pending"`` (nunca rodou), ``"running"`` (órfão de uma
        execução anterior interrompida) e ``"failed"`` — todos devem ser
        retentados na próxima execução.
        """
        cursor = self._conexao.execute("SELECT run_id FROM run_id WHERE status != 'success'")
        return [linha[0] for linha in cursor.fetchall()]

    def obter(self, run_id: str) -> dict | None:
        """Linha inteira de ``run_id`` como dict, ou ``None`` se não existir."""
        cursor = self._conexao.execute("SELECT * FROM run_id WHERE run_id = ?", (run_id,))
        linha = cursor.fetchone()
        if linha is None:
            return None
        colunas = [descricao[0] for descricao in cursor.description]
        return dict(zip(colunas, linha))

    def close(self) -> None:
        self._conexao.close()
