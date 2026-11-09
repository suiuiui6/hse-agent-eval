"""Deterministic HTTP fixtures for evaluation dependencies."""

from hse_agent_evals.fixtures.graphrag_server import GraphRAGFixtureServer
from hse_agent_evals.fixtures.model_proxy import ModelProxyServer

__all__ = ["GraphRAGFixtureServer", "ModelProxyServer"]
