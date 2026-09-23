import asyncio
import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END


class ReducerState(TypedDict):
    contexts: Annotated[list[str], operator.add]


async def node_a(state: ReducerState) -> dict:
    await asyncio.sleep(1)
    return {
        "contexts": ["A 노드 문장: test1"]
    }


async def node_b(state: ReducerState) -> dict:
    await asyncio.sleep(2)
    return {
        "contexts": ["B 노드 문장: test2"]
    }


def build_graph():
    graph = StateGraph(ReducerState)

    graph.add_node("node_a", node_a)
    graph.add_node("node_b", node_b)

    # START에서 두 Node를 병렬 실행
    graph.add_edge(START, "node_a")
    graph.add_edge(START, "node_b")

    # 두 Node 실행 후 종료
    graph.add_edge("node_a", END)
    graph.add_edge("node_b", END)

    return graph.compile()


sandbox_graph = build_graph()