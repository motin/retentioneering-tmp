from __future__ import annotations

import dataclasses
import json
import os
import random
import string
import warnings
from dataclasses import asdict
from typing import Any, Dict, List, Literal, MutableMapping, MutableSequence, cast

import networkx as nx
import pandas as pd
from IPython.core.display import HTML, display
from nanoid import generate

from retentioneering import RETE_CONFIG
from retentioneering import __version__ as RETE_VERSION
from retentioneering.backend import ServerManager
from retentioneering.backend.tracker import (
    collect_data_performance,
    time_performance,
    track,
    tracker,
)
from retentioneering.edgelist import Edgelist
from retentioneering.eventstream.types import EventstreamType
from retentioneering.nodelist import Nodelist
from retentioneering.nodelist.nodelist import (
    IS_AGGREGATED_COL,
    IS_GROUP_COL,
    IS_HIDDEN_BY_USER_COL,
    NAME_COL,
    PARENT_ID_COL,
)
from retentioneering.templates.transition_graph import TransitionGraphRenderer
from retentioneering.tooling.transition_graph.interface import (
    EdgeId,
    EnvId,
    NodeId,
    NodeLayout,
    RecalculationEdge,
    RecalculationNode,
    RecalculationSuccessResult,
    TargetId,
)
from retentioneering.tooling.typing.transition_graph import (
    GraphSettings,
    LayoutNode,
    NodeParams,
    NormType,
    Position,
    PreparedLink,
    Weight,
)
from retentioneering.tooling.typing.transition_graph.graph_types import (
    ThresholdWithFallback,
)
from retentioneering.utils.dict import clear_dict

from .interface import (
    Column,
    EdgeItem,
    Edges,
    EdgesCustomColors,
    Env,
    NodeItem,
    NodeLayoutDict,
    Nodes,
    NodesCusomColors,
    Normalization,
    RenameRule,
    SerializedEdgesCustomColors,
    SerializedGraph,
    SerializedState,
    Settings,
    SyncStatePayload,
    SyncStateSuccessResponse,
    Target,
    ThresholdValueMap,
    Tracker,
)

# RenameRule = Dict[str, Union[List[str], str]]

SESSION_ID_COL = "session_id"


class TransitionGraph:
    """
    A class that holds methods for transition graph visualization.

    Parameters
    ----------
    eventstream: EventstreamType
        Source eventstream.


    See Also
    --------
    .Eventstream.transition_graph : Call TransitionGraph tool as an eventstream method.
    .Eventstream.transition_matrix : Matrix representation of transition graph.
    .EventstreamSchema : Schema of eventstream columns, that could be used as weights.
    .TransitionGraph.plot : Interactive transition graph visualization.


    Notes
    -----
    See :doc:`transition graph user guide</user_guides/transition_graph>` for the details.

    """

    DEFAULT_GRAPH_URL = "https://static.server.retentioneering.com/package/@rete/transition-graph/version/3/dist/transition-graph.umd.js"
    _weights: MutableMapping[str, str] | None = None
    _edges_norm_type: NormType = None
    _nodes_norm_type: NormType = None
    _nodes_threshold: ThresholdValueMap
    _edges_threshold: ThresholdValueMap
    _recalculation_result: EventstreamType

    sync_data: SyncStatePayload | None = None
    node_layout: dict[str, NodeLayout]
    nodes_weights_range: ThresholdValueMap
    edges_weights_range: ThresholdValueMap
    nodes_custom_colors: NodesCusomColors | None
    edges_custom_colors: EdgesCustomColors | None

    @property
    def graph_url(self) -> str:
        env_url: str = os.getenv("RETE_TRANSITION_GRAPH_URL", "")
        return env_url if env_url else self.DEFAULT_GRAPH_URL

    @property
    def nodes_thresholds(self) -> ThresholdValueMap:
        return self._nodes_threshold

    @nodes_thresholds.setter
    def nodes_thresholds(self, value: ThresholdValueMap) -> None:
        if self._check_thresholds_for_norm_type(value=value, norm_type=self.nodes_norm_type):
            self._nodes_threshold = value

    @property
    def edges_thresholds(self) -> ThresholdValueMap:
        return self._edges_threshold

    @edges_thresholds.setter
    def edges_thresholds(self, value: ThresholdValueMap) -> None:
        if self._check_thresholds_for_norm_type(value=value, norm_type=self.edges_norm_type):
            self._edges_threshold = value

    def _check_thresholds_for_norm_type(self, value: ThresholdValueMap, norm_type: NormType) -> bool:
        values: List[int | float] = [val for key in value.values() for val in key.values()]  # type: ignore

        if norm_type is None:
            if not all(map(lambda x: x is None or x >= 0, values)):  # type: ignore
                raise ValueError(f"For normalization type {norm_type} all thresholds must be positive or None")
        else:
            if not all(map(lambda x: x is None or 0 <= x <= 1, values)):  # type: ignore
                raise ValueError(f"For normalization type {norm_type} all thresholds must be between 0 and 1 or None")

        return True

    @time_performance(
        scope="transition_graph",
        event_name="init",
    )
    def __init__(
        self,
        eventstream: EventstreamType,  # graph: dict,  # preprocessed graph
    ) -> None:
        from retentioneering.eventstream.eventstream import Eventstream

        sm = ServerManager()
        self.env: EnvId = sm.check_env()
        self.server = sm.create_server()

        self.server.register_action("save-graph-settings", lambda n: self._on_graph_settings_request(n))
        self.server.register_action("recalculate", lambda n: self._on_recalc_request(n))
        self.server.register_action("sync-state", lambda n: self._on_sync_state_request(n))

        self.eventstream: Eventstream = eventstream  # type: ignore

        self.event_col = self.eventstream.schema.event_name
        self.event_time_col = self.eventstream.schema.event_timestamp
        self.user_col = self.eventstream.schema.user_id

        self.spring_layout_config = {"k": 0.1, "iterations": 300, "nx_threshold": 1e-4}

        self.is_layout_loaded: bool = False
        self.graph_settings: GraphSettings | dict[str, Any] = {}
        self.render: TransitionGraphRenderer = TransitionGraphRenderer()
        self.normalizations: list[Normalization] = [
            Normalization(id="none", name="none", type="absolute"),
            Normalization(id="full", name="full", type="relative"),
            Normalization(id="node", name="node", type="relative"),
        ]
        self.node_layout = {}
        self.nodes_custom_colors = None
        self.edges_custom_colors = None

        self._recalculation_result = eventstream
        self.allowed_targets = self.__build_targets()

    @property
    @time_performance(
        scope="transition_graph",
        event_name="recalculation_result",
    )
    def recalculation_result(self) -> EventstreamType:
        """
        Export an eventstream after GUI actions that affect eventstream.

        Returns
        -------
        EventstreamType
            The modified event stream.

        Notes
        -----
        Renaming groups, nodes, and nested nodes in the GUI will not affect the resulting eventstream.
        The default group and node names will be returned.
        """
        return self._recalculation_result

    def __build_targets(self) -> list[Target]:
        nice_node = Target(
            id="nice_node",
            name="Positive",
            ignoreThreshold=True,
            edgeDirection="in",
            position="top-right",
        )
        bad_node = Target(
            id="bad_node", name="Negative", ignoreThreshold=True, edgeDirection="in", position="bottom-right"
        )
        source_node = Target(
            id="source_node",
            name="Source",
            ignoreThreshold=False,
            edgeDirection="both",
            position="top-left",
        )

        return [nice_node, bad_node, source_node]

    def _on_sync_state_request(self, sync_data: dict[str, Any]) -> dict:
        self.debug_sync_data = sync_data
        try:
            self.sync_data = SyncStatePayload(**sync_data)
            return asdict(SyncStateSuccessResponse(serverId=self.server.pk, requestId=""))
        except Exception as e:
            raise e  # for now for debugging purposes. Vladimir Makhanov

    def _define_weight_cols(self, custom_weight_cols: list[str] | None) -> list[str]:
        weight_cols = [
            self.eventstream.schema.event_id,
            self.eventstream.schema.user_id,
        ]
        if SESSION_ID_COL in self.eventstream.schema.custom_cols:
            weight_cols.append(SESSION_ID_COL)
        if isinstance(custom_weight_cols, list):
            for col in custom_weight_cols:
                if col not in weight_cols:
                    if col not in self.eventstream.schema.custom_cols:
                        raise ValueError(f"Custom weights column {col} not found in eventstream schema")
                    else:
                        weight_cols.append(col)
        return weight_cols

    @property
    def weights(self) -> MutableMapping[str, str] | None:
        return self._weights

    @weights.setter
    def weights(self, value: MutableMapping[str, str] | None) -> None:
        available_cols = self.__get_nodelist_cols()

        if value and ("edges" not in value or "nodes" not in value):
            raise ValueError("Allowed only: %s" % {"edges": "col_name", "nodes": "col_name"})

        if value and (value["edges"] not in available_cols or value["nodes"] not in available_cols):
            raise ValueError("Allowed only: %s" % {"edges": "col_name", "nodes": "col_name"})

        self._weights = value

    @property
    def edges_norm_type(self) -> NormType:  # type: ignore
        return self._edges_norm_type

    @edges_norm_type.setter
    def edges_norm_type(self, edges_norm_type: NormType) -> None:  # type: ignore
        allowed_edges_norm_types: list[str | None] = [None, "full", "node"]
        if edges_norm_type in allowed_edges_norm_types:
            self._edges_norm_type = edges_norm_type
        else:
            raise ValueError("Norm type should be one of: %s" % allowed_edges_norm_types)

    @property
    def render_edge_norm_type(self) -> NormType:  # type: ignore
        if self.edges_norm_type is None:
            return "none"
        else:
            return self.edges_norm_type

    @property
    def nodes_norm_type(self) -> NormType:  # type: ignore
        return self._nodes_norm_type

    @nodes_norm_type.setter
    def nodes_norm_type(self, nodes_norm_type: NormType) -> None:  # type: ignore
        if nodes_norm_type is not None:
            warnings.warn(f"Currently nodes_norm_type allowed to be None only")
        self._nodes_norm_type = None

    @property
    def nodes_edge_norm_type(self) -> NormType:  # type: ignore
        if self.nodes_norm_type is None:
            return "none"
        else:
            return self.edges_norm_type

    def make_layout_from_serialized_state(self, state: SerializedState) -> dict[str, NodeLayout]:
        node_layout: dict[str, NodeLayout] = {}

        for node in state.nodes["items"]:
            node_id = node.get("id")
            x = node.get("x")
            y = node.get("y")
            if node_id and x is not None and y is not None:
                node_layout[node_id] = NodeLayout(name=node_id, x=int(x), y=int(y))

        return node_layout

    def make_nodelist_from_serialized_state(self, state: SerializedState) -> pd.DataFrame:
        nodelist_data: List[Dict[str, Any]] = []
        weight_cols = self.weight_cols

        for node in state.nodes["items"]:
            new_node: Dict[str, Any] = {}
            new_node[self.event_col] = node.get("id")
            new_node[IS_HIDDEN_BY_USER_COL] = node.get("isHiddenByUser", False)
            new_node[PARENT_ID_COL] = node.get("parentNodeId", None)
            new_node[IS_GROUP_COL] = node.get("isGroup", False)
            new_node[IS_AGGREGATED_COL] = node.get("isAggregated", False)
            name = node.get("name")
            new_node[NAME_COL] = name if name != new_node[self.event_col] else new_node[self.event_col]

            weights: Dict[str, float] = node.get("weight", {})

            for weight_col in weight_cols:  # type: ignore
                weight = weights.get(weight_col, None)
                if weight is not None:
                    new_node[weight_col] = weight

            nodelist_data.append(new_node)

        return pd.DataFrame(nodelist_data)

    def _on_recalc_request(self, recalculate_data: dict[str, Any]) -> dict[str, Any]:
        self.recalculate_data = recalculate_data
        try:
            serialized_state: SerializedState = SerializedState(**recalculate_data)
        except Exception as e:
            raise Exception("Invalid recalculate data")

        try:
            new_nodelist = self.make_nodelist_from_serialized_state(serialized_state)
            self.node_layout = self.make_layout_from_serialized_state(serialized_state)

            self.nodes_thresholds = serialized_state.stateChanges["nodesThreshold"]  # type: ignore
            self.edges_thresholds = serialized_state.stateChanges["edgesThreshold"]  # type: ignore

            self._recalculation_result = self.__apply_nodelist(new_nodelist)

            nodes = self._prepare_nodes(
                nodelist=self.nodelist.nodelist_df,
            )
            edgelist = self.edgelist.get_filtered_edges()
            edgelist["type"] = "suit"
            links = self._prepare_edges(edgelist=edgelist, nodes_set=nodes)

            recalculation_answer = self._build_recalculation_answer(
                serialized_state=serialized_state, nodes=nodes, edges=links
            )

            return asdict(recalculation_answer)
        except Exception as err:
            raise ValueError("error! %s" % err)

    def _edge_id_by_map(self, edge: EdgeItem, mapping: dict[str, EdgeId]) -> EdgeId:
        return mapping.get(f'{[edge["sourceNodeId"], edge["targetNodeId"]]}', edge["id"])

    def _build_recalculation_answer(
        self, serialized_state: SerializedState, nodes: list[NodeItem], edges: list[EdgeItem]
    ) -> RecalculationSuccessResult:
        renamed_nodes = {}
        for node_item in serialized_state.nodes["items"]:
            renamed_nodes[node_item["id"]] = node_item["name"]

        node_edge_map: dict[str, EdgeId] = {}
        node_id_name_mapping: dict[NodeId, str] = {node["id"]: node["name"] for node in serialized_state.nodes["items"]}
        node_name_id_mapping: dict[NodeId, str] = {node["name"]: node["id"] for node in serialized_state.nodes["items"]}
        for edge in serialized_state.edges["items"]:
            node_edge_map[
                f'{[node_id_name_mapping[edge["sourceNodeId"]]]}, {node_id_name_mapping[edge["targetNodeId"]]}'
            ] = edge["id"]

        response_nodes: dict[NodeId, RecalculationNode] = {
            node_name_id_mapping.get(node["id"], node["id"]): RecalculationNode(
                id=node_name_id_mapping.get(node["id"], node["id"]),
                size=node["size"],
                weight=node["weight"],
                targetId=node["targetId"],  # type: ignore
            )
            for node in nodes
        }
        self._response_nodes = response_nodes

        response_edges: list[RecalculationEdge] = []
        for edge in edges:
            edge_actual_id = self._edge_id_by_map(edge=edge, mapping=node_edge_map)
            response_edges.append(
                RecalculationEdge(
                    id=edge_actual_id,
                    sourceNodeId=node_name_id_mapping[renamed_nodes.get(edge["sourceNodeId"], edge["sourceNodeId"])],
                    targetNodeId=node_name_id_mapping[renamed_nodes.get(edge["targetNodeId"], edge["targetNodeId"])],
                    size=edge["size"],
                    weight=edge["weight"],
                )
            )

        res = RecalculationSuccessResult(
            nodes=response_nodes,
            edges=response_edges,
            nodesThresholds=self.nodes_thresholds,
            edgesThresholds=self.edges_thresholds,
            nodesThresholdsMinMax=self.nodelist.get_min_max(),
            edgesThresholdsMinMax=self.edgelist.get_min_max(),
        )
        return res

    def _on_graph_settings_request(self, settings: GraphSettings) -> None:
        self.graph_settings = settings

    def _on_layout_request(self, layout_nodes: MutableSequence[LayoutNode]) -> None:
        self.graph_updates = layout_nodes
        self.node_layout = {x.get("name", None): NodeLayout(**x) for x in layout_nodes}  # type: ignore

    def _make_node_params(
        self, targets: MutableMapping[str, str | None] | None = None
    ) -> MutableMapping[str, str | None] | dict[str, str | None]:
        if targets is not None:
            return self._map_targets(targets)  # type: ignore
        else:
            return self._map_targets(self.targets)  # type: ignore

    def _calc_layout(self, edgelist: pd.DataFrame, width: int, height: int) -> Position:
        G = nx.DiGraph()
        source_col = edgelist.columns[0]
        target_col = edgelist.columns[1]
        weight_col = edgelist.columns[2]

        G.add_weighted_edges_from(edgelist.loc[:, [source_col, target_col, weight_col]].values)

        pos = nx.layout.spring_layout(
            G,
            k=self.spring_layout_config["k"],
            iterations=self.spring_layout_config["iterations"],
            threshold=self.spring_layout_config["nx_threshold"],
            seed=0,
        )

        all_x_coords: list[float] = []
        all_y_coords: list[float] = []

        for j in pos.values():
            all_x_coords.append(j[0])
            all_y_coords.append(j[1])

        min_x = min(all_x_coords)
        min_y = min(all_y_coords)
        max_x = max(all_x_coords)
        max_y = max(all_y_coords)

        pos_new: Position = {
            i: [
                (j[0] - min_x) / (max_x - min_x) * (width - 150) + 75,
                (j[1] - min_y) / (max_y - min_y) * (height - 100) + 50,
            ]
            for i, j in pos.items()
        }
        return pos_new

    def __get_nodelist_cols(self) -> list[str]:
        default_col = self.nodelist_default_col
        custom_cols = self.weight_cols
        return list([default_col]) + list(custom_cols)  # type: ignore

    def _prepare_nodes(
        self,
        nodelist: pd.DataFrame,
        node_params: NodeParams | None = None,
        pos: Position | None = None,
        nodes_custom_colors: NodesCusomColors | None = None,
    ) -> list[NodeItem]:
        node_names: set[str] = set(nodelist[self.event_col])
        nodes_custom_colors = nodes_custom_colors if nodes_custom_colors else {}

        cols = self.__get_nodelist_cols()
        nodes_set: list[NodeItem] = []

        for idx, node_name in enumerate(node_names):
            row = nodelist.loc[nodelist[self.event_col] == node_name]
            custom_color = nodes_custom_colors.get(node_name)
            degree = {}
            weight = {}
            size = {}
            for weight_col in cols:
                max_degree = cast(float, nodelist[weight_col].max())
                r = row[weight_col]
                r = r.tolist()
                value = r[0]
                curr_degree = {}
                curr_degree["degree"] = (abs(value)) / abs(max_degree) * 30 + 4
                curr_degree["source"] = value
                degree[weight_col] = curr_degree
                size[weight_col] = curr_degree["degree"]
                weight[weight_col] = curr_degree["source"]

            node_pos = self.node_layout.get(node_name, None)
            target_id: TargetId = node_params.get(node_name, "suit_node") if node_params is not None else "suit_node"  # type: ignore

            nodelist_row_dict: dict = nodelist[nodelist[self.event_col] == node_name].iloc[0].to_dict()
            children: list[str] = nodelist.loc[nodelist[PARENT_ID_COL] == node_name, self.event_col].tolist()

            node: NodeItem = NodeItem(
                id=node_name,
                size=size,
                weight=weight,
                name=nodelist_row_dict[NAME_COL],
                isHiddenByUser=nodelist_row_dict[IS_HIDDEN_BY_USER_COL],
                parentNodeId=nodelist_row_dict[PARENT_ID_COL],
                isAggregated=nodelist_row_dict[IS_AGGREGATED_COL],
                isGroup=nodelist_row_dict[IS_GROUP_COL],
                children=children,
                targetId=target_id,
            )

            if custom_color is not None:
                node["customColor"] = custom_color

            if node_pos is not None:
                node["x"] = node_pos.x
                node["y"] = node_pos.y

            nodes_set.append(node)

        return nodes_set

    def _prepare_edges(
        self,
        edgelist: pd.DataFrame,
        nodes_set: list[NodeItem],
        edges_custom_colors: EdgesCustomColors | None = None,
    ) -> list[EdgeItem]:
        default_col = self.nodelist_default_col
        source_col = edgelist.columns[0]
        target_col = edgelist.columns[1]
        weight_col = edgelist.columns[2]
        custom_cols: list[str] = self.weight_cols  # type: ignore
        edges: list[EdgeItem] = []
        edges_custom_colors = edges_custom_colors if edges_custom_colors else {}

        edgelist["weight_norm"] = edgelist[weight_col] / edgelist[weight_col].abs().max()
        for _, row in edgelist.iterrows():
            default_col_weight: Weight = {
                "weight_norm": row.weight_norm,
                "weight": cast(float, row[weight_col]),  # type: ignore
            }
            weights = {
                default_col: default_col_weight,
            }
            for custom_weight_col in custom_cols:
                weight = cast(float, row[custom_weight_col])
                max_weight = cast(float, edgelist[custom_weight_col].abs().max())
                weight_norm = weight / max_weight
                col_weight: Weight = {
                    "weight_norm": weight_norm,
                    "weight": weight,
                }
                weights[custom_weight_col] = col_weight

            source_node_name = str(row[source_col])  # type: ignore
            target_node_name = str(row[target_col])  # type: ignore

            custom_color = edges_custom_colors.get((source_node_name, target_node_name))

            # list comprehension faster than filter
            source_node = [node for node in nodes_set if node["id"] == source_node_name][0]
            target_node = [node for node in nodes_set if node["id"] == target_node_name][0]

            if source_node is not None and target_node is not None:  # type: ignore
                edge_item = EdgeItem(
                    id=generate(),
                    sourceNodeId=source_node_name,
                    targetNodeId=target_node_name,
                    weight={col_name: weight["weight"] for col_name, weight in weights.items()},
                    size={col_name: weight["weight_norm"] for col_name, weight in weights.items()},
                    aggregatedEdges=[],
                )

                if custom_color is not None:
                    edge_item["customColor"] = custom_color

                edges.append(edge_item)

        return edges

    def _make_template_data(
        self,
        node_params: NodeParams,
        nodes_custom_colors: NodesCusomColors | None = None,
        edges_custom_colors: EdgesCustomColors | None = None,
    ) -> tuple[list[NodeItem], list[EdgeItem]]:
        edgelist = self.edgelist.get_filtered_edges()
        nodelist = self.nodelist.nodelist_df.copy()

        source_col = edgelist.columns[0]
        target_col = edgelist.columns[1]

        # calc edge type
        edgelist["type"] = edgelist.apply(
            lambda x: node_params.get(x[source_col])  # type: ignore
            if node_params.get(x[source_col]) == "source_node"
            else node_params.get(x[target_col]) or "suit",
            1,  # type: ignore
        )

        nodes = self._prepare_nodes(nodelist=nodelist, node_params=node_params, nodes_custom_colors=nodes_custom_colors)

        links = self._prepare_edges(edgelist=edgelist, nodes_set=nodes, edges_custom_colors=edges_custom_colors)
        return nodes, links

    def _to_json(self, data: Any) -> str:
        return json.dumps(data).encode("latin1").decode("utf-8")

    def _apply_settings(
        self,
        show_weights: bool | None = None,
        show_percents: bool | None = None,
        show_nodes_names: bool | None = None,
        show_all_edges_for_targets: bool | None = None,
        show_nodes_without_links: bool | None = None,
        show_edges_info_on_hover: bool | None = None,
    ) -> dict[str, Any]:
        settings = {
            "show_weights": show_weights,
            "show_percents": show_percents,
            "show_nodes_names": show_nodes_names,
            "show_all_edges_for_targets": show_all_edges_for_targets,
            "show_nodes_without_links": show_nodes_without_links,
            "show_edges_info_on_hover": show_edges_info_on_hover,
        }
        # @FIXME: idk why pyright doesn't like this. Vladimir Makhanov
        merged = {**self.graph_settings, **clear_dict(settings)}  # type: ignore

        return clear_dict(merged)

    def _map_targets(self, targets: dict[str, str | list[str]]) -> dict[str, str]:
        targets_mapping = {
            "positive": "nice_node",
            "negative": "bad_node",
            "source": "source_node",
        }
        mapped_targets = {}

        for target, nodes in targets.items():
            if nodes is None:  # type: ignore
                pass
            if isinstance(nodes, list):
                for node in nodes:
                    mapped_targets[node] = targets_mapping[target]
            else:
                mapped_targets[nodes] = targets_mapping[target]

        return mapped_targets

    @staticmethod
    def generateId(size: int = 6, chars: str = string.ascii_uppercase + string.digits) -> str:
        return "el" + "".join(random.choice(chars) for _ in range(size))

    def save_to_file(self, filename: str) -> None:
        serialized_graph = self.serialize()

        with open(filename, "w") as file:
            file.write(json.dumps(serialized_graph))

    def __load_from_file(self, filename: str) -> SerializedGraph | None:
        try:
            with open(filename, "r") as f:
                layout_data: SerializedGraph = json.load(f)
                return layout_data
        except Exception:
            warnings.warn(f"Failed to load graph dump!")
            return None

    def serialize(self) -> SerializedGraph:
        return {
            "nodelist": self.nodelist.to_dict(),
            "layout": self.serialize_node_layout(),
            "nodes_thresholds": self.nodes_thresholds,
            "edges_thresholds": self.edges_thresholds,
            "nodes_norm_type": self.nodes_norm_type,
            "edges_norm_type": self.edges_norm_type,
            "edges_weight_col": self.edges_weight_col,
            "nodes_weight_col": self.nodes_weight_col,
            "targets": self.targets,
            "weight_cols": self.weight_cols,  # type: ignore
            "nodes_custom_colors": self.nodes_custom_colors,
            "edges_custom_colors": self.serialize_edges_custom_colors(),
        }

    def deserialize_edges_custom_colors(
        self, serialized_edges_custom_colors: SerializedEdgesCustomColors | None
    ) -> EdgesCustomColors | None:
        if serialized_edges_custom_colors is None:
            return None

        result: EdgesCustomColors = {}

        for edge in serialized_edges_custom_colors:
            result[(edge["source"], edge["target"])] = edge["color"]

        return result

    def serialize_edges_custom_colors(self) -> SerializedEdgesCustomColors | None:
        if self.edges_custom_colors is None:
            return None

        result: SerializedEdgesCustomColors = []

        for edge, color in self.edges_custom_colors.items():
            result.append({"source": edge[0], "target": edge[1], "color": color})

        return result

    def serialize_node_layout(self) -> dict[str, NodeLayoutDict]:
        node_layout_dict: dict[str, NodeLayoutDict] = {}

        for key, value in self.node_layout.items():
            node_layout_dict[key] = dataclasses.asdict(value)  # type: ignore

        return node_layout_dict

    def deserialize_node_layout(self, serialized_layout: dict[str, NodeLayoutDict]) -> dict[str, NodeLayout]:
        node_layout: dict[str, NodeLayout] = {}

        for key, value in serialized_layout.items():
            x: float | int = value.get("x")  # type: ignore
            y: float | int = value.get("y")  # type: ignore

            node_layout[key] = NodeLayout(name=key, x=int(x), y=int(y))

        return node_layout

    def _load_layout_from_dump(self, layout_dump: str) -> dict[str, NodeLayout]:
        # load layout from json file
        with open(layout_dump, "r") as f:
            layout_data: dict[str, NodeLayoutDict] = json.load(f)

        return self.deserialize_node_layout(layout_data)

    @time_performance(
        scope="transition_graph",
        event_name="plot",
    )
    def plot(
        self,
        targets: MutableMapping[str, str | None] | None = None,
        edges_norm_type: NormType | None = None,
        nodes_threshold: ThresholdWithFallback | None = None,
        nodes_norm_type: NormType | None = None,
        edges_threshold: ThresholdWithFallback | None = None,
        nodes_weight_col: str | None = None,
        edges_weight_col: str | None = None,
        custom_weight_cols: list[str] | None = None,
        width: str | int | float = "100%",
        height: str | int | float = "60vh",
        show_weights: bool = True,
        show_percents: bool = False,
        show_nodes_names: bool = True,
        show_all_edges_for_targets: bool = True,
        show_nodes_without_links: bool = False,
        show_edge_info_on_hover: bool = True,
        layout_file: str | None = None,
        nodes_custom_colors: NodesCusomColors | None = None,
        edges_custom_colors: EdgesCustomColors | None = None,
        nodelist: Nodelist | pd.DataFrame | None = None,
        import_file: str | None = None,
    ) -> None:
        """
        Create interactive transition graph visualization with callback to sourcing eventstream.

        Parameters
        ----------
        edges_norm_type : {"full", "node", None}, default None
            Type of normalization that is used to calculate weights for graph edges.
            Based on ``edges_weight_col`` parameter the weight values are calculated.

            - If ``None``, normalization is not used, the absolute values are taken.
            - If ``full``, normalization across the whole eventstream.
            - If ``node``, normalization across each node (or outgoing transitions from each node).

            See :ref:`Transition graph user guide <transition_graph_weights>` for the details.

        nodes_norm_type : {"full", "node", None}, default None
            Currently not implemented. Always None.

        edges_weight_col : str, optional
            A column name from the :py:class:`.EventstreamSchema` which values will control the final
            edges' weights and displayed width as well.

            For each edge is calculated:

            - If ``None`` or ``user_id`` - the number of unique users.
            - If ``event_id`` - the number of transitions.
            - If ``session_id`` - the number of unique sessions.
            - If ``custom_col`` - the number of unique values in selected column.

            See :ref:`Transition graph user guide <transition_graph_weights>` for the details.

        edges_threshold : dict, optional
            Threshold mapping that defines the minimal weights for edges displayed on the canvas.

            - Keys should be of type str and contain the weight column names (the values from the
              :py:class:`.EventstreamSchema`).
            - Values of the dict are the thresholds for the edges that will be displayed.

            Support multiple weighting columns. In that case, logical OR will be applied.
            Edges with value less than at least one of thresholds will be hidden.
            Example: {'event_id': 100, user_id: 50}.

            See :ref:`Transition graph user guide<transition_graph_thresholds>` for the details.

        nodes_weight_col : str, optional
            A column name from the :py:class:`.EventstreamSchema` which values control the final
            nodes' weights and displayed diameter as well.

            For each node is calculated:

            - If ``None`` or ``user_id`` - the number of unique users.
            - If ``event_id`` - the number of events.
            - If ``session_id`` - the number of unique sessions.
            - If ``custom_col`` - the number of unique values in selected column.

            See :ref:`Transition graph user guide <transition_graph_weights>` for the details.

        nodes_threshold : dict, optional
            Threshold mapping that defines the minimal weights for nodes displayed on the canvas.

            - Keys should be of type str and contain the weight column names (the values from the
              :py:class:`.EventstreamSchema`).
            - Values of the dict are the thresholds for the nodes that will be displayed.
              They should be of type int or float.

            Support multiple weighting columns. In that case, logical OR will be applied.
            Nodes with value less than at least one of thresholds will be hidden.
            Example: {'event_id': 100, user_id: 50}.

            See :ref:`Transition graph user guide<transition_graph_thresholds>` for the details.

        custom_weight_cols : list of str, optional
            Custom columns from the :py:class:`.EventstreamSchema` that can be selected in ``edges_weight_col``
            and ``nodes_weight_col`` parameters. If ``session_col=session_id`` exists,
            it is added by default to this list.

        targets : dict, optional
            Events mapping that defines which nodes and edges should be colored for better visualization.

            - Possible keys: "positive" (green), "negative" (red), "source" (orange).
            - Possible values: list of events of a given type.

            See :ref:`Transition graph user guide<transition_graph_color_settings>` for the details.

        nodes_custom_colors : dict, optional
            Set nodes color explicitly. The dict keys are node names, the values are the corresponding colors.
            A color can be defined either as an HTML standard color name or a HEX code.
            See :ref:`Transition graph user guide<transition_graph_color_settings>` for the details.

        edges_custom_colors : dict, optional
            Set edges color explicitly. The dict keys are tuples of length 2, e.g. (path_start', 'catalog'),
            the values are the corresponding colors.
            A color can be defined either as an HTML standard color name or a HEX code.
            See :ref:`Transition graph user guide<transition_graph_color_settings>` for the details.

        width : str, int or float, default "100%"
            The width of the plot can be specified in the following ways:

            - In pixels (int or float);
            - In other CSS units (str). For example, the default value of "100%" means the plot will occupy 100%
              of the width of the Jupyter Notebook cell.
        height : str, int or float, default "60vh"
            The height of the plot can be specified as follows:

            - In pixels (int or float);
            - In other CSS units (str). For example, the default value of "60vh" means the plot will occupy 60%
              of the height of the browser window.

            The resulting height can't be lower than 600px.
        show_weights : bool, default True
            Hide/display the edge weight labels. By default, weights are shown.
        show_percents : bool, default False
            Display edge weights as percents. Available only if an edge normalization type is chosen.
            By default, weights are displayed in fractions.
        show_nodes_names : bool, default True
            Hide/display the node names. By default, names are shown.
        show_all_edges_for_targets : bool, default True
            This displaying option allows to ignore the threshold filters and always display
            any edge connected to a target node. By default, all such edges are shown.
        show_nodes_without_links : bool, default False
            Setting a threshold filter might remove all the edges connected to a node.
            Such isolated nodes might be considered as useless. This displaying option
            hides them in the canvas as well.
        show_edge_info_on_hover : bool, default True
            This parameter determines whether information about an edge (weight, source node, target node)
            is displayed when hovering the mouse over it.
        nodelist : pd.DataFrame, default None
            A DataFrame containing information about nodes, such as weights, parents, etc.
        layout_file : str, default None
            A string path to a JSON file containing the configuration for nodes layout.
        import_file : str, default None
            A string path to a JSON file containing complete dump of a graph.

        Returns
        -------
            Rendered IFrame graph.

        Notes
        -----
        1. If all the edges connected to a node are hidden, the node becomes hidden as well.
           In order to avoid it - use ``show_nodes_without_links=True`` parameter in code or in the interface.
        2. The thresholds may use their own weighting columns both for nodes and for edges independently
           of weighting columns defined in ``edges_weight_col`` and ``nodes_weight_col`` arguments.

        See :doc:`TransitionGraph user guide </user_guides/transition_graph>` for the details.
        """
        if edges_norm_type is None and show_percents:
            raise ValueError("If show_percents=True, edges_norm_type should be 'full' or 'node'!")

        called_params = {
            "edges_norm_type": edges_norm_type,
            "nodes_norm_type": nodes_norm_type,
            "targets": targets,
            "nodes_threshold": nodes_threshold,
            "edges_threshold": edges_threshold,
            "nodes_weight_col": nodes_weight_col,
            "edges_weight_col": edges_weight_col,
            "custom_weight_cols": custom_weight_cols,
            "width": width,
            "height": height,
            "show_weights": show_weights,
            "show_percents": show_percents,
            "show_nodes_names": show_nodes_names,
            "show_all_edges_for_targets": show_all_edges_for_targets,
            "show_nodes_without_links": show_nodes_without_links,
            "show_edge_info_on_hover": show_edge_info_on_hover,
            "layout_dump": layout_file,
        }
        not_hash_values = ["edges_norm_type", "targets", "width", "height"]

        graph_dump = self.__load_from_file(import_file) if import_file else None

        if layout_file is not None:
            try:
                self.node_layout = self._load_layout_from_dump(layout_file)
                self.is_layout_loaded = True
            except Exception:
                warnings.warn(f"Failed to load layout dump")
                self.is_layout_loaded = False
        elif graph_dump is not None:
            self.node_layout = self.deserialize_node_layout(graph_dump["layout"])
            self.is_layout_loaded = True

        self.__prepare_graph_for_plot(
            edges_weight_col=edges_weight_col,
            nodes_threshold=nodes_threshold,
            edges_threshold=edges_threshold,
            edges_norm_type=edges_norm_type,
            nodes_norm_type=nodes_norm_type,
            nodes_weight_col=nodes_weight_col,
            targets=targets,
            custom_weight_cols=custom_weight_cols,
            nodes_custom_colors=nodes_custom_colors,
            edges_custom_colors=edges_custom_colors,
            graph_dump=graph_dump,
        )

        external_nodelist_df: pd.DataFrame | None = None

        if nodelist is not None:
            external_nodelist_df = nodelist if isinstance(nodelist, pd.DataFrame) else nodelist.nodelist_df
        elif graph_dump is not None:
            external_nodelist_df = pd.DataFrame(graph_dump["nodelist"])

        self.__apply_nodelist(nodelist_df=external_nodelist_df)

        node_params = self._make_node_params(targets)

        nodes, links = self._make_template_data(
            node_params=node_params,
            nodes_custom_colors=self.nodes_custom_colors,
            edges_custom_colors=self.edges_custom_colors,
        )

        prepared_nodes = self._prepare_nodes_for_plot(node_list=nodes)

        env = Env(
            id=self.env,
            serverId=self.server.pk,
            kernelId=self.server.kernel_id,
            kernelName="",
            libVersion=RETE_VERSION,
        )
        tracker = Tracker(
            hwid=RETE_CONFIG.user.pk,
            scope="transition_graph",
            eventstreamIndex=self.eventstream._eventstream_index,
        )
        edges = Edges(
            items=links,
            normalizations=self.normalizations,
            selectedNormalizationId=self.render_edge_norm_type,
            columns=prepared_nodes["columns"],
            threshold=self.edges_thresholds,
            weightsRange=self.edges_weights_range,
            selectedThresholdColumnId=self.edges_weight_col,
            selectedWeightsColumnId=self.edges_weight_col,
        )
        settings = Settings(
            showEdgesWeightsOnCanvas=show_weights,
            convertWeightsToPercents=show_percents,
            doNotFilterTargetNodes=show_all_edges_for_targets,
            showEdgesInfoOnHover=show_edge_info_on_hover,
            showNodesNamesOnCanvas=show_nodes_names,
            showNodesWithoutEdges=show_nodes_without_links,
        )

        init_params = SerializedState(
            env=env,
            tracker=tracker,
            useLayoutDump=self.is_layout_loaded,
            nodes=prepared_nodes,
            edges=edges,
            settings=settings,
        )

        widget_id = self.generateId()

        valid_width: str = f"{width}px" if isinstance(width, (int, float)) else width
        valid_height: str = f"{height}px" if isinstance(height, (int, float)) else height
        display(
            HTML(
                self.render.show(
                    widget_id=widget_id,
                    script_url=f"{self.graph_url}?id={widget_id}",
                    style=f"width: 100%; width: {valid_width}; height: 60vh; height: {valid_height}; min-height: 600px; box-sizing: border-box;",
                    state=json.dumps(asdict(init_params)),
                )
            )
        )
        collect_data_performance(
            scope="transition_graph",
            event_name="metadata",
            called_params=called_params,
            not_hash_values=not_hash_values,
            performance_data={"unique_nodes": len(nodes), "unique_links": len(links)},
            eventstream_index=self.eventstream._eventstream_index,
        )

    def _prepare_nodes_for_plot(self, node_list: list[NodeItem]) -> Nodes:
        columns = [Column(id=col, name=col) for col in self.weight_cols]  # type: ignore
        nodes = Nodes(
            normalizations=self.normalizations,
            selectedNormalizationId=self.render_edge_norm_type,
            items=node_list,
            columns=columns,
            threshold=self.nodes_thresholds,
            weightsRange=self.nodes_weights_range,
            selectedThresholdColumnId=self.nodes_weight_col,
            selectedWeightsColumnId=self.nodes_weight_col,
            targets=self.allowed_targets,
            defaultColumnId=self.nodelist_default_col,
            sortField="name",
            sortOrder="asc",
        )
        return nodes

    def __prepare_graph_for_plot(
        self,
        edges_weight_col: str | None = None,
        edges_threshold: ThresholdWithFallback | None = None,
        nodes_weight_col: str | None = None,
        nodes_threshold: ThresholdWithFallback | None = None,
        edges_norm_type: NormType | None = None,
        nodes_norm_type: NormType | None = None,
        targets: MutableMapping[str, str | None] | None = None,
        custom_weight_cols: list[str] | None = None,
        nodes_custom_colors: NodesCusomColors | None = None,
        edges_custom_colors: EdgesCustomColors | None = None,
        graph_dump: SerializedGraph | None = None,
    ) -> None:
        if targets:
            self.targets = targets
        elif graph_dump is not None:
            self.targets = graph_dump["targets"]  # type: ignore
        else:
            self.targets = {"positive": None, "negative": None, "source": None}

        self.edges_norm_type = edges_norm_type
        self.nodes_norm_type = nodes_norm_type

        if self.edges_norm_type is None and graph_dump is not None:
            self.edges_norm_type = graph_dump.get("edges_norm_type")

        if self.nodes_norm_type is None and graph_dump is not None:
            self.nodes_norm_type = graph_dump.get("nodes_norm_type")

        self.nodelist_default_col = self.eventstream.schema.event_id
        self.edgelist_default_col = self.eventstream.schema.event_id

        self.weight_cols = (
            graph_dump.get("weight_cols") if graph_dump is not None else self._define_weight_cols(custom_weight_cols)
        )

        if nodes_weight_col:
            self.nodes_weight_col = nodes_weight_col
        elif graph_dump is not None:
            self.nodes_weight_col = graph_dump["nodes_weight_col"]
        else:
            self.nodes_weight_col = self.eventstream.schema.user_id

        if edges_weight_col:
            self.edges_weight_col = edges_weight_col
        elif graph_dump is not None:
            self.edges_weight_col = graph_dump["edges_weight_col"]
        else:
            self.edges_weight_col = self.eventstream.schema.user_id

        if nodes_custom_colors:
            self.nodes_custom_colors = nodes_custom_colors
        elif graph_dump is not None:
            self.nodes_custom_colors = graph_dump["nodes_custom_colors"]
        else:
            self.nodes_custom_colors = None

        if edges_custom_colors:
            self.edges_custom_colors = edges_custom_colors
        elif graph_dump is not None:
            self.edges_custom_colors = self.deserialize_edges_custom_colors(graph_dump["edges_custom_colors"])
        else:
            self.edges_custom_colors = None

        # calculate nodelist & edgelist on initial eventstream without any thresholds
        self.initial_nodelist: Nodelist = Nodelist(
            weight_cols=self.weight_cols,
            time_col=self.event_time_col,
            event_col=self.event_col,
        )
        self.initial_nodelist.calculate_nodelist(data=self.eventstream.to_dataframe())

        self.initial_edgelist: Edgelist = Edgelist(eventstream=self.eventstream)
        self.initial_edgelist.calculate_edgelist(
            weight_cols=self.weight_cols,  # type: ignore
            norm_type=self.edges_norm_type,
        )

        self.nodes_weights_range = self.initial_nodelist.get_min_max()
        self.edges_weights_range = self.initial_edgelist.get_min_max()

        if nodes_threshold is None:
            if graph_dump is not None:
                self.nodes_thresholds = graph_dump["nodes_thresholds"]
            else:
                self.nodes_thresholds = {}
        else:
            self.nodes_thresholds = self.__threshold_fallback(nodes_threshold, threshold_type="nodes")

        if edges_threshold is None:
            if graph_dump is not None:
                self.edges_thresholds = graph_dump["edges_thresholds"]
            else:
                self.edges_thresholds = {}
        else:
            self.edges_thresholds = self.__threshold_fallback(edges_threshold, threshold_type="edges")

    def __apply_nodelist(self, nodelist_df: pd.DataFrame | None = None) -> EventstreamType:
        prev_nodes_min_max = self.nodelist.get_min_max() if hasattr(self, "nodelist") and self.nodelist is not None else None  # type: ignore
        prev_edges_min_max = self.edgelist.get_min_max() if hasattr(self, "edgelist") and self.edgelist is not None else None  # type: ignore

        self.nodelist: Nodelist = self.initial_nodelist.copy()
        if nodelist_df is not None:
            self.nodelist.update(nodelist_df)

        curr_stream = self.eventstream

        hidden_nodes = self.nodelist.get_hidden_by_user_nodes()
        if len(hidden_nodes) > 0:
            curr_stream = curr_stream.filter_events(func=lambda df, schema: ~df[schema.event_name].isin(hidden_nodes))  # type: ignore

        rename_rules = self.nodelist.groups_to_rename_rules()
        if len(rename_rules) > 0:
            curr_stream = curr_stream.rename(rules=rename_rules)  # type: ignore

        self.nodelist.calculate_nodelist(data=curr_stream.to_dataframe())

        if prev_nodes_min_max is not None:
            self.nodes_thresholds = self.nodelist.fit_threshold(
                threshold=self.nodes_thresholds, prev_min_max=prev_nodes_min_max
            )

        self.nodelist.update_threshold(nodes_thresholds=self.nodes_thresholds)

        out_of_threshold_nodes = self.nodelist.get_out_of_threshold_nodes(only_ungrouped=True)
        if len(out_of_threshold_nodes) > 0:
            curr_stream = curr_stream.filter_events(func=lambda df, schema: ~df[schema.event_name].isin(out_of_threshold_nodes))  # type: ignore

        self.edgelist: Edgelist = Edgelist(eventstream=curr_stream)
        self.edgelist.calculate_edgelist(
            weight_cols=self.weight_cols,  # type: ignore
            norm_type=self.edges_norm_type,
        )

        if prev_edges_min_max is not None:
            self.edges_thresholds = self.edgelist.fit_threshold(
                threshold=self.edges_thresholds, prev_min_max=prev_edges_min_max
            )

        self.edgelist.update_threshold(edges_threshold=self.edges_thresholds)

        # use aliases after all calculations
        aliases_rename_rules = self.nodelist.renamed_events_to_rename_rules()
        if len(aliases_rename_rules) > 0:
            curr_stream = curr_stream.rename(rules=aliases_rename_rules)  # type: ignore

        return curr_stream

    def __threshold_fallback(
        self, threshold: ThresholdWithFallback, threshold_type: Literal["nodes"] | Literal["edges"]
    ) -> ThresholdValueMap:
        result: ThresholdValueMap = {}

        for key, value in threshold.items():
            weights_range = (
                self.nodes_weights_range.get(key, None)
                if threshold_type == "nodes"
                else self.edges_weights_range.get(key, None)
            )

            if weights_range is None:
                raise ValueError(f"threshold is invalid. Column '{key}' doesn't found!")

            if isinstance(value, dict):
                result[key] = value
                continue

            result[key] = {"min": float(value), "max": float(weights_range["max"])}

        return result
