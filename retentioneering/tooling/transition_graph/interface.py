from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, MutableMapping, Tuple, TypedDict, Union

from typing_extensions import NotRequired, TypeAlias

NodeType: TypeAlias = str
NodeId: TypeAlias = str
EdgeId: TypeAlias = str
EdgesGroupId: TypeAlias = str
ColumnId: TypeAlias = str
TargetId: TypeAlias = str
NormalizationId = Union[Literal["none", "full", "node"], None]
NormalizationType = Literal["absolute", "relative"]
NumericColumnMap = Dict[ColumnId, float]
NodesSortField = Literal["name", "weight", "custom"]
SortOrder = Literal["asc", "desc", "custom"]
EnvId = Literal["classic", "colab"]
RenameRule = Dict[str, Union[List[str], str]]


class ThresholdValue(TypedDict):
    min: float | int
    max: float | int


ThresholdValueMap = Dict[ColumnId, ThresholdValue]


class Target(TypedDict):
    id: TargetId
    name: str
    ignoreThreshold: NotRequired[bool]
    edgeDirection: NotRequired[Literal["in", "out", "both"]]
    position: NotRequired[Literal["top-right", "bottom-right", "top-left", "bottom-left"]]
    description: NotRequired[str]


class NodeItem(TypedDict):
    id: str
    name: str
    isHiddenByUser: NotRequired[bool]  # NOSONAR
    visibilityRule: NotRequired[int]  # NOSONAR
    size: NumericColumnMap
    weight: NumericColumnMap
    isGroup: NotRequired[bool]  # NOSONAR
    isAggregated: NotRequired[bool]  # NOSONAR
    children: list[str]
    parentNodeId: NotRequired[str]  # NOSONAR
    description: NotRequired[str]
    x: NotRequired[float]
    y: NotRequired[float]
    targetId: NotRequired[TargetId]
    customColor: NotRequired[str]


class EdgeItem(TypedDict):
    id: EdgeId
    sourceNodeId: NodeId  # NOSONAR
    targetNodeId: NodeId  # NOSONAR
    size: NumericColumnMap
    weight: NumericColumnMap
    aggregatedEdges: list[str]  # NOSONAR
    description: NotRequired[str]
    customColor: NotRequired[str]


class Column(TypedDict):
    id: str
    name: str
    isDefault: NotRequired[bool]  # NOSONAR
    description: NotRequired[str]


class Normalization(TypedDict):
    id: NormalizationId
    name: str
    type: NormalizationType
    description: NotRequired[str]


NodesCusomColors = Dict[NodeId, str]
EdgesCustomColors = Dict[Tuple[NodeId, NodeId], str]


@dataclass
class Settings:
    showEdgesWeightsOnCanvas: bool = True  # NOSONAR
    convertWeightsToPercents: bool = False  # NOSONAR
    doNotFilterTargetNodes: bool = True  # NOSONAR
    showEdgesInfoOnHover: bool = True  # NOSONAR
    showNodesNamesOnCanvas: bool = True  # NOSONAR
    showNodesWithoutEdges: bool = False  # NOSONAR
    openSidebarByDefault: bool = True  # NOSONAR


@dataclass
class Env:
    id: EnvId | None = None
    serverId: str | None = None  # NOSONAR
    kernelId: str | None = None  # NOSONAR
    kernelName: str | None = None  # NOSONAR
    libVersion: str | None = None  # NOSONAR


@dataclass
class Tracker:
    eventstreamIndex: int
    scope: str
    hwid: str | None = None


class Nodes(TypedDict):
    normalizations: list[Normalization]
    selectedNormalizationId: NormalizationId  # NOSONAR
    columns: list[Column]
    defaultColumnId: ColumnId
    threshold: ThresholdValueMap
    weightsRange: ThresholdValueMap
    selectedThresholdColumnId: ColumnId  # NOSONAR
    selectedWeightsColumnId: ColumnId  # NOSONAR
    items: list[NodeItem]
    targets: list[Target]  # NOSONAR
    sortField: NodesSortField
    sortOrder: SortOrder


class EdgesInitialStateMutated(TypedDict):
    hiddenNodes: list[NodeId]
    items: list[EdgeItem]


class EdgesInitialState(TypedDict):
    defaultItems: list[EdgeItem]
    mutated: EdgesInitialStateMutated


class Edges(TypedDict):
    normalizations: list[Normalization]
    selectedNormalizationId: NormalizationId  # NOSONAR
    columns: list[Column]
    threshold: ThresholdValueMap
    weightsRange: ThresholdValueMap
    selectedThresholdColumnId: ColumnId  # NOSONAR
    selectedWeightsColumnId: ColumnId  # NOSONAR
    items: list[EdgeItem]
    initialState: NotRequired[EdgesInitialState]  # NOSONAR


@dataclass
class InitializationParams:
    env: Env
    tracker: Tracker
    nodes: Nodes
    edges: Edges
    settings: Settings
    recalculationChanges: dict | None = None  # NOSONAR
    useLayoutDump: bool | None = None  # NOSONAR


@dataclass
class NodeLayout:
    name: str
    x: int
    y: int


class NodeLayoutDict(TypedDict):
    name: str
    x: int
    y: int


@dataclass
class Request:
    serverId: str
    requestId: str
    requestType: str | None
    method: str


@dataclass
class RequestWithPayload(Request):
    payload: dict


@dataclass
class ResponseErrorData:
    code: str = ""
    message: str = ""


@dataclass
class Response:
    serverId: str
    requestId: str
    method: str


@dataclass
class SuccessResponseData:
    data: dict = field(default_factory=dict)
    isSuccess: bool = True


@dataclass
class SuccessResponse(Response):
    result: SuccessResponseData | RecalculationSuccessResult | SyncStateResult = field(
        default_factory=SuccessResponseData
    )


@dataclass
class FailResponseData:
    data: ResponseErrorData = field(default_factory=ResponseErrorData)
    isSuccess: bool = False


@dataclass
class FailResponse(Response):
    result: FailResponseData = field(default_factory=FailResponseData)


@dataclass
class SyncStateNodes:
    sortField: NodesSortField
    sortOrder: SortOrder
    selectedNormalizationId: NormalizationId
    selectedWeightsColumnId: ColumnId
    threshold: ThresholdValueMap
    selectedThresholdColumnId: ColumnId
    items: list[NodeItem]


@dataclass
class SyncStateEdges:
    selectedNormalizationId: NormalizationId
    selectedWeightsColumnId: ColumnId
    threshold: ThresholdValueMap
    selectedThresholdColumnId: ColumnId
    items: list[EdgeItem]
    initialItems: list[EdgeItem]


@dataclass
class SyncStatePayload:
    nodes: SyncStateNodes
    edges: SyncStateEdges
    settings: Settings
    recalculationChanges: dict


@dataclass
class SyncStateResult:
    isSuccess: bool = True


@dataclass
class SyncStateSuccessResponse(SuccessResponse):
    result: SyncStateResult = field(default_factory=SyncStateResult)
    method: str = "sync-state"


@dataclass
class RecalculationNode:
    id: NodeId
    size: NumericColumnMap
    weight: NumericColumnMap
    targetId: TargetId | None


@dataclass
class RecalculationEdge:
    id: EdgeId
    sourceNodeId: NodeId
    targetNodeId: NodeId
    size: NumericColumnMap
    weight: NumericColumnMap


@dataclass
class RecalculationSuccessResult:
    nodes: dict[NodeId, RecalculationNode] = field(default_factory=dict)
    edges: list[RecalculationEdge] = field(default_factory=list)
    nodesThresholds: ThresholdValueMap = field(default_factory=dict)
    edgesThresholds: ThresholdValueMap = field(default_factory=dict)
    nodesThresholdsMinMax: ThresholdValueMap = field(default_factory=dict)
    edgesThresholdsMinMax: ThresholdValueMap = field(default_factory=dict)


@dataclass
class RecalculationSuccessData:
    data: RecalculationSuccessResult = field(default_factory=RecalculationSuccessResult)
    isSuccess: bool = True


@dataclass
class RecalculationSuccessResponse(SuccessResponse):
    result: RecalculationSuccessResult = field(default_factory=RecalculationSuccessResult)
    method: str = "recalculate"


@dataclass
class NodeAggregationData:
    ids: list[NodeId]
    weight: NumericColumnMap


NodeAggregation: TypeAlias = Dict[NodeId, NodeAggregationData]


@dataclass
class EdgeAggregationData:
    ids: list[EdgeId]
    weight: NumericColumnMap
    size: NumericColumnMap


EdgeAggregation: TypeAlias = Dict[EdgesGroupId, EdgeAggregationData]


@dataclass
class StateChanges:
    hiddenNodes: list[NodeId] = field(default_factory=list)
    nodesThreshold: ThresholdValueMap = field(default_factory=dict)
    edgesThreshold: ThresholdValueMap = field(default_factory=dict)
    nodesAggregation: list[NodeAggregation] = field(default_factory=list)
    edgesAggregation: list[EdgeAggregation] = field(default_factory=list)


@dataclass
class SerializedState:
    env: Env
    tracker: Tracker
    nodes: Nodes
    edges: Edges
    settings: Settings
    recalculationChanges: StateChanges = field(default_factory=StateChanges)
    stateChanges: StateChanges = field(default_factory=StateChanges)
    useLayoutDump: bool | None = None  # NOSONAR


@dataclass
class RecalculationRequest:
    server_id: str
    request_id: str
    payload: SerializedState
    method: str = "recalculate"


@dataclass
class RecalculationResponse:
    server_id: str
    request_id: str
    result: RecalculationSuccessResult
    method: str = "recalculate"


SerializedNodelist = List[Dict[str, Any]]


class SerializedEdgeColor(TypedDict):
    source: str
    target: str
    color: str


SerializedEdgesCustomColors = List[SerializedEdgeColor]


class SerializedGraph(TypedDict):
    nodelist: SerializedNodelist
    layout: dict[str, NodeLayoutDict]
    nodes_thresholds: ThresholdValueMap
    edges_thresholds: ThresholdValueMap
    nodes_norm_type: Union[Literal["full", "node", "none"], None]
    edges_norm_type: Union[Literal["full", "node", "none"], None]
    nodes_weight_col: str
    edges_weight_col: str
    targets: MutableMapping[str, str | None] | dict[str, None]
    weight_cols: list[str]
    nodes_custom_colors: NodesCusomColors | None
    edges_custom_colors: SerializedEdgesCustomColors | None
