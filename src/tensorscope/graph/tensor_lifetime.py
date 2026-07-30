from __future__ import annotations

from dataclasses import dataclass

from tensorscope.graph.model import (
    GraphModel,
    GraphModelError,
    Subgraph,
    Tensor,
    TensorId,
)
from tensorscope.graph.tensor_size import (
    calculate_tensor_size,
)


class TensorLifetimeError(GraphModelError):
    """Raised when tensor lifetimes cannot be determined."""


@dataclass(frozen=True)
class TensorLifetime:
    """Lifetime of one tensor in allocation-scope coordinates."""

    tensor_id: TensorId
    first_created: int | None
    last_used: int | None
    needs_allocation: bool
    is_subgraph_input: bool
    is_subgraph_output: bool

    def __post_init__(self) -> None:
        if self.tensor_id < 0:
            raise TensorLifetimeError(
                f"Tensor ID must be non-negative: {self.tensor_id}"
            )

        if (
            self.first_created is not None
            and self.first_created < 0
        ):
            raise TensorLifetimeError(
                "First-created scope must be non-negative: "
                f"{self.first_created}"
            )

        if (
            self.last_used is not None
            and self.last_used < 0
        ):
            raise TensorLifetimeError(
                "Last-used scope must be non-negative: "
                f"{self.last_used}"
            )

        if (
            self.first_created is not None
            and self.last_used is not None
            and self.last_used < self.first_created
        ):
            raise TensorLifetimeError(
                "Last-used scope cannot precede first-created "
                f"scope for tensor {self.tensor_id}: "
                f"{self.first_created}..{self.last_used}"
            )

        if self.needs_allocation and not self.is_initialized:
            raise TensorLifetimeError(
                "Runtime tensor requiring allocation has no "
                f"complete lifetime: tensor {self.tensor_id}"
            )

    @property
    def is_initialized(self) -> bool:
        return (
            self.first_created is not None
            and self.last_used is not None
        )

    @property
    def duration(self) -> int:
        """Number of inclusive allocation scopes in the lifetime."""

        if not self.is_initialized:
            return 0

        assert self.first_created is not None
        assert self.last_used is not None

        return self.last_used - self.first_created + 1

    def is_live_at(self, scope: int) -> bool:
        """Return whether the tensor is live at an allocation scope."""

        if scope < 0:
            raise TensorLifetimeError(
                f"Allocation scope must be non-negative: {scope}"
            )

        if not self.is_initialized:
            return False

        assert self.first_created is not None
        assert self.last_used is not None

        return self.first_created <= scope <= self.last_used

    def overlaps(
        self,
        other: TensorLifetime,
    ) -> bool:
        """Return whether two inclusive lifetimes overlap."""

        if not self.is_initialized or not other.is_initialized:
            return False

        assert self.first_created is not None
        assert self.last_used is not None
        assert other.first_created is not None
        assert other.last_used is not None

        return not (
            self.last_used < other.first_created
            or other.last_used < self.first_created
        )


@dataclass(frozen=True)
class SubgraphLifetimeAnalysis:
    """Tensor lifetime analysis for one subgraph."""

    subgraph_id: int
    operator_scope_count: int
    lifetimes: tuple[TensorLifetime, ...]

    def __post_init__(self) -> None:
        if self.subgraph_id < 0:
            raise TensorLifetimeError(
                "Subgraph ID must be non-negative: "
                f"{self.subgraph_id}"
            )

        if self.operator_scope_count < 0:
            raise TensorLifetimeError(
                "Operator scope count must be non-negative: "
                f"{self.operator_scope_count}"
            )

        tensor_ids = tuple(
            lifetime.tensor_id
            for lifetime in self.lifetimes
        )

        if tensor_ids != tuple(range(len(self.lifetimes))):
            raise TensorLifetimeError(
                "Lifetime tensor IDs must be contiguous and "
                f"match their positions: {tensor_ids}"
            )

    def tensor(
        self,
        tensor_id: TensorId,
    ) -> TensorLifetime:
        try:
            return self.lifetimes[tensor_id]
        except IndexError as error:
            raise TensorLifetimeError(
                f"Unknown tensor lifetime ID: {tensor_id}"
            ) from error

    @property
    def plannable_lifetimes(
        self,
    ) -> tuple[TensorLifetime, ...]:
        return tuple(
            lifetime
            for lifetime in self.lifetimes
            if lifetime.needs_allocation
        )


@dataclass(frozen=True)
class GraphLifetimeAnalysis:
    """Lifetime analyses for all subgraphs in a graph model."""

    subgraphs: tuple[SubgraphLifetimeAnalysis, ...]

    def subgraph(
        self,
        subgraph_id: int,
    ) -> SubgraphLifetimeAnalysis:
        try:
            return self.subgraphs[subgraph_id]
        except IndexError as error:
            raise TensorLifetimeError(
                f"Unknown subgraph lifetime ID: {subgraph_id}"
            ) from error

    @property
    def primary_subgraph(
        self,
    ) -> SubgraphLifetimeAnalysis:
        if not self.subgraphs:
            raise TensorLifetimeError(
                "Graph lifetime analysis contains no subgraphs"
            )

        return self.subgraphs[0]


@dataclass
class _MutableLifetime:
    first_created: int | None = None
    last_used: int | None = None

    def mark_created(
        self,
        scope: int,
    ) -> None:
        if self.first_created is None:
            self.first_created = scope

    def mark_used(
        self,
        scope: int,
    ) -> None:
        if self.last_used is not None and scope < self.last_used:
            raise TensorLifetimeError(
                "Lifetime scopes must be processed "
                "monotonically"
            )

        self.last_used = scope


def _needs_runtime_allocation(
    tensor: Tensor,
) -> bool:
    if tensor.has_constant_data:
        return False

    if tensor.is_variable:
        return False

    size = calculate_tensor_size(tensor)

    return size.storage_bytes > 0


def _validate_tensor_reference(
    subgraph: Subgraph,
    tensor_id: int,
    *,
    allow_optional: bool,
) -> bool:
    if tensor_id == -1 and allow_optional:
        return False

    if tensor_id < 0 or tensor_id >= len(subgraph.tensors):
        raise TensorLifetimeError(
            f"Subgraph {subgraph.id} references invalid "
            f"tensor ID: {tensor_id}"
        )

    return True


def calculate_subgraph_lifetimes(
    subgraph: Subgraph,
) -> SubgraphLifetimeAnalysis:
    """
    Calculate tensor lifetimes using TFLM allocation scopes.

    Scope zero represents entry into the subgraph. Operator zero uses
    scope one, operator one uses scope two, and so on.
    """

    mutable_lifetimes = [
        _MutableLifetime()
        for _ in subgraph.tensors
    ]

    input_ids = set(subgraph.inputs)
    output_ids = set(subgraph.outputs)

    # Subgraph inputs exist before the first operator executes.
    for tensor_id in subgraph.inputs:
        _validate_tensor_reference(
            subgraph,
            tensor_id,
            allow_optional=False,
        )

        lifetime = mutable_lifetimes[tensor_id]
        lifetime.mark_created(0)
        lifetime.mark_used(0)

    for operator in subgraph.operators:
        scope = operator.id + 1

        # TFLM marks outputs as created before marking all input and
        # output uses for the current operator.
        for tensor_id in operator.outputs:
            if not _validate_tensor_reference(
                subgraph,
                tensor_id,
                allow_optional=False,
            ):
                continue

            mutable_lifetimes[
                tensor_id
            ].mark_created(scope)

        # Optional operator inputs may be represented by -1.
        for tensor_id in operator.inputs:
            if not _validate_tensor_reference(
                subgraph,
                tensor_id,
                allow_optional=True,
            ):
                continue

            mutable_lifetimes[
                tensor_id
            ].mark_used(scope)

        for tensor_id in operator.outputs:
            if not _validate_tensor_reference(
                subgraph,
                tensor_id,
                allow_optional=False,
            ):
                continue

            mutable_lifetimes[
                tensor_id
            ].mark_used(scope)

    final_scope = len(subgraph.operators)

    # Outputs remain valid through the end of the subgraph invocation.
    # Marking creation here also supports empty subgraphs.
    for tensor_id in subgraph.outputs:
        _validate_tensor_reference(
            subgraph,
            tensor_id,
            allow_optional=False,
        )

        lifetime = mutable_lifetimes[tensor_id]
        lifetime.mark_created(final_scope)
        lifetime.mark_used(final_scope)

    result: list[TensorLifetime] = []

    for tensor in subgraph.tensors:
        mutable = mutable_lifetimes[tensor.id]

        needs_allocation = _needs_runtime_allocation(
            tensor
        )

        if needs_allocation and (
            mutable.first_created is None
            or mutable.last_used is None
        ):
            raise TensorLifetimeError(
                "Runtime tensor requires allocation but is "
                "not connected to the executable graph: "
                f"subgraph {subgraph.id}, tensor {tensor.id} "
                f"({tensor.name!r})"
            )

        result.append(
            TensorLifetime(
                tensor_id=tensor.id,
                first_created=mutable.first_created,
                last_used=mutable.last_used,
                needs_allocation=needs_allocation,
                is_subgraph_input=tensor.id in input_ids,
                is_subgraph_output=tensor.id in output_ids,
            )
        )

    return SubgraphLifetimeAnalysis(
        subgraph_id=subgraph.id,
        operator_scope_count=len(subgraph.operators),
        lifetimes=tuple(result),
    )


def calculate_graph_lifetimes(
    graph: GraphModel,
) -> GraphLifetimeAnalysis:
    """Calculate tensor lifetimes for every graph subgraph."""

    return GraphLifetimeAnalysis(
        subgraphs=tuple(
            calculate_subgraph_lifetimes(subgraph)
            for subgraph in graph.subgraphs
        )
    )