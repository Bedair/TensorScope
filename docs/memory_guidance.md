# Model-level memory guidance

`tensorscope analyze` reports deterministic findings and qualitative optimization
recommendations for the statically planned arena head. It examines exact tensor
sizes, alignment, lifetimes, the selected peak, reuse blockers, graph inputs and
outputs, ADD/MUL merge inputs, and an optional arena-head budget. It does not
automatically optimize or rewrite a model.

## Rules and thresholds

- Peak concentration: one tensor at least 50% of peak live aligned bytes, the
  top two at least 75%, or the top three at least 90%.
- Long-lived tensor: at least 25% of planned head and live for at least three
  execution scopes.
- Reuse blocker: overlaps at least two tensors whose combined aligned size is
  at least 25% of planned head. Its ranking score is the blocker size times the
  blocked count, plus the blocked tensors' aligned bytes.
- Alignment: total overhead at least 10%, at least three tensors smaller than
  one alignment block, or an individual tensor with at least 50% overhead.
- Input retention: graph input live for at least three scopes and at least 10%
  of planned head.
- Output retention: graph output at least 25% of planned head. Required outputs
  cannot simply be freed early.
- Branch/merge pressure: ADD or MUL has at least two runtime inputs totaling at
  least 50% of planned head.
- Budget pressure: exceeded is critical; exact fit is high; at least 90% is
  high; at least 75% is medium; below 75% is informational/comfortable.

Exact size and threshold findings use `exact` confidence. Conditional model
change recommendations use `medium` confidence because graph replanning,
accuracy, semantics, and kernel support must be checked after any change.
Overall risk is the highest finding severity. Findings are ordered by severity,
known impact score, category, affected IDs, and stable finding ID.

For `hello_world_float.tflite`, guidance identifies concentrated fully connected
activation memory, exact lifetime overlap blockers, and alignment overhead from
small runtime tensors. For `simple_add_model.tflite`, it identifies simultaneous
ADD inputs and the required output allocation. These observations are derived
from the model rather than hard-coded model names.

Recommendations describe possible directions—such as narrowing a dominant
tensor, reviewing a late consumer, or reducing tiny intermediates—not guaranteed
byte savings. TensorScope does not prove a graph rewrite safe, recommend changing
allocator alignment, estimate scratch or arena tail, or establish complete MCU
or firmware fit. This guidance covers planned arena head only, and model accuracy,
operator support, and graph semantics must be revalidated after every model change.
