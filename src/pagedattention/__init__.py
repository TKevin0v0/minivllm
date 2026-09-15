"""PagedAttention 教学模块：页表映射 + 按 slot 写 / 按 block_table 读。

与 ``src.kv`` 的分工：
- **本包**：纯存储契约（高内聚）——不感知前缀缓存 / Sequence 生命周期
- **kv**：块分配、前缀索引、Engine 编排（通过 ForwardContext 注入本包）

Attention 只依赖 ``PagedStore`` + ``slot_mapping`` / ``block_table``，与 KVManager 解耦。

对外公开 API（``__all__``）::

    slots_tensor   — 逻辑区间 → GPU slot_mapping
    PagedStore     — write / gather

内部实现（下划线，见各模块）::

    slots._slot_of / slots._slots_for_range
    store._BlockTensorStore / _coerce_kv_token_major / _flat_layer_views
"""

from .slots import slots_tensor
from .store import PagedStore

__all__ = [
    "PagedStore",
    "slots_tensor",
]
