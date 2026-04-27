import threading
from collections import OrderedDict

from planner.simulation.client import SimulationResult

_CacheKey = tuple[str, str, int, int, int, float, int]


class SimulationCache:
    def __init__(self, max_size: int = 2048) -> None:
        self._max = max_size
        self._data: OrderedDict[_CacheKey, SimulationResult] = OrderedDict()
        self._lock = threading.Lock()

    def _key(
        self,
        model: str,
        gpu: str,
        tp: int,
        prompt_tokens: int,
        output_tokens: int,
        qps: float,
        prefix_tokens: int,
    ) -> _CacheKey:
        return (model, gpu, tp, prompt_tokens, output_tokens, qps, prefix_tokens)

    def get(
        self,
        model: str,
        gpu: str,
        tp: int,
        prompt_tokens: int,
        output_tokens: int,
        qps: float,
        prefix_tokens: int,
    ) -> SimulationResult | None:
        key = self._key(model, gpu, tp, prompt_tokens, output_tokens, qps, prefix_tokens)
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def put(
        self,
        model: str,
        gpu: str,
        tp: int,
        prompt_tokens: int,
        output_tokens: int,
        qps: float,
        prefix_tokens: int,
        result: SimulationResult,
    ) -> None:
        key = self._key(model, gpu, tp, prompt_tokens, output_tokens, qps, prefix_tokens)
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = result
            if len(self._data) > self._max:
                self._data.popitem(last=False)


# Module-level singleton: shared across all requests in the same process.
# Results are deterministic, so cross-request sharing is safe.
_default_cache = SimulationCache(max_size=2048)


def get_default_cache() -> SimulationCache:
    return _default_cache
