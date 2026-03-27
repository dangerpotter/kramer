# Deprecation Notice

The top-level `kramer/` package is **deprecated** and will be removed in a future version.

All active development has moved to `src/kramer/` and `src/orchestrator/`.

Please update your imports to use the `src.` prefix:

```python
# Old (deprecated)
from kramer.some_module import SomeClass

# New
from src.kramer.some_module import SomeClass
```
