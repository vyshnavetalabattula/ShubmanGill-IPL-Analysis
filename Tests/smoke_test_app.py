"""Executes app.py against a stub `streamlit` module to catch NameErrors/typos when Streamlit isn't installed.
It does NOT render a real UI - run `streamlit run app.py` for that."""
import contextlib
import sys
import types


class Stub(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = {}
        self.calls = []

    @contextlib.contextmanager
    def _cm(self):
        yield self

    def cache_resource(self, *a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]

    def columns(self, spec, **k):
        return [self for _ in range(spec if isinstance(spec, int) else len(spec))]

    def tabs(self, labels):
        return [self for _ in labels]

    def __enter__(self): return self
    def __exit__(self, *a): return False

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        def call(*a, **k):
            self.calls.append(name)
            if name == "selectbox": return a[1][k.get("index", 0)]
            if name == "radio": return a[1][0]
            if name == "number_input": return k["value"]
            if name == "checkbox": return k.get("value", False)
            if name == "form_submit_button": return True
            if name in ("form", "expander"): return self
            if name == "pyplot": return None
            return self
        return call

    def __getitem__(self, i): return self


st = Stub()
sys.modules["streamlit"] = st
import matplotlib
matplotlib.use("Agg")
import runpy
runpy.run_path("app.py", run_name="__main__")
need = {"metric", "pyplot", "dataframe", "warning", "form_submit_button"}
assert need <= set(st.calls), f"app did not reach: {need - set(st.calls)}"
print("smoke test passed: app executed end-to-end with stub; widgets called:", len(st.calls))
