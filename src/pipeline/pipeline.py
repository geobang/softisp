import threading

class Pipeline:
    def __init__(self, manifest):
        self.manifest = manifest
        self.canonical = manifest.get("canonical_name", "pipeline")
        self._lock = threading.Lock()
        self._coeff_store = {}
        self._required = self._compute_required_coeffs()

    def _compute_required_coeffs(self):
        required = {}
        for spec in self.manifest.get("pipeline", []):
            blk = spec["block"].lower()
            coeffs = set(spec.get("coeff_names", []))
            required.setdefault(blk, set()).update(coeffs)
        return required

    def set_coeff_val(self, block, coeff, value):
        with self._lock:
            self._coeff_store[(block.lower(), coeff.lower())] = value

    def get_coeff_val(self, block, coeff):
        with self._lock:
            return self._coeff_store.get((block.lower(), coeff.lower()))

    def set_get_coeff_val(self, block, coeff, value=None):
        if value is not None:
            self.set_coeff_val(block, coeff, value)
        return self.get_coeff_val(block, coeff)

    def set_coeffs(self, block, coeffs: dict):
        for k, v in coeffs.items():
            self.set_coeff_val(block, k, v)

    def get_coeffs(self, block):
        block = block.lower()
        with self._lock:
            return {c: v for (b, c), v in self._coeff_store.items() if b == block}

    def get_all_coeffs(self):
        with self._lock:
            missing = {}
            for block, reqs in self._required.items():
                have = {c for (b, c) in self._coeff_store.keys() if b == block}
                miss = sorted(reqs - have)
                if miss:
                    missing[block] = miss
            if missing:
                raise ValueError(f"Missing required coeffs: {missing}")
            return {f"{b}_{c}": v for (b, c), v in self._coeff_store.items()}
