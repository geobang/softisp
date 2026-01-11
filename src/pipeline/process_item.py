import time

class ProcessItem:
    def __init__(self, frame_id, image=None, coeffs_bulk=None):
        self.frame_id = frame_id
        self.image = image
        self.coeffs_bulk = dict(coeffs_bulk or {})
        self.timestamp = time.time()
