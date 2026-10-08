from collections import Counter
from threading import Lock


class DetectionMetrics:
    def __init__(self) -> None:
        self._lock = Lock()

        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_detections = 0
        self.total_latency_ms = 0.0
        self.class_counts = Counter()

    def record_success(
        self,
        detections: list[dict],
        latency_ms: float,
    ) -> None:
        with self._lock:
            self.total_requests += 1
            self.successful_requests += 1
            self.total_detections += len(detections)
            self.total_latency_ms += latency_ms

            for detection in detections:
                class_name = detection.get(
                    "class_name",
                    "unknown",
                )

                self.class_counts[class_name] += 1

    def record_failure(self) -> None:
        with self._lock:
            self.total_requests += 1
            self.failed_requests += 1

    def snapshot(self) -> dict:
        with self._lock:
            average_latency_ms = (
                self.total_latency_ms
                / self.successful_requests
                if self.successful_requests > 0
                else 0.0
            )

            average_detections = (
                self.total_detections
                / self.successful_requests
                if self.successful_requests > 0
                else 0.0
            )

            return {
                "total_requests": self.total_requests,
                "successful_requests": (
                    self.successful_requests
                ),
                "failed_requests": self.failed_requests,
                "total_detections": self.total_detections,
                "average_detections_per_image": round(
                    average_detections,
                    2,
                ),
                "average_latency_ms": round(
                    average_latency_ms,
                    2,
                ),
                "class_counts": dict(
                    self.class_counts.most_common()
                ),
            }
