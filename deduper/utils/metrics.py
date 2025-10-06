"""
Metrics collection for monitoring application performance.
"""

import time
import threading
from typing import Dict, Any, Optional
from collections import defaultdict, deque
from .logging_config import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """Collects and stores application metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._lock = threading.Lock()
        self._metrics = defaultdict(lambda: deque(maxlen=max_history))
        self._counters = defaultdict(int)
        self._gauges = defaultdict(float)
        self._timers = defaultdict(list)
        
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        with self._lock:
            key = self._format_key(name, tags)
            self._counters[key] += value
            self._metrics[f"counter.{name}"].append({
                'timestamp': time.time(),
                'value': value,
                'tags': tags or {}
            })
            logger.debug(f"Counter {name} incremented by {value}")
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric value."""
        with self._lock:
            key = self._format_key(name, tags)
            self._gauges[key] = value
            self._metrics[f"gauge.{name}"].append({
                'timestamp': time.time(),
                'value': value,
                'tags': tags or {}
            })
            logger.debug(f"Gauge {name} set to {value}")
    
    def record_timer(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record a timer metric."""
        with self._lock:
            key = self._format_key(name, tags)
            self._timers[key].append(duration)
            # Keep only recent timer values
            if len(self._timers[key]) > self.max_history:
                self._timers[key] = self._timers[key][-self.max_history:]
            
            self._metrics[f"timer.{name}"].append({
                'timestamp': time.time(),
                'value': duration,
                'tags': tags or {}
            })
            logger.debug(f"Timer {name} recorded: {duration:.3f}s")
    
    def get_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> int:
        """Get current counter value."""
        with self._lock:
            key = self._format_key(name, tags)
            return self._counters.get(key, 0)
    
    def get_gauge(self, name: str, tags: Optional[Dict[str, str]] = None) -> float:
        """Get current gauge value."""
        with self._lock:
            key = self._format_key(name, tags)
            return self._gauges.get(key, 0.0)
    
    def get_timer_stats(self, name: str, tags: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get timer statistics (min, max, avg, count)."""
        with self._lock:
            key = self._format_key(name, tags)
            values = self._timers.get(key, [])
            
            if not values:
                return {'min': 0, 'max': 0, 'avg': 0, 'count': 0}
            
            return {
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values),
                'count': len(values)
            }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics."""
        with self._lock:
            return {
                'counters': dict(self._counters),
                'gauges': dict(self._gauges),
                'timers': {k: self.get_timer_stats(k) for k in self._timers.keys()},
                'timestamp': time.time()
            }
    
    def reset_metrics(self):
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._timers.clear()
            self._metrics.clear()
            logger.info("All metrics reset")
    
    def _format_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Format metric key with tags."""
        if not tags:
            return name
        
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, metrics: MetricsCollector, name: str, tags: Optional[Dict[str, str]] = None):
        self.metrics = metrics
        self.name = name
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics.record_timer(self.name, duration, self.tags)


# Global metrics collector instance
metrics = MetricsCollector()


def timer(name: str, tags: Optional[Dict[str, str]] = None):
    """Decorator for timing function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with TimerContext(metrics, name, tags):
                return func(*args, **kwargs)
        # Preserve the original function's metadata for Flask
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__module__ = func.__module__
        wrapper.__qualname__ = func.__qualname__
        return wrapper
    return decorator


def increment_counter(name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
    """Convenience function to increment a counter."""
    metrics.increment_counter(name, value, tags)


def set_gauge(name: str, value: float, tags: Optional[Dict[str, str]] = None):
    """Convenience function to set a gauge."""
    metrics.set_gauge(name, value, tags)


def record_timer(name: str, duration: float, tags: Optional[Dict[str, str]] = None):
    """Convenience function to record a timer."""
    metrics.record_timer(name, duration, tags)

