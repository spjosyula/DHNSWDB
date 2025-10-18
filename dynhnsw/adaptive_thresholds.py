"""Adaptive threshold selection for layer-adaptive multi-path search.

This module implements data-driven threshold optimization for determining
the number of search paths based on query difficulty.

Research Goal: Replace hardcoded thresholds (0.8, 0.9) with adaptive thresholds
learned from dataset characteristics and performance feedback.
"""

import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class ThresholdConfig:
    """Configuration for adaptive thresholds."""

    # Current thresholds (learned or initialized)
    t1: float = 0.8  # Easy/Medium boundary
    t2: float = 0.9  # Medium/Hard boundary

    # Statistics
    num_updates: int = 0
    performance_history: List[Tuple[float, float, float]] = None  # (t1, t2, recall)

    def __post_init__(self):
        if self.performance_history is None:
            self.performance_history = []


class AdaptiveThresholdSelector:
    """Learns optimal thresholds for path selection based on difficulty distribution.

    Strategy:
    1. Calibration Phase: Collect difficulty and recall statistics
    2. Optimization Phase: Use grid search or gradient-free optimization
    3. Online Adaptation: Continuously refine based on feedback

    The goal is to maximize recall while controlling latency by finding optimal
    difficulty thresholds that determine when to use 1, 2, or 3 paths.
    """

    def __init__(
        self,
        initial_t1: float = 0.8,
        initial_t2: float = 0.9,
        calibration_queries: int = 100,
        update_frequency: int = 50,
    ):
        """Initialize adaptive threshold selector.

        Args:
            initial_t1: Initial easy/medium threshold
            initial_t2: Initial medium/hard threshold
            calibration_queries: Number of queries before first optimization
            update_frequency: Update thresholds every N queries
        """
        self.config = ThresholdConfig(t1=initial_t1, t2=initial_t2)
        self.calibration_queries = calibration_queries
        self.update_frequency = update_frequency

        # Calibration data collection
        self.calibration_buffer: List[Dict] = []
        self.is_calibrated = False

        # Performance tracking
        self.difficulty_history: List[float] = []
        self.recall_history: List[float] = []
        self.path_count_history: List[int] = []

    def get_thresholds(self) -> Tuple[float, float]:
        """Get current thresholds for path selection.

        Returns:
            Tuple of (t1, t2) where:
            - difficulty < t1: 1 path (easy)
            - t1 <= difficulty < t2: 2 paths (medium)
            - difficulty >= t2: 3 paths (hard)
        """
        return (self.config.t1, self.config.t2)

    def record_query(
        self,
        difficulty: float,
        num_paths_used: int,
        recall: float,
        latency_ms: float
    ) -> None:
        """Record query result for threshold learning.

        Args:
            difficulty: Query difficulty score
            num_paths_used: Number of paths used for this query
            recall: Recall@k achieved
            latency_ms: Query latency in milliseconds
        """
        # Store in history
        self.difficulty_history.append(difficulty)
        self.recall_history.append(recall)
        self.path_count_history.append(num_paths_used)

        # Store in calibration buffer if not yet calibrated
        if not self.is_calibrated:
            self.calibration_buffer.append({
                'difficulty': difficulty,
                'num_paths': num_paths_used,
                'recall': recall,
                'latency': latency_ms
            })

            # Check if ready to calibrate
            if len(self.calibration_buffer) >= self.calibration_queries:
                self._run_calibration()
                self.is_calibrated = True

        # Periodic updates after calibration
        elif len(self.difficulty_history) % self.update_frequency == 0:
            self._run_incremental_update()

    def _run_calibration(self) -> None:
        """Initial calibration using collected data.

        Strategy: Find thresholds that maximize average recall across difficulty bins.
        Uses percentile-based initialization followed by local search.
        """
        if len(self.calibration_buffer) < 20:
            # Not enough data for meaningful calibration
            return

        difficulties = np.array([q['difficulty'] for q in self.calibration_buffer])
        recalls = np.array([q['recall'] for q in self.calibration_buffer])

        # Strategy 1: Percentile-based initialization
        # Place thresholds to balance query distribution across bins
        t1_percentile = 33  # 33rd percentile
        t2_percentile = 67  # 67th percentile

        candidate_t1 = float(np.percentile(difficulties, t1_percentile))
        candidate_t2 = float(np.percentile(difficulties, t2_percentile))

        # Ensure minimum separation between thresholds
        min_separation = 0.05
        if candidate_t2 - candidate_t1 < min_separation:
            candidate_t2 = candidate_t1 + min_separation

        # Validate candidates improve over current thresholds
        current_score = self._evaluate_thresholds(
            self.config.t1, self.config.t2, difficulties, recalls
        )
        candidate_score = self._evaluate_thresholds(
            candidate_t1, candidate_t2, difficulties, recalls
        )

        if candidate_score > current_score:
            self.config.t1 = candidate_t1
            self.config.t2 = candidate_t2
            self.config.num_updates += 1
            print(f"[Threshold Calibration] Updated: t1={self.config.t1:.3f}, t2={self.config.t2:.3f} (score: {candidate_score:.4f})")
        else:
            print(f"[Threshold Calibration] Keeping current thresholds (score: {current_score:.4f})")

    def _run_incremental_update(self) -> None:
        """Incremental threshold update based on recent performance.

        Uses sliding window of recent queries to adapt thresholds gradually.
        """
        window_size = min(self.update_frequency * 2, len(self.difficulty_history))
        recent_difficulties = np.array(self.difficulty_history[-window_size:])
        recent_recalls = np.array(self.recall_history[-window_size:])

        # Small local search around current thresholds
        best_t1 = self.config.t1
        best_t2 = self.config.t2
        best_score = self._evaluate_thresholds(best_t1, best_t2, recent_difficulties, recent_recalls)

        # Search in small neighborhood
        delta = 0.02  # Search within ±0.02 of current values
        for t1_offset in [-delta, 0, delta]:
            for t2_offset in [-delta, 0, delta]:
                candidate_t1 = np.clip(self.config.t1 + t1_offset, 0.5, 0.95)
                candidate_t2 = np.clip(self.config.t2 + t2_offset, 0.6, 1.0)

                # Ensure t2 > t1
                if candidate_t2 <= candidate_t1 + 0.05:
                    continue

                score = self._evaluate_thresholds(
                    candidate_t1, candidate_t2, recent_difficulties, recent_recalls
                )

                if score > best_score:
                    best_score = score
                    best_t1 = candidate_t1
                    best_t2 = candidate_t2

        # Update if improvement found
        if best_t1 != self.config.t1 or best_t2 != self.config.t2:
            self.config.t1 = best_t1
            self.config.t2 = best_t2
            self.config.num_updates += 1
            print(f"[Threshold Update] t1={self.config.t1:.3f}, t2={self.config.t2:.3f} (score: {best_score:.4f})")

    def _evaluate_thresholds(
        self,
        t1: float,
        t2: float,
        difficulties: npt.NDArray[np.float64],
        recalls: npt.NDArray[np.float64]
    ) -> float:
        """Evaluate quality of threshold pair.

        Strategy: Maximize average recall with penalty for imbalanced bins.

        Args:
            t1: Easy/medium threshold
            t2: Medium/hard threshold
            difficulties: Array of query difficulties
            recalls: Array of corresponding recalls

        Returns:
            Score (higher is better)
        """
        # Assign queries to bins
        easy_mask = difficulties < t1
        medium_mask = (difficulties >= t1) & (difficulties < t2)
        hard_mask = difficulties >= t2

        # Compute average recall per bin
        easy_recall = np.mean(recalls[easy_mask]) if np.any(easy_mask) else 0.0
        medium_recall = np.mean(recalls[medium_mask]) if np.any(medium_mask) else 0.0
        hard_recall = np.mean(recalls[hard_mask]) if np.any(hard_mask) else 0.0

        # Overall average recall
        avg_recall = np.mean(recalls)

        # Penalty for empty bins (we want reasonable distribution)
        bin_counts = [np.sum(easy_mask), np.sum(medium_mask), np.sum(hard_mask)]
        bin_penalty = 0.0
        for count in bin_counts:
            if count == 0:
                bin_penalty += 0.1  # Penalize empty bins
            elif count < len(difficulties) * 0.05:  # Less than 5% of queries
                bin_penalty += 0.05  # Penalize very small bins

        # Score: maximize recall while avoiding empty bins
        score = avg_recall - bin_penalty

        return float(score)

    def get_statistics(self) -> Dict:
        """Get statistics about threshold adaptation.

        Returns:
            Dictionary with adaptation statistics
        """
        if len(self.difficulty_history) == 0:
            return {
                'is_calibrated': self.is_calibrated,
                'num_updates': self.config.num_updates,
                'current_t1': self.config.t1,
                'current_t2': self.config.t2,
            }

        difficulties = np.array(self.difficulty_history)
        recalls = np.array(self.recall_history)

        # Current bin distribution
        easy_mask = difficulties < self.config.t1
        medium_mask = (difficulties >= self.config.t1) & (difficulties < self.config.t2)
        hard_mask = difficulties >= self.config.t2

        return {
            'is_calibrated': self.is_calibrated,
            'num_updates': self.config.num_updates,
            'current_t1': self.config.t1,
            'current_t2': self.config.t2,
            'total_queries': len(self.difficulty_history),
            'bin_distribution': {
                'easy': int(np.sum(easy_mask)),
                'medium': int(np.sum(medium_mask)),
                'hard': int(np.sum(hard_mask)),
            },
            'recall_by_bin': {
                'easy': float(np.mean(recalls[easy_mask])) if np.any(easy_mask) else 0.0,
                'medium': float(np.mean(recalls[medium_mask])) if np.any(medium_mask) else 0.0,
                'hard': float(np.mean(recalls[hard_mask])) if np.any(hard_mask) else 0.0,
            }
        }
