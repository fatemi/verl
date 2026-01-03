# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Prioritized Batch Sampler for GRPO Training.

This module implements a priority-based sampling strategy where problems are sampled
based on their empirical difficulty. Problems with success rates closer to 0.5
(maximum uncertainty) are prioritized, using omega = p * (1 - p) as the priority score.

Components:
    - RLDatasetWithProblemId: Dataset wrapper that adds persistent problem_id to samples.
    - MaxHeap: Max-heap data structure for efficient priority-based extraction.
    - ProblemPriorityManager: Manages EMA success rates and priority scores.
    - PriorityBatchSampler: PyTorch Sampler that yields batches based on priority.

Usage:
    from verl.utils.dataset.priority_sampler import (
        RLDatasetWithProblemId,
        ProblemPriorityManager,
        PriorityBatchSampler,
        create_priority_sampler,
    )

    # Wrap existing dataset to add problem_id
    dataset = RLDatasetWithProblemId(
        data_files=data_paths,
        tokenizer=tokenizer,
        config=data_config,
    )

    # Create priority manager and sampler
    priority_manager, sampler = create_priority_sampler(
        dataset_size=len(dataset),
        batch_size=32,
        config={'alpha': 0.8, 'initial_omega': float('inf')},
    )

    # In training loop, after computing rewards:
    priority_manager.record_results(problem_ids, rewards)
"""

import logging
from collections import defaultdict
from typing import Iterator, Optional

import numpy as np
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)

__all__ = [
    "RLDatasetWithProblemId",
    "MaxHeap",
    "ProblemPriorityManager",
    "PriorityBatchSampler",
    "create_priority_sampler",
]


# =============================================================================
# Dataset with Problem ID
# =============================================================================


class RLDatasetWithProblemId:
    """
    Dataset wrapper that adds a persistent problem_id to each sample.

    This class wraps an existing RL dataset (like RLHFDataset) and adds a
    `problem_id` field to each returned sample. The problem_id is the row
    index into the dataset and remains stable across shuffling.

    This is implemented as a wrapper rather than a subclass to maintain
    compatibility with any RL dataset implementation.

    Attributes:
        base_dataset: The underlying dataset being wrapped.

    Example:
        from verl.utils.dataset.rl_dataset import RLHFDataset

        base_dataset = RLHFDataset(data_files=..., tokenizer=..., config=...)
        dataset = RLDatasetWithProblemId(base_dataset)

        sample = dataset[42]
        assert sample['problem_id'] == 42
    """

    def __init__(self, base_dataset):
        """
        Initialize the wrapper.

        Args:
            base_dataset: The underlying dataset to wrap. Must support
                         __len__ and __getitem__.
        """
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        """Return the number of problems in the dataset."""
        return len(self.base_dataset)

    def __getitem__(self, item: int) -> dict:
        """
        Get a sample with problem_id added.

        Args:
            item: Index into the dataset.

        Returns:
            Dictionary containing the sample data with 'problem_id' added.
            The problem_id equals the item index and is stable across shuffling.
        """
        sample = self.base_dataset[item]
        # Add persistent problem_id (0-based index into the dataset)
        # This ID is stable across shuffling since it corresponds to the dataset row index
        sample["problem_id"] = item
        return sample

    def __getattr__(self, name):
        """Delegate attribute access to the base dataset."""
        return getattr(self.base_dataset, name)


def create_rl_dataset_with_problem_id(
    data_files,
    tokenizer,
    config,
    processor=None,
    max_samples: int = -1,
):
    """
    Factory function to create an RL dataset with problem_id support.

    This creates an RLHFDataset and wraps it with RLDatasetWithProblemId.

    Args:
        data_files: Path(s) to data files.
        tokenizer: Tokenizer for text processing.
        config: Dataset configuration.
        processor: Optional multimodal processor.
        max_samples: Maximum samples to load (-1 for all).

    Returns:
        RLDatasetWithProblemId instance.
    """
    from verl.utils.dataset.rl_dataset import RLHFDataset

    base_dataset = RLHFDataset(
        data_files=data_files,
        tokenizer=tokenizer,
        config=config,
        processor=processor,
        max_samples=max_samples,
    )
    return RLDatasetWithProblemId(base_dataset)


# =============================================================================
# Heap Data Structures
# =============================================================================


class MinHeap:
    """
    Min-heap for tracking least-recently-tested problems.

    Used for solved/unsolved pools where we want to prioritize retesting
    problems that haven't been tested for the longest time.

    Attributes:
        priority: Array of priorities (lower = higher priority for extraction).
        heap: Array of problem IDs arranged as a min-heap.
        position: Dictionary mapping problem ID to its position in the heap.
    """

    def __init__(self, capacity: int):
        """
        Initialize an empty min-heap.

        Args:
            capacity: Maximum number of problems (for priority array sizing).
        """
        self.priority = np.full(capacity, float("inf"), dtype=np.float64)
        self.heap: list[int] = []
        self.position: dict[int, int] = {}

    def _swap(self, i: int, j: int) -> None:
        """Swap elements at positions i and j in the heap."""
        pid_i, pid_j = self.heap[i], self.heap[j]
        self.heap[i], self.heap[j] = pid_j, pid_i
        self.position[pid_i] = j
        self.position[pid_j] = i

    def _sift_up(self, i: int) -> None:
        """Restore heap property by sifting element at position i upward."""
        while i > 0:
            parent = (i - 1) // 2
            if self.priority[self.heap[i]] >= self.priority[self.heap[parent]]:
                break
            self._swap(i, parent)
            i = parent

    def _sift_down(self, i: int) -> None:
        """Restore heap property by sifting element at position i downward."""
        n = len(self.heap)
        while True:
            left = 2 * i + 1
            right = 2 * i + 2
            smallest = i

            if left < n and self.priority[self.heap[left]] < self.priority[self.heap[smallest]]:
                smallest = left
            if right < n and self.priority[self.heap[right]] < self.priority[self.heap[smallest]]:
                smallest = right

            if smallest == i:
                break
            self._swap(i, smallest)
            i = smallest

    def extract_min(self) -> int:
        """
        Remove and return the problem ID with the smallest priority (oldest).

        Returns:
            The problem ID with minimum priority.

        Raises:
            IndexError: If the heap is empty.
        """
        if not self.heap:
            raise IndexError("extract_min from empty heap")

        root = self.heap[0]
        del self.position[root]

        if len(self.heap) > 1:
            last = self.heap.pop()
            self.heap[0] = last
            self.position[last] = 0
            self._sift_down(0)
        else:
            self.heap.pop()

        return root

    def insert(self, pid: int, priority: float) -> None:
        """
        Insert a problem ID with the given priority.

        Args:
            pid: Problem ID to insert.
            priority: Priority score (lower = extracted sooner).
        """
        self.priority[pid] = priority
        self.heap.append(pid)
        pos = len(self.heap) - 1
        self.position[pid] = pos
        self._sift_up(pos)

    def remove(self, pid: int) -> None:
        """
        Remove a specific problem ID from the heap.

        Args:
            pid: Problem ID to remove.
        """
        if pid not in self.position:
            return

        pos = self.position[pid]
        del self.position[pid]

        if pos == len(self.heap) - 1:
            self.heap.pop()
            return

        # Move last element to this position
        last = self.heap.pop()
        self.heap[pos] = last
        self.position[last] = pos

        # Restore heap property
        if pos > 0 and self.priority[last] < self.priority[self.heap[(pos - 1) // 2]]:
            self._sift_up(pos)
        else:
            self._sift_down(pos)

    def peek_min(self) -> Optional[int]:
        """Return the problem ID with minimum priority without removing it."""
        return self.heap[0] if self.heap else None

    def __len__(self) -> int:
        """Return the number of elements in the heap."""
        return len(self.heap)

    def __contains__(self, pid: int) -> bool:
        """Check if a problem ID is in the heap."""
        return pid in self.position


class MaxHeap:
    """
    Max-heap over problem indices.

    Stores problem indices arranged as a binary max-heap based on priority scores.
    Supports efficient extract_max() and insert() operations in O(log n) time.

    Attributes:
        priority: Array of priority scores, indexed by problem ID.
        heap: Array of problem IDs arranged as a max-heap.
        position: Dictionary mapping problem ID to its position in the heap.
    """

    def __init__(self, priorities: np.ndarray):
        """
        Initialize the max-heap.

        Args:
            priorities: Array of initial priority scores for each problem.
                       Index j contains the priority omega_j for problem j.
        """
        self.priority = priorities.copy()
        self.heap = list(range(len(priorities)))
        # Track position of each problem in the heap for efficient updates
        self.position = {pid: i for i, pid in enumerate(self.heap)}
        self._heapify()

    def _swap(self, i: int, j: int) -> None:
        """Swap elements at positions i and j in the heap."""
        pid_i, pid_j = self.heap[i], self.heap[j]
        self.heap[i], self.heap[j] = pid_j, pid_i
        self.position[pid_i] = j
        self.position[pid_j] = i

    def _sift_up(self, i: int) -> None:
        """Restore heap property by sifting element at position i upward."""
        while i > 0:
            parent = (i - 1) // 2
            if self.priority[self.heap[i]] <= self.priority[self.heap[parent]]:
                break
            self._swap(i, parent)
            i = parent

    def _sift_down(self, i: int) -> None:
        """Restore heap property by sifting element at position i downward."""
        n = len(self.heap)
        while True:
            left = 2 * i + 1
            right = 2 * i + 2
            largest = i

            if left < n and self.priority[self.heap[left]] > self.priority[self.heap[largest]]:
                largest = left
            if right < n and self.priority[self.heap[right]] > self.priority[self.heap[largest]]:
                largest = right

            if largest == i:
                break
            self._swap(i, largest)
            i = largest

    def _heapify(self) -> None:
        """Build heap from array in O(n) time using bottom-up construction."""
        for i in reversed(range(len(self.heap) // 2)):
            self._sift_down(i)
        # Update position mapping after heapify
        self.position = {pid: i for i, pid in enumerate(self.heap)}

    def extract_max(self) -> int:
        """
        Remove and return the problem ID with the largest priority score.

        Returns:
            The problem ID with maximum priority.

        Raises:
            IndexError: If the heap is empty.
        """
        if not self.heap:
            raise IndexError("extract_max from empty heap")

        root = self.heap[0]
        del self.position[root]

        if len(self.heap) > 1:
            last = self.heap.pop()
            self.heap[0] = last
            self.position[last] = 0
            self._sift_down(0)
        else:
            self.heap.pop()

        return root

    def insert(self, pid: int, omega: float) -> None:
        """
        Insert a problem ID with the given priority score.

        Args:
            pid: Problem ID to insert.
            omega: Priority score for the problem.
        """
        self.priority[pid] = omega
        self.heap.append(pid)
        pos = len(self.heap) - 1
        self.position[pid] = pos
        self._sift_up(pos)

    def update_priority(self, pid: int, omega: float) -> None:
        """
        Update the priority of a problem already in the heap.

        Args:
            pid: Problem ID to update.
            omega: New priority score.
        """
        if pid not in self.position:
            # Problem not in heap, insert it
            self.insert(pid, omega)
            return

        old_omega = self.priority[pid]
        self.priority[pid] = omega
        pos = self.position[pid]

        if omega > old_omega:
            self._sift_up(pos)
        else:
            self._sift_down(pos)

    def peek_max(self) -> Optional[int]:
        """Return the problem ID with maximum priority without removing it."""
        return self.heap[0] if self.heap else None

    def __len__(self) -> int:
        """Return the number of elements in the heap."""
        return len(self.heap)

    def __contains__(self, pid: int) -> bool:
        """Check if a problem ID is in the heap."""
        return pid in self.position


# =============================================================================
# Problem Priority Manager
# =============================================================================


class ProblemPriorityManager:
    """
    Manages priority scores for problems based on empirical success rates.

    This class tracks the exponential moving average (EMA) of success rates for
    each problem and computes priority scores using omega = p * (1 - p), which
    gives maximum priority (0.25) to problems with 50% success rate.

    The intuition is that problems at the boundary of the model's capability
    (neither too easy nor too hard) provide the most useful training signal.

    Problems that are fully solved (μ_g = 1) or fully unsolved (μ_g = 0) are moved
    to separate pools (solved_pool, unsolved_pool) to avoid wasting training compute.
    These problems are periodically retested to check if their status has changed.

    Attributes:
        num_problems: Total number of problems in the dataset.
        alpha: EMA coefficient for updating success rates (higher = faster adaptation).
        initial_omega: Initial priority for unseen problems.
        ema_success_rate: Array of EMA success rates for each problem.
        priorities: Array of priority scores (omega) for each problem.
        seen_count: Array tracking how many times each problem has been seen.
        solved_pool: Min-heap of fully solved problems (μ_g = 1), prioritized by last_tested_step.
        unsolved_pool: Min-heap of fully unsolved problems (μ_g = 0), prioritized by last_tested_step.
    """

    def __init__(
        self,
        num_problems: int,
        alpha: float = 1.0,
        initial_omega: float = float("inf"),
        initial_success_rate: float = 0.5,
        solved_threshold: float = 1.0,
        unsolved_threshold: float = 0.0,
        success_bias: float = 0.0,
    ):
        """
        Initialize the priority manager.

        Args:
            num_problems: Total number of problems in the dataset.
            alpha: EMA coefficient for updating success rates (0 < alpha <= 1).
                   Higher values give more weight to recent observations.
            initial_omega: Initial priority for unseen problems.
                          Use float('inf') to prioritize unseen problems first.
            initial_success_rate: Initial success rate assumption for all problems.
            solved_threshold: μ_g threshold for considering a problem "solved" (default: 1.0).
            unsolved_threshold: μ_g threshold for considering a problem "unsolved" (default: 0.0).
            success_bias: Small bias added to omega when p >= 0.5 (default: 0.0).
                         This breaks symmetry by preferring problems the model is almost solving.
                         E.g., with bias=1e-4, "5/8 correct" has higher priority than "3/8 correct".
        """
        self.num_problems = num_problems
        self.alpha = alpha
        self.initial_omega = initial_omega
        self.initial_success_rate = initial_success_rate
        self.solved_threshold = solved_threshold
        self.unsolved_threshold = unsolved_threshold
        self.success_bias = success_bias

        # Initialize EMA success rates
        self.ema_success_rate = np.full(num_problems, initial_success_rate, dtype=np.float64)

        # Initialize priorities (omega = p * (1 - p) or initial_omega for unseen)
        self.priorities = np.full(num_problems, initial_omega, dtype=np.float64)

        # Track how many times each problem has been seen
        self.seen_count = np.zeros(num_problems, dtype=np.int64)

        # Track the last step when each problem was tested
        self.last_tested_step = np.zeros(num_problems, dtype=np.int64)

        # Current training step (updated externally)
        self.current_step = 0

        # Build the max-heap for active problems
        self._heap = MaxHeap(self.priorities)

        # Track problems currently extracted (not yet re-inserted)
        self._extracted_problems: set[int] = set()

        # Pools for fully solved/unsolved problems (min-heaps keyed by last_tested_step)
        # Problems with oldest last_tested_step are extracted first for retesting
        self._solved_pool = MinHeap(num_problems)
        self._unsolved_pool = MinHeap(num_problems)

        # Track which pool each problem is in (for fast lookup)
        # Values: 'heap', 'solved', 'unsolved', 'extracted'
        self._problem_location: dict[int, str] = {}
        for pid in range(num_problems):
            self._problem_location[pid] = "heap"

        # Track original location before extraction (for flip detection)
        # This remembers where a problem was before it was extracted
        self._extracted_from: dict[int, str] = {}  # pid -> 'heap', 'solved', 'unsolved'

    def record_results(self, problem_ids: np.ndarray, rewards: np.ndarray) -> dict:
        """
        Update success rates and priorities based on rollout results.

        This method should be called after computing rewards for a batch of rollouts.
        It groups rewards by problem_id, computes mean reward per problem, updates
        EMA success rates, and re-inserts problems into the heap with updated priorities.

        Args:
            problem_ids: Array of problem IDs for each rollout sample.
                        Shape: (num_samples,) where samples may be grouped by problem.
            rewards: Array of reward values for each rollout sample.
                    Shape: (num_samples,) matching problem_ids.
                    Values should be in [0, 1] range for proper priority computation.

        Returns:
            Dictionary with statistics:
                - 'num_problems_updated': Number of unique problems updated
                - 'mean_success_rate': Average success rate of updated problems
                - 'mean_priority': Average priority of updated problems
        """
        # Group rewards by problem_id
        problem_rewards = defaultdict(list)
        for pid, reward in zip(problem_ids, rewards):
            problem_rewards[int(pid)].append(float(reward))

        # Compute mean per group (this is μ_g from GRPO)
        group_means = {pid: np.mean(rewards_list) for pid, rewards_list in problem_rewards.items()}

        # Update using pre-computed group means
        return self.record_group_success_rates(group_means)

    def record_group_success_rates(self, group_success_rates: dict, is_retest: bool = False) -> dict:
        """
        Update success rates and priorities using pre-computed group success rates.

        This is more efficient when μ_g (group mean) is already computed (e.g., in GRPO).
        It directly uses the provided success rates instead of recomputing from raw rewards.

        Problems with μ_g = 1 (fully solved) or μ_g = 0 (fully unsolved) are moved to
        separate pools instead of being reinserted into the main heap.

        Args:
            group_success_rates: Dictionary mapping problem_id -> success_rate (μ_g).
                                The success rate should be in [0, 1] range.
            is_retest: If True, these are results from a retest batch.

        Returns:
            Dictionary with statistics:
                - 'num_problems_updated': Number of unique problems updated
                - 'mean_success_rate': Average success rate of updated problems
                - 'mean_priority': Average priority of updated problems
                - 'num_moved_to_solved': Problems moved to solved pool
                - 'num_moved_to_unsolved': Problems moved to unsolved pool
                - 'num_flipped_from_solved': Problems that flipped from solved
                - 'num_flipped_from_unsolved': Problems that flipped from unsolved
        """
        updated_pids = []
        updated_success_rates = []
        updated_priorities = []
        num_moved_to_solved = 0
        num_moved_to_unsolved = 0
        num_flipped_from_solved = 0
        num_flipped_from_unsolved = 0

        for pid, mean_reward in group_success_rates.items():
            pid = int(pid)  # Ensure integer type

            # Track if this problem was in a pool before (for flip detection)
            # Check both current location and where it was extracted from
            current_loc = self._problem_location.get(pid, "heap")
            extracted_from = self._extracted_from.get(pid, None)

            # Use extracted_from if available (problem was extracted), otherwise use current location
            original_loc = extracted_from if extracted_from else current_loc
            was_in_solved = original_loc == "solved"
            was_in_unsolved = original_loc == "unsolved"

            # Clear the extracted_from tracking since we're processing results
            if pid in self._extracted_from:
                del self._extracted_from[pid]

            # Update EMA success rate
            if self.seen_count[pid] == 0:
                # First observation: use the observed value directly
                self.ema_success_rate[pid] = mean_reward
            else:
                # EMA update: p_new = alpha * observation + (1 - alpha) * p_old
                self.ema_success_rate[pid] = (
                    self.alpha * mean_reward + (1 - self.alpha) * self.ema_success_rate[pid]
                )

            self.seen_count[pid] += 1
            self.last_tested_step[pid] = self.current_step

            # Compute new priority: omega = p * (1 - p) + success_bias (if p >= 0.5)
            # The bias breaks symmetry, preferring problems the model is almost solving
            p = self.ema_success_rate[pid]
            omega = p * (1 - p)
            if self.success_bias > 0 and p >= 0.5:
                omega += self.success_bias
            self.priorities[pid] = omega

            # Determine where this problem should go
            if mean_reward >= self.solved_threshold:
                # Fully solved - move to solved pool
                self._move_to_pool(pid, "solved")
                num_moved_to_solved += 1
                if was_in_unsolved:
                    num_flipped_from_unsolved += 1
            elif mean_reward <= self.unsolved_threshold:
                # Fully unsolved - move to unsolved pool
                self._move_to_pool(pid, "unsolved")
                num_moved_to_unsolved += 1
                if was_in_solved:
                    num_flipped_from_solved += 1
            else:
                # Uncertain - move to main heap
                self._move_to_heap(pid, omega)
                if was_in_solved:
                    num_flipped_from_solved += 1
                elif was_in_unsolved:
                    num_flipped_from_unsolved += 1

            updated_pids.append(pid)
            updated_success_rates.append(p)
            updated_priorities.append(omega)

        return {
            "num_problems_updated": len(updated_pids),
            "mean_success_rate": np.mean(updated_success_rates) if updated_success_rates else 0.0,
            "mean_priority": np.mean(updated_priorities) if updated_priorities else 0.0,
            "num_moved_to_solved": num_moved_to_solved,
            "num_moved_to_unsolved": num_moved_to_unsolved,
            "num_flipped_from_solved": num_flipped_from_solved,
            "num_flipped_from_unsolved": num_flipped_from_unsolved,
        }

    def _move_to_pool(self, pid: int, pool_name: str) -> None:
        """
        Move a problem to the solved or unsolved pool.

        Args:
            pid: Problem ID.
            pool_name: Either 'solved' or 'unsolved'.
        """
        current_location = self._problem_location.get(pid, "heap")

        # Remove from current location
        if current_location == "heap":
            if pid in self._heap:
                # Remove from heap (extract and don't reinsert)
                # This is expensive, but happens rarely
                self._remove_from_heap(pid)
        elif current_location == "solved":
            if pid in self._solved_pool:
                self._solved_pool.remove(pid)
        elif current_location == "unsolved":
            if pid in self._unsolved_pool:
                self._unsolved_pool.remove(pid)
        elif current_location == "extracted":
            self._extracted_problems.discard(pid)

        # Add to target pool
        pool = self._solved_pool if pool_name == "solved" else self._unsolved_pool
        pool.insert(pid, self.last_tested_step[pid])
        self._problem_location[pid] = pool_name

    def _move_to_heap(self, pid: int, omega: float) -> None:
        """
        Move a problem to the main heap.

        Args:
            pid: Problem ID.
            omega: Priority score for the problem.
        """
        current_location = self._problem_location.get(pid, "heap")

        # Remove from current location
        if current_location == "solved":
            if pid in self._solved_pool:
                self._solved_pool.remove(pid)
        elif current_location == "unsolved":
            if pid in self._unsolved_pool:
                self._unsolved_pool.remove(pid)
        elif current_location == "extracted":
            self._extracted_problems.discard(pid)
        elif current_location == "heap":
            # Already in heap, just update priority
            self._heap.update_priority(pid, omega)
            return

        # Insert into heap
        self._heap.insert(pid, omega)
        self._problem_location[pid] = "heap"

    def _remove_from_heap(self, pid: int) -> None:
        """
        Remove a problem from the max heap.

        This is O(n) in worst case but happens rarely (only when moving to pools).
        """
        if pid not in self._heap.position:
            return

        pos = self._heap.position[pid]
        del self._heap.position[pid]

        if pos == len(self._heap.heap) - 1:
            self._heap.heap.pop()
            return

        # Move last element to this position
        last = self._heap.heap.pop()
        self._heap.heap[pos] = last
        self._heap.position[last] = pos

        # Restore heap property
        old_priority = self._heap.priority[pid]
        new_priority = self._heap.priority[last]
        if new_priority > old_priority:
            self._heap._sift_up(pos)
        else:
            self._heap._sift_down(pos)

    def get_batch(self, batch_size: int) -> list[int]:
        """
        Extract a batch of problem IDs with highest priority.

        Problems are removed from the heap and tracked in _extracted_problems.
        They will be re-inserted when record_results() is called.

        Args:
            batch_size: Number of problems to extract.

        Returns:
            List of problem IDs with highest priorities.

        Raises:
            ValueError: If batch_size exceeds available problems in heap.
        """
        if batch_size > len(self._heap):
            raise ValueError(
                f"Requested batch_size={batch_size} but only {len(self._heap)} "
                f"problems available in heap. {len(self._extracted_problems)} problems "
                f"are currently extracted and awaiting results."
            )

        batch = []
        for _ in range(batch_size):
            pid = self._heap.extract_max()
            batch.append(pid)
            self._extracted_problems.add(pid)
            self._extracted_from[pid] = "heap"  # Track where it came from
            self._problem_location[pid] = "extracted"

        return batch

    def get_retest_problems(
        self,
        n_from_solved: int = 0,
        n_from_unsolved: int = 0,
    ) -> tuple[list[int], list[int]]:
        """
        Get problems from solved/unsolved pools for retesting.

        Extracts the least-recently-tested problems from each pool.

        Args:
            n_from_solved: Number of problems to extract from solved pool.
            n_from_unsolved: Number of problems to extract from unsolved pool.

        Returns:
            Tuple of (solved_problems, unsolved_problems) lists.
        """
        solved_problems = []
        unsolved_problems = []

        # Extract from solved pool (least recently tested first)
        for _ in range(min(n_from_solved, len(self._solved_pool))):
            pid = self._solved_pool.extract_min()
            solved_problems.append(pid)
            self._extracted_problems.add(pid)
            self._extracted_from[pid] = "solved"  # Track where it came from
            self._problem_location[pid] = "extracted"

        # Extract from unsolved pool (least recently tested first)
        for _ in range(min(n_from_unsolved, len(self._unsolved_pool))):
            pid = self._unsolved_pool.extract_min()
            unsolved_problems.append(pid)
            self._extracted_problems.add(pid)
            self._extracted_from[pid] = "unsolved"  # Track where it came from
            self._problem_location[pid] = "extracted"

        return solved_problems, unsolved_problems

    def get_batch_with_retest(
        self,
        batch_size: int,
        n_retest_solved: int = 0,
        n_retest_unsolved: int = 0,
    ) -> dict:
        """
        Get a batch that includes both priority-sampled and retest problems.

        This is the main method to call when you want to include retesting
        in regular training batches.

        Args:
            batch_size: Total desired batch size.
            n_retest_solved: Number of problems to retest from solved pool.
            n_retest_unsolved: Number of problems to retest from unsolved pool.

        Returns:
            Dictionary with:
                - 'problem_ids': List of all problem IDs in the batch.
                - 'from_heap': List of problem IDs from the main heap.
                - 'from_solved': List of problem IDs retested from solved pool.
                - 'from_unsolved': List of problem IDs retested from unsolved pool.
        """
        # First, get retest problems
        from_solved, from_unsolved = self.get_retest_problems(
            n_from_solved=n_retest_solved,
            n_from_unsolved=n_retest_unsolved,
        )

        # Calculate how many more we need beyond retest
        n_retest_total = len(from_solved) + len(from_unsolved)
        remaining = max(0, batch_size - n_retest_total)

        # Fill remaining slots with priority: heap > unsolved > solved
        from_heap = []
        fallback_unsolved = []
        fallback_solved = []

        # 1. First, try to fill from heap (highest priority)
        heap_size = len(self._heap)
        if remaining > 0 and heap_size > 0:
            n_from_heap = min(remaining, heap_size)
            for _ in range(n_from_heap):
                pid = self._heap.extract_max()
                from_heap.append(pid)
                self._extracted_problems.add(pid)
                self._extracted_from[pid] = "heap"
                self._problem_location[pid] = "extracted"
            remaining -= len(from_heap)

        # 2. If still need more, fall back to unsolved pool
        unsolved_size = len(self._unsolved_pool)
        if remaining > 0 and unsolved_size > 0:
            n_fallback = min(remaining, unsolved_size)
            _, fallback_unsolved = self.get_retest_problems(
                n_from_solved=0,
                n_from_unsolved=n_fallback,
            )
            remaining -= len(fallback_unsolved)

        # 3. If still need more, fall back to solved pool
        solved_size = len(self._solved_pool)
        if remaining > 0 and solved_size > 0:
            n_fallback = min(remaining, solved_size)
            fallback_solved, _ = self.get_retest_problems(
                n_from_solved=n_fallback,
                n_from_unsolved=0,
            )
            remaining -= len(fallback_solved)

        # Combine all problem IDs
        all_problems = from_heap + from_solved + from_unsolved + fallback_unsolved + fallback_solved

        return {
            "problem_ids": all_problems,
            "from_heap": from_heap,
            "from_solved": from_solved + fallback_solved,
            "from_unsolved": from_unsolved + fallback_unsolved,
        }

    def set_current_step(self, step: int) -> None:
        """
        Update the current training step.

        This should be called at the beginning of each training step
        so that last_tested_step is tracked correctly.

        Args:
            step: Current training step number.
        """
        self.current_step = step

    def should_retest(self, current_step: int, n_retest_steps: int) -> bool:
        """
        Check if it's time to perform a retest.

        Args:
            current_step: Current training step.
            n_retest_steps: Retest every n_retest_steps steps.

        Returns:
            True if retesting should be performed this step.
        """
        if n_retest_steps <= 0:
            return False
        return current_step > 0 and current_step % n_retest_steps == 0

    def reinsert_problems(self, problem_ids: list[int]) -> None:
        """
        Manually re-insert problems to their original location without updating priorities.

        Use this if problems need to be re-inserted without new reward data
        (e.g., if rollout failed or was skipped, or batch was dropped).

        Args:
            problem_ids: List of problem IDs to re-insert.
        """
        for pid in problem_ids:
            if pid not in self._extracted_problems:
                continue
            
            # Get original location (where problem was before extraction)
            original_loc = self._extracted_from.get(pid, "heap")
            
            # Reinsert to original location
            if original_loc == "solved":
                self._solved_pool.insert(pid, self.last_tested_step[pid])
                self._problem_location[pid] = "solved"
            elif original_loc == "unsolved":
                self._unsolved_pool.insert(pid, self.last_tested_step[pid])
                self._problem_location[pid] = "unsolved"
            else:  # "heap" or unknown
                self._heap.insert(pid, self.priorities[pid])
                self._problem_location[pid] = "heap"
            
            self._extracted_problems.discard(pid)
            if pid in self._extracted_from:
                del self._extracted_from[pid]

    @property
    def heap_size(self) -> int:
        """Number of problems currently in the heap."""
        return len(self._heap)

    @property
    def extracted_count(self) -> int:
        """Number of problems currently extracted (awaiting results)."""
        return len(self._extracted_problems)

    @property
    def solved_pool_size(self) -> int:
        """Number of problems in the solved pool."""
        return len(self._solved_pool)

    @property
    def unsolved_pool_size(self) -> int:
        """Number of problems in the unsolved pool."""
        return len(self._unsolved_pool)

    def get_statistics(self) -> dict:
        """
        Get current statistics about the priority manager.

        Returns:
            Dictionary with:
                - 'num_seen': Number of problems seen at least once
                - 'num_unseen': Number of problems never seen
                - 'mean_success_rate': Mean success rate across seen problems
                - 'std_success_rate': Std of success rates across seen problems
                - 'mean_priority': Mean priority across all problems
                - 'heap_size': Current heap size
                - 'extracted_count': Problems awaiting results
                - 'solved_pool_size': Problems in solved pool
                - 'unsolved_pool_size': Problems in unsolved pool
        """
        seen_mask = self.seen_count > 0
        num_seen = seen_mask.sum()

        return {
            "num_seen": int(num_seen),
            "num_unseen": int(self.num_problems - num_seen),
            "mean_success_rate": float(self.ema_success_rate[seen_mask].mean()) if num_seen > 0 else 0.0,
            "std_success_rate": float(self.ema_success_rate[seen_mask].std()) if num_seen > 0 else 0.0,
            "mean_priority": float(self.priorities[self.priorities < float("inf")].mean())
            if (self.priorities < float("inf")).any()
            else 0.0,
            "heap_size": self.heap_size,
            "extracted_count": self.extracted_count,
            "solved_pool_size": self.solved_pool_size,
            "unsolved_pool_size": self.unsolved_pool_size,
        }

    def state_dict(self) -> dict:
        """
        Get the state of the priority manager for checkpointing.

        Returns:
            Dictionary containing all state needed to restore the priority manager.
        """
        return {
            # Config (for validation on load)
            "num_problems": self.num_problems,
            "alpha": self.alpha,
            "initial_omega": self.initial_omega,
            "initial_success_rate": self.initial_success_rate,
            "solved_threshold": self.solved_threshold,
            "unsolved_threshold": self.unsolved_threshold,
            "success_bias": self.success_bias,
            # Per-problem arrays
            "ema_success_rate": self.ema_success_rate.copy(),
            "priorities": self.priorities.copy(),
            "seen_count": self.seen_count.copy(),
            "last_tested_step": self.last_tested_step.copy(),
            # Current step
            "current_step": self.current_step,
            # Heap state
            "heap_heap": list(self._heap.heap),
            "heap_priority": self._heap.priority.copy(),
            "heap_position": dict(self._heap.position),
            # Solved pool state
            "solved_pool_heap": list(self._solved_pool.heap),
            "solved_pool_priority": self._solved_pool.priority.copy(),
            "solved_pool_position": dict(self._solved_pool.position),
            # Unsolved pool state
            "unsolved_pool_heap": list(self._unsolved_pool.heap),
            "unsolved_pool_priority": self._unsolved_pool.priority.copy(),
            "unsolved_pool_position": dict(self._unsolved_pool.position),
            # Tracking dicts/sets
            "extracted_problems": list(self._extracted_problems),
            "problem_location": dict(self._problem_location),
            "extracted_from": dict(self._extracted_from),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Restore the priority manager state from a checkpoint.

        Args:
            state_dict: Dictionary from state_dict() containing saved state.
        """
        # Validate config matches
        if state_dict["num_problems"] != self.num_problems:
            raise ValueError(
                f"Cannot load state: num_problems mismatch "
                f"(checkpoint={state_dict['num_problems']}, current={self.num_problems})"
            )

        # Restore per-problem arrays
        self.ema_success_rate = state_dict["ema_success_rate"].copy()
        self.priorities = state_dict["priorities"].copy()
        self.seen_count = state_dict["seen_count"].copy()
        self.last_tested_step = state_dict["last_tested_step"].copy()

        # Restore current step
        self.current_step = state_dict["current_step"]

        # Restore heap
        self._heap.heap = list(state_dict["heap_heap"])
        self._heap.priority = state_dict["heap_priority"].copy()
        self._heap.position = dict(state_dict["heap_position"])

        # Restore solved pool
        self._solved_pool.heap = list(state_dict["solved_pool_heap"])
        self._solved_pool.priority = state_dict["solved_pool_priority"].copy()
        self._solved_pool.position = dict(state_dict["solved_pool_position"])

        # Restore unsolved pool
        self._unsolved_pool.heap = list(state_dict["unsolved_pool_heap"])
        self._unsolved_pool.priority = state_dict["unsolved_pool_priority"].copy()
        self._unsolved_pool.position = dict(state_dict["unsolved_pool_position"])

        # Restore tracking dicts/sets
        self._extracted_problems = set(state_dict["extracted_problems"])
        self._problem_location = dict(state_dict["problem_location"])
        self._extracted_from = dict(state_dict["extracted_from"])

        print(f"Loaded priority manager state: "
              f"heap={len(self._heap)}, solved={len(self._solved_pool)}, "
              f"unsolved={len(self._unsolved_pool)}, seen={self.seen_count.sum()}")


# =============================================================================
# Priority Batch Sampler
# =============================================================================


class PriorityBatchSampler(Sampler):
    """
    Batch sampler that selects top-priority problems using ProblemPriorityManager.

    This sampler integrates with VeRL's training loop and works with datasets
    that include problem_id (like RLDatasetWithProblemId).

    The sampler yields batches of problem indices based on their priority scores.
    Problems with higher omega = p * (1 - p) values are sampled first.

    Supports exploration via `explore_prob` parameter:
    - With probability `explore_prob`, samples are drawn uniformly at random.
    - With probability `1 - explore_prob`, samples are drawn based on priority.

    Also supports periodic retesting of solved/unsolved problems.

    Usage:
        priority_manager = ProblemPriorityManager(num_problems=len(dataset), ...)
        sampler = PriorityBatchSampler(
            priority_manager=priority_manager,
            batch_size=32,
            num_batches_per_epoch=100,
            explore_prob=0.1,  # 10% random exploration
        )
        dataloader = DataLoader(dataset, batch_sampler=sampler)

        for batch in dataloader:
            # batch contains samples with problem_ids
            ...
            # After computing rewards:
            priority_manager.record_results(problem_ids, rewards)

    Attributes:
        priority_manager: The ProblemPriorityManager instance.
        batch_size: Number of problems per batch.
        num_batches_per_epoch: Number of batches to yield per epoch.
        explore_prob: Probability of random exploration.
        retest_n_steps: Retest every n steps (0 = disabled).
        retest_n_from_solved: Number of problems to retest from solved pool.
        retest_n_from_unsolved: Number of problems to retest from unsolved pool.
    """

    def __init__(
        self,
        priority_manager: ProblemPriorityManager,
        batch_size: int,
        num_batches_per_epoch: Optional[int] = None,
        drop_last: bool = True,
        explore_prob: float = 0.0,
        seed: Optional[int] = None,
        retest_n_steps: int = 0,
        retest_n_from_solved: int = 4,
        retest_n_from_unsolved: int = 4,
    ):
        """
        Initialize the priority batch sampler.

        Args:
            priority_manager: ProblemPriorityManager instance that tracks priorities.
            batch_size: Number of problems to sample per batch.
            num_batches_per_epoch: Number of batches per epoch. If None, computed as
                                   num_problems // batch_size.
            drop_last: If True and num_batches_per_epoch is None, drop incomplete batch.
            explore_prob: Probability of random exploration (0.0 to 1.0). Default: 0.0.
            seed: Random seed for reproducibility. Default: None.
            retest_n_steps: Retest every n steps (0 to disable). Default: 0.
            retest_n_from_solved: Problems to retest from solved pool. Default: 4.
            retest_n_from_unsolved: Problems to retest from unsolved pool. Default: 4.
        """
        self.priority_manager = priority_manager
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.explore_prob = explore_prob
        self._rng = np.random.default_rng(seed)

        # Retest configuration
        self.retest_n_steps = retest_n_steps
        self.retest_n_from_solved = retest_n_from_solved
        self.retest_n_from_unsolved = retest_n_from_unsolved
        self._internal_step = 0

        if num_batches_per_epoch is None:
            num_problems = priority_manager.num_problems
            if drop_last:
                self.num_batches_per_epoch = num_problems // batch_size
            else:
                self.num_batches_per_epoch = (num_problems + batch_size - 1) // batch_size
        else:
            self.num_batches_per_epoch = num_batches_per_epoch

    def _get_random_batch(self, batch_size: int) -> list[int]:
        """Get a random batch of problem IDs for exploration.
        
        Samples uniformly from all available problems (heap + pools),
        properly extracting each from its current location.
        """
        # Sample from all problems uniformly at random
        all_pids = list(range(self.priority_manager.num_problems))
        # Filter out already extracted problems
        available_pids = [pid for pid in all_pids if pid not in self.priority_manager._extracted_problems]

        if len(available_pids) < batch_size:
            # Not enough available, return what we have
            batch = available_pids
        else:
            batch = self._rng.choice(available_pids, size=batch_size, replace=False).tolist()

        # Properly extract each problem from its current location
        for pid in batch:
            current_loc = self.priority_manager._problem_location.get(pid, "heap")
            
            # Remove from current location
            if current_loc == "heap":
                if pid in self.priority_manager._heap:
                    # Remove from heap (need to use internal removal)
                    self.priority_manager._remove_from_heap(pid)
            elif current_loc == "solved":
                if pid in self.priority_manager._solved_pool:
                    self.priority_manager._solved_pool.remove(pid)
            elif current_loc == "unsolved":
                if pid in self.priority_manager._unsolved_pool:
                    self.priority_manager._unsolved_pool.remove(pid)
            
            # Track where it came from (for flip detection in record_group_success_rates)
            self.priority_manager._extracted_from[pid] = current_loc
            
            # Mark as extracted
            self.priority_manager._extracted_problems.add(pid)
            self.priority_manager._problem_location[pid] = "extracted"

        return batch

    def _should_retest(self) -> bool:
        """Check if this step should include retest problems."""
        if self.retest_n_steps <= 0:
            return False
        return self._internal_step > 0 and self._internal_step % self.retest_n_steps == 0

    def get_batch_with_metadata(self) -> dict:
        """
        Get a batch with metadata about where problems came from.

        Returns:
            Dictionary with:
                - 'problem_ids': List of all problem IDs in the batch.
                - 'from_heap': List of problem IDs from the main heap.
                - 'from_solved': List of problem IDs retested from solved pool.
                - 'from_unsolved': List of problem IDs retested from unsolved pool.
                - 'is_retest_batch': Whether this batch includes retest problems.
        """
        self._internal_step += 1
        self.priority_manager.set_current_step(self._internal_step)

        # Check if we should include retest problems
        is_retest = self._should_retest()

        if is_retest:
            result = self.priority_manager.get_batch_with_retest(
                batch_size=self.batch_size,
                n_retest_solved=self.retest_n_from_solved,
                n_retest_unsolved=self.retest_n_from_unsolved,
            )
            result["is_retest_batch"] = True
            return result
        else:
            # Normal batch - decide between exploration and exploitation
            if self.explore_prob > 0 and self._rng.random() < self.explore_prob:
                # Exploration: random sampling
                available = self.priority_manager.num_problems - self.priority_manager.extracted_count
                current_batch_size = min(self.batch_size, available)
                batch = self._get_random_batch(current_batch_size) if current_batch_size > 0 else []
                from_heap = batch
                from_solved = []
                from_unsolved = []
            else:
                # Exploitation: priority-based sampling
                # Priority order: heap > unsolved pool > solved pool
                from_heap = []
                from_unsolved = []
                from_solved = []
                remaining = self.batch_size
                
                # 1. First, sample from heap (highest priority - these are "trainable")
                heap_available = self.priority_manager.heap_size
                if heap_available > 0:
                    n_from_heap = min(remaining, heap_available)
                    from_heap = self.priority_manager.get_batch(n_from_heap)
                    remaining -= len(from_heap)
                else:
                    # Heap is empty - WARNING: no trainable problems!
                    # All problems are either fully solved (μ_g=1) or fully unsolved (μ_g=0).
                    # Training signal will be zero or near-zero (wasted compute).
                    logger.warning(
                        f"Priority heap is empty! No trainable problems available. "
                        f"solved_pool={self.priority_manager.solved_pool_size}, "
                        f"unsolved_pool={self.priority_manager.unsolved_pool_size}. "
                        f"Falling back to pool sampling (likely zero gradient)."
                    )
                
                # 2. If still need more, sweep unsolved pool (oldest first)
                # Unsolved problems might become solvable as model improves
                if remaining > 0 and self.priority_manager.unsolved_pool_size > 0:
                    n_from_unsolved = min(remaining, self.priority_manager.unsolved_pool_size)
                    # get_retest_problems returns (solved_list, unsolved_list)
                    _, from_unsolved = self.priority_manager.get_retest_problems(
                        n_from_solved=0,
                        n_from_unsolved=n_from_unsolved,
                    )
                    remaining -= len(from_unsolved)
                
                # 3. If still need more, try solved pool (least useful, but better than nothing)
                if remaining > 0 and self.priority_manager.solved_pool_size > 0:
                    n_from_solved = min(remaining, self.priority_manager.solved_pool_size)
                    # get_retest_problems returns (solved_list, unsolved_list)
                    from_solved, _ = self.priority_manager.get_retest_problems(
                        n_from_solved=n_from_solved,
                        n_from_unsolved=0,
                    )
                    remaining -= len(from_solved)
                
                batch = from_heap + from_unsolved + from_solved

            return {
                "problem_ids": batch,
                "from_heap": from_heap,
                "from_solved": from_solved,
                "from_unsolved": from_unsolved,
                "is_retest_batch": len(from_solved) > 0 or len(from_unsolved) > 0,
            }

    def __iter__(self) -> Iterator[list[int]]:
        """
        Yield batches of problem indices.

        With probability explore_prob, yields random batches.
        Otherwise, yields priority-based batches.
        Periodically includes retest problems from solved/unsolved pools.

        Yields:
            List of problem indices (problem_ids) for each batch.
        """
        for _ in range(self.num_batches_per_epoch):
            batch_info = self.get_batch_with_metadata()
            batch = batch_info["problem_ids"]

            if len(batch) == 0:
                continue
            if len(batch) < self.batch_size and self.drop_last:
                # IMPORTANT: Reinsert problems that were extracted but won't be used
                # Otherwise they stay in _extracted_problems forever!
                self.priority_manager.reinsert_problems(batch)
                continue

            yield batch

    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return self.num_batches_per_epoch

    def state_dict(self) -> dict:
        """
        Get the state of the sampler for checkpointing.

        Returns:
            Dictionary containing sampler state and priority manager state.
        """
        return {
            "internal_step": self._internal_step,
            "priority_manager": self.priority_manager.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Restore the sampler state from a checkpoint.

        Args:
            state_dict: Dictionary from state_dict() containing saved state.
        """
        self._internal_step = state_dict["internal_step"]
        self.priority_manager.load_state_dict(state_dict["priority_manager"])
        print(f"Loaded sampler state: internal_step={self._internal_step}")


# =============================================================================
# Factory Functions
# =============================================================================


def create_priority_sampler(
    dataset_size: int,
    batch_size: int,
    config: dict,
    seed: Optional[int] = None,
) -> tuple[ProblemPriorityManager, PriorityBatchSampler]:
    """
    Factory function to create a priority sampler from config.

    Args:
        dataset_size: Number of problems in the dataset.
        batch_size: Number of problems per batch.
        config: Configuration dictionary or PrioritySamplerConfig with keys:
            - 'enabled': Whether priority sampling is enabled (checked externally)
            - 'alpha': EMA coefficient (default: 1.0)
            - 'explore_prob': Probability of random exploration (default: 0.0)
            - 'initial_omega': Initial priority (default: inf)
            - 'initial_success_rate': Initial success rate (default: 0.5)
            - 'solved_threshold': μ_g threshold for "solved" (default: 1.0)
            - 'unsolved_threshold': μ_g threshold for "unsolved" (default: 0.0)
            - 'num_batches_per_epoch': Batches per epoch (default: None)
            - 'drop_last': Drop incomplete batches (default: True)
            - 'retest': Retest configuration sub-dict with:
                - 'n_steps': Retest every n steps (default: 0)
                - 'n_from_solved': Problems to retest from solved pool (default: 4)
                - 'n_from_unsolved': Problems to retest from unsolved pool (default: 4)
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (ProblemPriorityManager, PriorityBatchSampler).
    """
    # Support both dict and dataclass-like config
    def get_val(obj, key, default):
        if hasattr(obj, "get"):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # Extract config values with defaults
    alpha = get_val(config, "alpha", 1.0)
    explore_prob = get_val(config, "explore_prob", 0.0)
    initial_omega = get_val(config, "initial_omega", float("inf"))
    initial_success_rate = get_val(config, "initial_success_rate", 0.5)
    solved_threshold = get_val(config, "solved_threshold", 1.0)
    unsolved_threshold = get_val(config, "unsolved_threshold", 0.0)
    success_bias = get_val(config, "success_bias", 0.0)
    num_batches_per_epoch = get_val(config, "num_batches_per_epoch", None)
    drop_last = get_val(config, "drop_last", True)

    # Extract retest config
    retest_config = get_val(config, "retest", {})
    retest_n_steps = get_val(retest_config, "n_steps", 0) if retest_config else 0
    retest_n_from_solved = get_val(retest_config, "n_from_solved", 4) if retest_config else 4
    retest_n_from_unsolved = get_val(retest_config, "n_from_unsolved", 4) if retest_config else 4

    # Create priority manager
    priority_manager = ProblemPriorityManager(
        num_problems=dataset_size,
        alpha=alpha,
        initial_omega=initial_omega,
        initial_success_rate=initial_success_rate,
        solved_threshold=solved_threshold,
        unsolved_threshold=unsolved_threshold,
        success_bias=success_bias,
    )

    # Create sampler
    sampler = PriorityBatchSampler(
        priority_manager=priority_manager,
        batch_size=batch_size,
        num_batches_per_epoch=num_batches_per_epoch,
        drop_last=drop_last,
        explore_prob=explore_prob,
        seed=seed,
        retest_n_steps=retest_n_steps,
        retest_n_from_solved=retest_n_from_solved,
        retest_n_from_unsolved=retest_n_from_unsolved,
    )

    return priority_manager, sampler


def is_priority_sampler_enabled(data_config) -> bool:
    """
    Check if priority sampling is enabled in the data config.

    Args:
        data_config: The data configuration (OmegaConf DictConfig or dict).

    Returns:
        True if priority sampling is enabled, False otherwise.
    """
    if hasattr(data_config, "get"):
        priority_config = data_config.get("priority_sampler", None)
        if priority_config is None:
            return False
        if hasattr(priority_config, "get"):
            return priority_config.get("enabled", False)
        return getattr(priority_config, "enabled", False)
    return False


def create_priority_sampler_from_data_config(
    dataset,
    data_config,
    seed: Optional[int] = None,
) -> tuple[Optional[ProblemPriorityManager], Optional[PriorityBatchSampler]]:
    """
    Create a priority sampler from the data configuration.

    This is a convenience function that:
    1. Checks if priority sampling is enabled
    2. Wraps the dataset with RLDatasetWithProblemId if needed
    3. Creates the priority manager and sampler

    Args:
        dataset: The base dataset (will be wrapped with RLDatasetWithProblemId).
        data_config: The data configuration containing priority_sampler settings.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (ProblemPriorityManager, PriorityBatchSampler) if enabled,
        or (None, None) if disabled.
    """
    if not is_priority_sampler_enabled(data_config):
        return None, None

    priority_config = data_config.get("priority_sampler", {})
    batch_size = data_config.get("train_batch_size", 32)

    # Use seed from data_config if not explicitly provided
    if seed is None:
        seed = data_config.get("seed", None)

    return create_priority_sampler(
        dataset_size=len(dataset),
        batch_size=batch_size,
        config=priority_config,
        seed=seed,
    )
