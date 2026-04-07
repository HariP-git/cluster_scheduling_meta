# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Base class for all scheduler pipeline stage modules.

Every stage module follows the same uniform interface (Strategy Pattern):
    Config → Module(config) → execute(cluster, context) → get_report()

The environment doesn't know or care what each module does internally.
It just calls the same interface. You can swap, add, or remove modules
without touching the environment code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ModuleConfig:
    """Base configuration for scheduler modules. Subclass to add knobs."""

    pass


class SchedulerModule(ABC):
    """
    Abstract base for all pipeline stage modules.

    Subclasses must implement:
        _default_config() → ModuleConfig
        execute(cluster, context) → dict
    """

    def __init__(self, config: ModuleConfig | None = None):
        self.config = config or self._default_config()
        self._report: dict = {}

    @abstractmethod
    def _default_config(self) -> ModuleConfig:
        """Return default configuration for this module."""
        ...

    @abstractmethod
    def execute(self, cluster, context: dict) -> dict:
        """
        Execute this stage's logic.

        Args:
            cluster: The Cluster object (shared domain state).
            context: Accumulated context from previous stages.

        Returns:
            dict of results to merge into the pipeline context.
        """
        ...

    def get_report(self) -> dict:
        """Return a report of what this module did."""
        return self._report
