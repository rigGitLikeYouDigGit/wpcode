"""
wpplan - Multi-goal task planning and exploration system

Core components:
- State: Discrete and continuous world state
- Action: Property-based action templates
- Goal: Target conditions with satisfaction predicates
- Agent: Knowledge, beliefs, and planning
- Planner: Search with leverage/influence heuristics

next:
- add a "gather available actions" step, so agent looks round known objects
and generates actions available, based on its knowledge
- consider a way to emulate thought processes of other agents?
- collate physical movement tasks into something like a navmesh - use as
prototype for hierarchical grouping, or learning
"""

from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from wpplan.state import State, Variable, Value
from wpplan.action import Action, ActionTemplate
from wpplan.goal import Goal
from wpplan.influence import InfluenceGraph
from wpplan.reversible import ReversibleState, Transaction, try_action, commit_action
from wpplan.action_cache import ActionCache, SmartActionGenerator
from wpplan.relations import Relation, RelationType, RelationGraph, create_containment_relation

__all__ = [
    'State',
    'Variable',
    'Value',
    'Action',
    'ActionTemplate',
    'Goal',
    'InfluenceGraph',
    'ReversibleState',
    'Transaction',
    'try_action',
    'commit_action',
    'ActionCache',
    'SmartActionGenerator',
    'Relation',
    'RelationType',
    'RelationGraph',
    'create_containment_relation',
]
