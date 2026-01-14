"""
Example usage of cleaned-up spatial acceleration API.

Shows how to use typed dataclasses and automatic strategy selection.
"""
import numpy as np
import jax.numpy as jnp
from wpsim.spatial import (
	buildSpatialAcceleration, querySpatial, SpatialStrategy
)


def exampleBasicUsage():
	"""Basic usage with automatic strategy selection"""

	# Create some test triangles
	surfaceTris = jnp.array([
		[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
		[[1, 0, 0], [1, 1, 0], [0, 1, 0]],
		[[0, 0, 1], [1, 0, 1], [0, 1, 1]],
	])  # (3, 3, 3)

	# Build spatial acceleration with automatic strategy
	gridData, strategy, config = buildSpatialAcceleration(
		surfaceTris,
		cellSize=0.1,
		strategy=SpatialStrategy.AUTO  # automatic selection
	)

	print(f"Selected strategy: {strategy.value}")
	print(f"Config: cellSize={config.cellSize}, useNeighborhood={config.useNeighborhood}")

	# Query some points
	queryPoints = jnp.array([
		[0.5, 0.5, 0.0],
		[0.2, 0.3, 0.5],
	])  # (2, 3)

	results = querySpatial(queryPoints, gridData, surfaceTris, strategy, config)

	print(f"Best distances: {results.bestDistSq}")
	print(f"Best triangle indices: {results.bestTriIdx}")
	print(f"Closest points:\n{results.bestPoint}")


def exampleExplicitStrategy():
	"""Usage with explicit strategy selection"""

	#surfaceTris = jnp.random.uniform(-1, 1, (100, 3, 3))
	surfaceTris = jnp.array(np.random.uniform(-1, 1, (100, 3, 3)))

	# Force BVH strategy
	gridData, strategy, config = buildSpatialAcceleration(
		surfaceTris,
		cellSize=0.05,
		strategy=SpatialStrategy.BVH  # explicit choice
	)

	print(f"Forced strategy: {strategy.value}")

	queryPoints = jnp.array(np.random.uniform(-1, 1, (10, 3)))
	results = querySpatial(queryPoints, gridData, surfaceTris, strategy, config)

	print(f"Queried {len(queryPoints)} points")
	print(f"Found {jnp.sum(results.bestTriIdx >= 0)} valid matches")


def exampleCustomConfig():
	"""Usage with custom configuration"""

	surfaceTris = jnp.array(np.random.uniform(-5, 5, (200, 3, 3)))

	# Build with custom parameters
	gridData, strategy, config = buildSpatialAcceleration(
		surfaceTris,
		cellSize=0.2,
		tableSize=131072,  # larger hash table
		maxBucketSearch=32,  # allow more triangles per bucket
		useNeighborhood=True,  # enable 27-cell search
		strategy=SpatialStrategy.AUTO
	)

	print(f"Strategy: {strategy.value}")
	print(f"Config: tableSize={config.tableSize}, maxBucketSearch={config.maxBucketSearch}")

	# Check for overflow if using hash strategy
	if strategy == SpatialStrategy.MULTI_CELL_HASH:
		print(f"Max bucket size: {gridData.maxBucketSize}")
		print(f"Overflow count: {gridData.overflowCount}")
		if gridData.overflowCount > 0:
			print(f"Warning: {gridData.overflowCount} buckets exceed maxBucketSearch!")


if __name__ == "__main__":
	print("=== Basic Usage ===")
	exampleBasicUsage()

	print("\n=== Explicit Strategy ===")
	exampleExplicitStrategy()

	print("\n=== Custom Config ===")
	exampleCustomConfig()
