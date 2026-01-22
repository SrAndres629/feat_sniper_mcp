import warnings
from .physics_engine.engine import physics_engine, PhysicsEngine

# [DEPRECATED] Use nexus_core.physics_engine.engine instead
warnings.warn("kinetic_engine is deprecated. Use physics_engine instead.", DeprecationWarning)

kinetic_engine = physics_engine
KineticEngine = PhysicsEngine
