from __future__ import annotations
import types, typing as T
import pprint
import threading
import time
import json
import uuid
import zmq
from wplib import log

class Discoverable:
	"""Component object discoverable on a network. Local by default.
	Objects grouped by pools, which are like namespaces. Each pool has a unique name, and objects within the pool have unique names as well.
	Discoverables in a pool echo all messages sent within the pool.

	Implementation uses ZeroMQ for messaging and discovery:
	- PUB/SUB pattern for pool-based message broadcasting
	- REQ/REP pattern for request-response messaging
	- Heartbeat system to detect dead instances

	object supports following functions:

	- start(): start the object, making it active on the network.
	- stop(): stop the object, making it inactive on the network.

	- setDiscoverable(state): make the object discoverable on the network,
	allowing other components to find it or not.

	- discover(): return a list of discoverable objects on the network, including their pool and name.
	- discover(pool): return a list of discoverable objects in the specified pool,
	- allPools(): return a list of all pools with discoverable objects on the network.
	- setPool(pool): set the pool for this object, which is like a namespace for discovery. Objects in the same pool can discover each other.

	- sendMessage(message): send a message to all discoverable objects in
	this pool. Objects in the same pool receive the message.
	- sendMessage(message, pool): send a message to all discoverable objects in
	the specified pool. Objects in the same pool receive the message.
	- sendMessage(message, target): send a message to a specific discoverable object on the network, identified by its pool and name. Only the target object receives the message (but may be echoed to its pool)

	- onMessage(callback): register a callback function to be called when a message is received by this object. The callback receives the message and metadata about the sender (pool, name).
	- onRequest(callback): register a callback function to be called when a request is received by this object. The callback receives the request message and metadata about the sender (pool, name), and should return a response message.
	"""

	# Class-level registry for all discoverable objects
	_registry: dict[str, dict[str, Discoverable]] = {}  # {pool: {name: instance}}
	_registryLock = threading.Lock()

	# Heartbeat tracking: {uuid: lastHeartbeatTime}
	_heartbeats: dict[str, float] = {}
	_heartbeatsLock = threading.Lock()

	# ZeroMQ context shared across all instances
	_zmqContext = None

	# Discovery and heartbeat settings
	DISCOVERY_PORT = 5670
	BEACON_INTERVAL = 1.0  # seconds
	HEARTBEAT_INTERVAL = 2.0  # seconds
	HEARTBEAT_TIMEOUT = 6.0  # seconds - consider dead if no heartbeat for this long

	def __init__(self, name: str, pool: str = "default"):
		"""Initialize a discoverable object.

		Args:
			name: Unique name for this object within its pool
			pool: Pool/namespace this object belongs to
		"""
		self.name = name
		self.pool = pool
		self.uuid = str(uuid.uuid4())

		self._discoverable = True
		self._active = False
		self._messageCallback = None
		self._requestCallback = None

		# ZeroMQ sockets
		self._pubSocket = None  # For publishing messages
		self._subSocket = None  # For receiving messages from pool
		self._repSocket = None  # For handling requests
		self._repPort = None

		# Threading
		self._listenerThread = None
		self._heartbeatThread = None
		self._cleanupThread = None
		self._stopEvent = threading.Event()

		# Initialize ZeroMQ context
		if Discoverable._zmqContext is None:
			Discoverable._zmqContext = zmq.Context()

	def start(self):
		"""Start the object, making it active on the network."""
		if self._active:
			log.warning(f"Discoverable {self.pool}/{self.name} already active")
			return

		self._stopEvent.clear()

		# Set up PUB socket for broadcasting messages
		self._pubSocket = Discoverable._zmqContext.socket(zmq.PUB)
		pubPort = self._pubSocket.bind_to_random_port("tcp://127.0.0.1")
		log.info(f"PUB socket bound to port {pubPort}")

		# Set up SUB socket for receiving pool messages
		self._subSocket = Discoverable._zmqContext.socket(zmq.SUB)
		# Subscribe to messages for our pool
		self._subSocket.setsockopt_string(zmq.SUBSCRIBE, self.pool)
		# Subscribe to heartbeat messages
		self._subSocket.setsockopt_string(zmq.SUBSCRIBE, "__heartbeat__")

		# Set up REP socket for handling requests
		self._repSocket = Discoverable._zmqContext.socket(zmq.REP)
		self._repPort = self._repSocket.bind_to_random_port("tcp://127.0.0.1")
		log.info(f"REP socket bound to port {self._repPort}")

		# Register this instance
		with Discoverable._registryLock:
			if self.pool not in Discoverable._registry:
				Discoverable._registry[self.pool] = {}
			Discoverable._registry[self.pool][self.name] = self

		# Register initial heartbeat
		with Discoverable._heartbeatsLock:
			Discoverable._heartbeats[self.uuid] = time.time()

		# Connect to other instances in the same pool
		self._connectToPool()

		# Start listener thread
		self._listenerThread = threading.Thread(target=self._listenLoop, daemon=True)
		self._listenerThread.start()

		# Start heartbeat thread
		self._heartbeatThread = threading.Thread(target=self._heartbeatLoop, daemon=True)
		self._heartbeatThread.start()

		# Start cleanup thread (removes dead instances)
		self._cleanupThread = threading.Thread(target=self._cleanupLoop, daemon=True)
		self._cleanupThread.start()

		self._active = True
		log.info(f"Discoverable {self.pool}/{self.name} started")

	def stop(self):
		"""Stop the object, making it inactive on the network."""
		if not self._active:
			return

		self._active = False
		self._stopEvent.set()

		# Send goodbye message before stopping
		self._sendGoodbye()

		# Unregister from registry
		with Discoverable._registryLock:
			if self.pool in Discoverable._registry and self.name in Discoverable._registry[self.pool]:
				del Discoverable._registry[self.pool][self.name]
				if not Discoverable._registry[self.pool]:
					del Discoverable._registry[self.pool]

		# Remove heartbeat entry
		with Discoverable._heartbeatsLock:
			if self.uuid in Discoverable._heartbeats:
				del Discoverable._heartbeats[self.uuid]

		# Close sockets
		if self._pubSocket:
			self._pubSocket.close()
		if self._subSocket:
			self._subSocket.close()
		if self._repSocket:
			self._repSocket.close()

		# Wait for threads to finish
		if self._listenerThread:
			self._listenerThread.join(timeout=2.0)
		if self._heartbeatThread:
			self._heartbeatThread.join(timeout=2.0)
		if self._cleanupThread:
			self._cleanupThread.join(timeout=2.0)

		log.info(f"Discoverable {self.pool}/{self.name} stopped")

	def setDiscoverable(self, state: bool):
		"""Make the object discoverable or hidden on the network."""
		self._discoverable = state
		log.info(f"Discoverable {self.pool}/{self.name} set to {state}")

	def setPool(self, pool: str):
		"""Change the pool for this object."""
		wasActive = self._active
		if wasActive:
			self.stop()

		self.pool = pool

		if wasActive:
			self.start()

	def discover(self, pool: str = None) -> list[dict]:
		"""Discover objects on the network.

		Args:
			pool: Optional pool name to filter by

		Returns:
			List of dicts with 'pool', 'name', and 'uuid' keys
		"""
		results = []
		currentTime = time.time()

		with Discoverable._registryLock:
			if pool:
				if pool in Discoverable._registry:
					for name, instance in Discoverable._registry[pool].items():
						if instance._discoverable and self._isAlive(instance.uuid, currentTime):
							results.append({
								'pool': instance.pool,
								'name': instance.name,
								'uuid': instance.uuid
							})
			else:
				for poolName, poolDict in Discoverable._registry.items():
					for name, instance in poolDict.items():
						if instance._discoverable and self._isAlive(instance.uuid, currentTime):
							results.append({
								'pool': instance.pool,
								'name': instance.name,
								'uuid': instance.uuid
							})
		return results

	def allPools(self) -> list[str]:
		"""Return list of all pools with discoverable objects."""
		with Discoverable._registryLock:
			return list(Discoverable._registry.keys())

	def sendMessage(self, message: T.Any, pool: str = None, target: str = None):
		"""Send a message to objects in the network.

		Args:
			message: Message to send (will be JSON serialized)
			pool: Optional pool to send to (defaults to this object's pool)
			target: Optional target object name for direct messaging
		"""
		if not self._active:
			log.warning(f"Cannot send message: {self.pool}/{self.name} not active")
			return

		targetPool = pool or self.pool

		envelope = {
			'type': 'message',
			'senderPool': self.pool,
			'senderName': self.name,
			'senderUuid': self.uuid,
			'targetPool': targetPool,
			'targetName': target,
			'message': message,
			'timestamp': time.time()
		}

		# Serialize and send via PUB socket with pool as topic
		msgBytes = json.dumps(envelope).encode('utf-8')
		self._pubSocket.send_multipart([targetPool.encode('utf-8'), msgBytes])

		log.debug(f"Message sent from {self.pool}/{self.name} to pool {targetPool}")

	def onMessage(self, callback: T.Callable[[T.Any, dict], None]):
		"""Register callback for receiving messages.

		Args:
			callback: Function that takes (message, metadata) where metadata
					  contains 'pool', 'name', 'uuid' of sender
		"""
		self._messageCallback = callback

	def onRequest(self, callback: T.Callable[[T.Any, dict], T.Any]):
		"""Register callback for handling requests.

		Args:
			callback: Function that takes (request, metadata) and returns response
		"""
		self._requestCallback = callback

	def _isAlive(self, instanceUuid: str, currentTime: float = None) -> bool:
		"""Check if an instance is still alive based on heartbeat."""
		if currentTime is None:
			currentTime = time.time()

		with Discoverable._heartbeatsLock:
			if instanceUuid not in Discoverable._heartbeats:
				return False
			lastHeartbeat = Discoverable._heartbeats[instanceUuid]
			return (currentTime - lastHeartbeat) < self.HEARTBEAT_TIMEOUT

	def _connectToPool(self):
		"""Connect SUB socket to all other instances in the same pool."""
		with Discoverable._registryLock:
			if self.pool in Discoverable._registry:
				for name, instance in Discoverable._registry[self.pool].items():
					if instance != self and instance._pubSocket:
						# Get the last bound endpoint
						endpoint = instance._pubSocket.getsockopt_string(zmq.LAST_ENDPOINT)
						self._subSocket.connect(endpoint)
						log.debug(f"Connected to {self.pool}/{name} at {endpoint}")

	def _listenLoop(self):
		"""Background thread for listening to messages and requests."""
		poller = zmq.Poller()
		poller.register(self._subSocket, zmq.POLLIN)
		poller.register(self._repSocket, zmq.POLLIN)

		while not self._stopEvent.is_set():
			try:
				socks = dict(poller.poll(timeout=100))

				# Handle SUB messages (broadcasts)
				if self._subSocket in socks:
					topic, msgBytes = self._subSocket.recv_multipart()
					topicStr = topic.decode('utf-8')
					envelope = json.loads(msgBytes.decode('utf-8'))

					# Handle different message types
					msgType = envelope.get('type', 'message')

					if msgType == 'heartbeat':
						self._handleHeartbeat(envelope)
					elif msgType == 'goodbye':
						self._handleGoodbye(envelope)
					elif msgType == 'message':
						# Check if message is for us (or broadcast to pool)
						targetName = envelope.get('targetName')
						if targetName is None or targetName == self.name:
							# Don't process our own messages
							if envelope['senderUuid'] != self.uuid:
								self._handleMessage(envelope)

				# Handle REP requests (direct)
				if self._repSocket in socks:
					msgBytes = self._repSocket.recv()
					envelope = json.loads(msgBytes.decode('utf-8'))
					response = self._handleRequest(envelope)
					responseBytes = json.dumps(response).encode('utf-8')
					self._repSocket.send(responseBytes)

			except zmq.ZMQError as e:
				if not self._stopEvent.is_set():
					log.error(f"ZMQ error in listen loop: {e}")
			except Exception as e:
				log.error(f"Error in listen loop: {e}")

	def _handleMessage(self, envelope: dict):
		"""Handle received message."""
		if self._messageCallback:
			metadata = {
				'pool': envelope['senderPool'],
				'name': envelope['senderName'],
				'uuid': envelope['senderUuid'],
				'timestamp': envelope['timestamp']
			}
			try:
				self._messageCallback(envelope['message'], metadata)
			except Exception as e:
				log.error(f"Error in message callback: {e}")

	def _handleRequest(self, envelope: dict) -> dict:
		"""Handle received request and return response."""
		if self._requestCallback:
			metadata = {
				'pool': envelope['senderPool'],
				'name': envelope['senderName'],
				'uuid': envelope['senderUuid'],
				'timestamp': envelope['timestamp']
			}
			try:
				response = self._requestCallback(envelope['message'], metadata)
				return {
					'success': True,
					'response': response
				}
			except Exception as e:
				log.error(f"Error in request callback: {e}")
				return {
					'success': False,
					'error': str(e)
				}
		else:
			return {
				'success': False,
				'error': 'No request handler registered'
			}

	def _handleHeartbeat(self, envelope: dict):
		"""Handle received heartbeat from another instance."""
		senderUuid = envelope.get('senderUuid')
		if senderUuid:
			with Discoverable._heartbeatsLock:
				Discoverable._heartbeats[senderUuid] = time.time()
			log.debug(f"Heartbeat received from {envelope.get('senderName')}")

	def _handleGoodbye(self, envelope: dict):
		"""Handle goodbye message from instance that's shutting down."""
		senderUuid = envelope.get('senderUuid')
		senderName = envelope.get('senderName')
		senderPool = envelope.get('senderPool')

		log.info(f"Goodbye received from {senderPool}/{senderName}")

		# Remove from heartbeats immediately
		with Discoverable._heartbeatsLock:
			if senderUuid in Discoverable._heartbeats:
				del Discoverable._heartbeats[senderUuid]

		# Remove from registry
		with Discoverable._registryLock:
			if senderPool in Discoverable._registry:
				if senderName in Discoverable._registry[senderPool]:
					del Discoverable._registry[senderPool][senderName]
					if not Discoverable._registry[senderPool]:
						del Discoverable._registry[senderPool]

	def _heartbeatLoop(self):
		"""Background thread for sending heartbeat messages."""
		while not self._stopEvent.is_set():
			try:
				# Update our own heartbeat
				with Discoverable._heartbeatsLock:
					Discoverable._heartbeats[self.uuid] = time.time()

				# Broadcast heartbeat
				envelope = {
					'type': 'heartbeat',
					'senderPool': self.pool,
					'senderName': self.name,
					'senderUuid': self.uuid,
					'timestamp': time.time()
				}

				msgBytes = json.dumps(envelope).encode('utf-8')
				self._pubSocket.send_multipart([b"__heartbeat__", msgBytes])

				log.debug(f"Heartbeat sent from {self.pool}/{self.name}")

			except Exception as e:
				if not self._stopEvent.is_set():
					log.error(f"Error in heartbeat loop: {e}")

			# Wait for next heartbeat interval
			self._stopEvent.wait(self.HEARTBEAT_INTERVAL)

	def _cleanupLoop(self):
		"""Background thread for cleaning up dead instances."""
		while not self._stopEvent.is_set():
			try:
				currentTime = time.time()
				deadInstances = []

				# Find dead instances
				with Discoverable._heartbeatsLock:
					for instanceUuid, lastHeartbeat in list(Discoverable._heartbeats.items()):
						if (currentTime - lastHeartbeat) > self.HEARTBEAT_TIMEOUT:
							deadInstances.append(instanceUuid)

				# Remove dead instances from registry
				if deadInstances:
					with Discoverable._registryLock:
						for poolName, poolDict in list(Discoverable._registry.items()):
							for name, instance in list(poolDict.items()):
								if instance.uuid in deadInstances:
									log.warning(f"Removing dead instance: {poolName}/{name}")
									del poolDict[name]
							if not poolDict:
								del Discoverable._registry[poolName]

					# Remove from heartbeats
					with Discoverable._heartbeatsLock:
						for instanceUuid in deadInstances:
							if instanceUuid in Discoverable._heartbeats:
								del Discoverable._heartbeats[instanceUuid]

			except Exception as e:
				if not self._stopEvent.is_set():
					log.error(f"Error in cleanup loop: {e}")

			# Check for dead instances every 2 seconds
			self._stopEvent.wait(2.0)

	def _sendGoodbye(self):
		"""Send goodbye message before shutting down."""
		try:
			envelope = {
				'type': 'goodbye',
				'senderPool': self.pool,
				'senderName': self.name,
				'senderUuid': self.uuid,
				'timestamp': time.time()
			}

			msgBytes = json.dumps(envelope).encode('utf-8')
			self._pubSocket.send_multipart([b"__heartbeat__", msgBytes])

			# Give the message time to be sent
			time.sleep(0.1)

			log.debug(f"Goodbye sent from {self.pool}/{self.name}")
		except Exception as e:
			log.error(f"Error sending goodbye: {e}")

	def __repr__(self):
		return f"<Discoverable {self.pool}/{self.name} active={self._active}>"
