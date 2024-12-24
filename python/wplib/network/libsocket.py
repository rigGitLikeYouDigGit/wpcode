from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

import sys, os, socket, contextlib, struct
import orjson
from collections import namedtuple
from typing import NamedTuple

def getFreeLocalPortIndex():
	s = socket.socket()
	s.bind(("localhost", 0))
	portId = s.getsockname()[1]
	s.close()
	return portId

# def portIsOpen(host, port, timeout=0.1):
# 	with contextlib.closing(
# 			socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock: #type:socket.socket
# 		sock.settimeout(timeout)
# 		return sock.connect_ex((host, port))


class SocketStatus:
	CLOSE_WAIT = "CLOSE_WAIT"
	ESTABLISHED = "ESTABLISHED"
	LISTENING = "LISTENING"
	TIME_WAIT = "TIME_WAIT"

class SocketRecord(NamedTuple):
	type : str
	localAddress : str
	foreignAddress : str
	state : (T.LiteralString["CLOSE_WAIT"],
	         T.LiteralString["ESTABLISHED"],
	         T.LiteralString["LISTENING"],
	         T.LiteralString["TIME_WAIT"])

def getOpenSockets()->list[SocketRecord]:
	"""return list of
	"""
	cmd = "netstat -an -p TCP"
	result = os.popen(cmd).read()

	statuses = ("CLOSE_WAIT", "ESTABLISHED", "LISTENING", "TIME_WAIT" )
	resultStatuses = []
	for i in result.split("\n"):
		tokens = tuple(filter(None, i.split()))
		if not tokens:
			continue
		if tokens[-1] in statuses:
			resultStatuses.append(SocketRecord(*tokens))
	return resultStatuses

def localPortSocketMap()->dict[int, SocketRecord]:
	return {int(i.localAddress.split(":")[-1]) : i for i in getOpenSockets()}


"""functions to prefix length of each message packet, so we can
ensure we always get a whole message no matter the length.

also adding the orjson serialisation to bytes here - 
gives these functions too much responsibility, BUT it also means 
we can work with normal python objects right up until we use these
functions to dump them into the network.
no ambiguity about when to convert to bytes - caller code doesn't do it at all
"""
# from Adam Rosenfield on SO
LEN_PREFIX_CHAR = '>I'
def sendMsg(sock:socket.socket, msg, serialise=True):
	# Prefix each message with a 4-byte length (network byte order)
	if serialise:
		msg = orjson.dumps(msg)
	msg = struct.pack(LEN_PREFIX_CHAR, len(msg)) + msg
	sock.sendall(msg)

def recvMsg(sock:socket.socket, deserialise=True):
	# Read message length and unpack it into an integer
	raw_msglen = recvall(sock, 4)
	if not raw_msglen:
		return None
	msglen = struct.unpack(LEN_PREFIX_CHAR, raw_msglen)[0]
	# Read the message data
	result = recvall(sock, msglen)
	if deserialise:
		result = orjson.loads(result)
	return result

def recvall(sock:socket.socket, n):
	# Helper function to recv n bytes or return None if EOF is hit
	data = bytearray()
	while len(data) < n:
		packet = sock.recv(n - len(data))
		if not packet:
			return None
		data.extend(packet)
	return data


if __name__ == '__main__':

	print(getFreeLocalPortIndex())
