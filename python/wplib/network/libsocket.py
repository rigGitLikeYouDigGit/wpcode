from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

import sys, os, socket, contextlib, struct

def getFreeLocalPortIndex():
	s = socket.socket()
	s.bind(("localhost", 0))
	portId = s.getsockname()[1]
	s.close()
	return portId

def portIsOpen(host, port, timeout=0.1):
	with contextlib.closing(
			socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock: #type:socket.socket
		sock.settimeout(timeout)
		return sock.connect_ex((host, port))

# from Adam Rosenfield on SO
LEN_PREFIX_CHAR = '>I'
def sendMsg(sock, msg):
	# Prefix each message with a 4-byte length (network byte order)
	msg = struct.pack(LEN_PREFIX_CHAR, len(msg)) + msg
	sock.sendall(msg)

def recvMsg(sock):
	# Read message length and unpack it into an integer
	raw_msglen = recvall(sock, 4)
	if not raw_msglen:
		return None
	msglen = struct.unpack(LEN_PREFIX_CHAR, raw_msglen)[0]
	# Read the message data
	return recvall(sock, msglen)

def recvall(sock, n):
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
