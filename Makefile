all: proto

proto: lib/proto/trace_pb2.py

lib/proto/trace_pb2.py: lib/proto/trace.proto
	protoc $< --python_out=.

.phony: proto