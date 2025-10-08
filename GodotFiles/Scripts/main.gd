extends Node

var server := TCPServer.new()
var client : StreamPeerTCP

func _ready():
	server.listen(5005)
	print("Listening on port 5005")

func _process(_delta):
	if not client and server.is_connection_available():
		client = server.take_connection()
		print("Client connected")

	if client and client.get_available_bytes() > 0:
		var msg = client.get_utf8_string(client.get_available_bytes())
		print("Received:", msg)
		apply_boid_rule(msg)

func apply_boid_rule(rule: String):
	match rule:
		"follow":
			print("follow")
			#BoidManager.set_rule(FOLLOW)
		"scatter":
			print("scatter")
			#BoidManager.set_rule(SCATTER)
		"evade":
			print("evade")
			#BoidManager.set_rule(EVADE)
		"seek":
			print("seek")
			#BoidManager.set_rule(SEEK)
