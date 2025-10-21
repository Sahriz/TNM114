extends Node

var server := TCPServer.new()
var client : StreamPeerTCP

var manager : Manager = null

func _ready():
	server.listen(5005)
	print("Listening on port 5005")
	manager = get_parent().get_node("Manager") as Manager 

func _process(_delta):
	if not client and server.is_connection_available():
		client = server.take_connection()
		print("Client connected")

	if client and client.get_available_bytes() > 0:
		var msg = client.get_utf8_string(client.get_available_bytes())
		print("Received:", msg)
		if msg != "":
			send_to_manager(msg)

func send_to_manager(data: String) -> void:
	if manager:
		manager.receive_data_from_listener(data)
