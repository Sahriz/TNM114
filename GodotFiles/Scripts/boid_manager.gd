extends Node2D

@export var boid_scene: PackedScene
@export var num_boids: int = 100
@export var spawn_radius: float = 100.0

@onready var boid_container = $"../BoidContainer"

var boids = []

# Behavior weights
var separation_weight = 500.0
var alignment_weight = 750.0
var cohesion_weight = 750.0
var seek_weight = 50.0

var target_position: Vector2 = Vector2.ZERO


func _ready():
	spawn_boids()

func spawn_boids():
	for i in range(num_boids):
		var b = boid_scene.instantiate()
		b.position = Vector2(randf_range(-spawn_radius, spawn_radius), randf_range(-spawn_radius, spawn_radius))
		b.velocity = Vector2(randf_range(-50, 50), randf_range(-50, 50))
		boid_container.add_child(b)
		boids.append(b)

func _physics_process(delta):
	handle_input()
	update_boids(delta)

func handle_input():
	# Example: player can steer flock behavior
	if Input.is_action_pressed("ui_up"):
		alignment_weight += 0.01
	if Input.is_action_pressed("ui_down"):
		alignment_weight -= 0.01

func update_boids(delta):
	for b in boids:
		var neighbors = get_neighbors(b, 100.0)
		var sep = separation(b, neighbors)
		var ali = alignment(b, neighbors)
		var coh = cohesion(b, neighbors)
		var see = seek(b ,target_position)

		var steer = sep * separation_weight + ali * alignment_weight + coh * cohesion_weight
		b.apply_force(steer)

func get_neighbors(boid, radius):
	var neighbors = []
	for other in boids:
		if other == boid:
			continue
		if boid.position.distance_to(other.position) < radius:
			neighbors.append(other)
	return neighbors

# --- Boid Rules ---
func separation(boid, neighbors):
	var force = Vector2.ZERO
	for n in neighbors:
		var diff = boid.position - n.position
		var dist = diff.length()
		if dist > 0:
			force += diff.normalized() / dist
	return force.normalized()

func alignment(boid, neighbors):
	if neighbors.is_empty():
		return Vector2.ZERO
	var avg_vel = Vector2.ZERO
	for n in neighbors:
		avg_vel += n.velocity
	avg_vel /= neighbors.size()
	return (avg_vel.normalized() - boid.velocity.normalized()).normalized()

func cohesion(boid, neighbors):
	if neighbors.is_empty():
		return Vector2.ZERO
	var center = Vector2.ZERO
	for n in neighbors:
		center += n.position
	center /= neighbors.size()
	return (center - boid.position).normalized()

func seek(boid, target) -> Vector2:
	var desired = (target - boid.position)
	var distance = desired.length()
	if distance == 0:
		return Vector2.ZERO

	#desired = desired.normalized()
	var steer = desired - boid.velocity
	return steer.limit_length(1.0)  # Prevent oversteering


func _input(event):
	if event is InputEventMouseButton and event.pressed:
		target_position = event.position
		print("clicked")
