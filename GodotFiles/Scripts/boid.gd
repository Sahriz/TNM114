extends Node2D

var velocity: Vector2
var acceleration: Vector2

var max_speed = 300.0
var max_force = 100.0
var alive = true

func _ready():
	$Area2D.connect("area_entered", Callable(self, "_on_area_entered"))

func _physics_process(delta):
	if not alive:
		return

	velocity += acceleration * delta
	velocity = velocity.limit_length(max_speed)
	position += velocity * delta
	rotation = velocity.angle()
	acceleration = Vector2.ZERO

func apply_force(force: Vector2):
	acceleration += force.limit_length(max_force)

func _on_area_entered(area: Area2D):
	if area.is_in_group("projectile"):
		alive = false
		queue_free() # remove when hit
