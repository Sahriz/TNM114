extends Node2D
class_name Manager

@export var unit_scene: PackedScene
@export var num_units := 30
@export var radius := 100.0

var units: Array[Node2D] = []
var angles: Array[float] = []
var speeds: Array[float] = []
var ellipse_scales: Array[Vector2] = []

var center := Vector2.ZERO
var target := Vector2.ZERO
var velocity := Vector2.ZERO

func _ready():
	center = position
	target = position

	for i in range(num_units):
		var unit = unit_scene.instantiate()
		add_child(unit)
		units.append(unit)

		angles.append(randf() * TAU)

		# Half rotate clockwise, half counterclockwise
		var dir = 1.0
		if randf() < 0.5:
			dir = -1.0
		speeds.append(dir * (0.5 + randf() * 1.0))

		ellipse_scales.append(Vector2(
			lerp(0.2, 1.2, randf()),
			lerp(0.2, 1.2, randf())
		))

func _process(delta):
	# Smoothly move center toward target using spring-like motion (no slowdown tail)
	var diff = target - center
	var dist = diff.length()

	if dist > 0.0:
		var accel = diff.normalized() * delta * 20000.0  # reduced from 10000 to 1000
		velocity += accel

	# Clamp velocity so we don't overshoot the target
		if velocity.length() > dist / delta:
			velocity = velocity.normalized() * (dist / delta)
		
	velocity *= 0.85  # damp oscillation but keeps energy
	center += velocity * delta

	var t = Time.get_ticks_msec() * 0.001
	var move_speed = velocity.length()
	var orbit_factor = clamp(1.0 - move_speed * 1.5, 0.0, 1.0)

	for i in range(num_units):
		angles[i] += speeds[i] * delta * orbit_factor

		var r = radius + sin(t * 1.5 + i) * 5.0
		var ellipse = ellipse_scales[i]

		var orbit_offset = Vector2(
			cos(angles[i]) * r * ellipse.x,
			sin(angles[i]) * r * ellipse.y
		)

		# Small jitter for life
		var jitter = Vector2(
			sin(t * 0.7 + i * 2.1),
			cos(t * 0.9 + i * 1.7)
		) * 5.0 * orbit_factor

		var desired_pos = center + orbit_offset + jitter
		var current_pos = units[i].position

		# Move units fast enough regardless of how close they are
		var follow_speed = 80.0
		var desired_diff = desired_pos - current_pos
		
		var step = desired_diff * (delta * follow_speed)
		var min_step = 20.0
		var step_length = min(max(step.length(), min_step), desired_diff.length())
		step = desired_diff.normalized() * step_length

		var new_pos = current_pos + step

		# Face movement direction
		var direction = new_pos - current_pos
		if direction.length() > 0.1:
			units[i].rotation = direction.angle()

		units[i].position = new_pos

func _unhandled_input(event):
	if event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_LEFT:
		set_target(get_global_mouse_position())

func set_target(pos: Vector2):
	target = pos
	
func receive_data_from_listener(data: String) -> void:
	if data == "Scatter":
		set_target(Vector2(-100,-100))
	if data == "Cluster":
		set_target(Vector2(578,324))
	if data == "Target":
		set_target(Vector2(1200,700))
	if data == "Left":
		set_target(Vector2(100,324))
	if data == "Upleft":
		set_target(Vector2(100,100))
	if data == "Up":
		set_target(Vector2(576,100))
	if data == "UpRight":
		set_target(Vector2(1052,100))
	if data == "Right":
		set_target(Vector2(1052,324))
	if data == "Downright":
		set_target(Vector2(1052,548))
	if data == "Down":
		set_target(Vector2(576,548))
	if data == "Downleft":
		set_target(Vector2(100,558))
