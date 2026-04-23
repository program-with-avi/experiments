from ursina import *
import math

app = Ursina()

ground = Entity(model='plane', texture='grass',texture_scale=(2,2), scale=(10000,1,10000), collider='box')

light = DirectionalLight()
light.direction = (1, -1, 1)
scene.brightness = 0.001


Entity(model='cube', color=color.dark_gray, scale=(50, 0.1, 1000), position=(0, 0.05,10))
car = Entity(model='Koenigsegg6.obj', scale=(2, 2, 2), position=(0, 1, 0))

velocity = Vec3(0,0,0)
acceleration = 0.1
friction = 0.85
steering = 60
angle = 0
max_speed = 400

speed_text = Text(text='0 KPH', position=(-0.85, 0.45), scale=2, color=color.white)

car.collider = 'box'
def update():
    global angle, velocity
    
    speed_text.text = f"{int(velocity.length())} KPH"
    
    target_z_rotation = 0
    if held_keys['a'] and velocity.length() > 0.1:
        angle -= steering * time.dt
        target_z_rotation = 5
    if held_keys['d'] and velocity.length() > 0.1:
        angle += steering * time.dt
        target_z_rotation = -5
    
    car.rotation_y = angle
    is_moving_inp = False

    if held_keys['w']:
        velocity += car.forward * acceleration * time.dt * 100
        is_moving_inp = True
    elif held_keys['s']:
        reverse_speed = 400 
        velocity -= car.forward * (acceleration*reverse_speed) * time.dt * 100
        is_moving_inp = True
    
    
    current_speed = velocity.length()
    if current_speed > max_speed:
        velocity = velocity.normalized() * max_speed
        current_speed = max_speed
    
    if is_moving_inp and current_speed > 0.1:
        grip = 0.15
        velocity = lerp(velocity, car.forward * current_speed, grip)
    
    car.rotation_z = lerp(car.rotation_z, target_z_rotation, time.dt * 5)
    
    # Apply friction only when the kart is moving
    if velocity.length() > 0.1:
        actual_friction = math.pow(friction, time.dt * 30)
        velocity *= actual_friction
    
    new_position = car.position + velocity * time.dt
    
    if distance(new_position, car.position) > 0:
        car.position = new_position
    
    target_cam_pos = car.position + (car.back * 70) + (0, 40, 0)
    camera.position = lerp(camera.position, target_cam_pos, time.dt * 3)
    camera.look_at(car)
    camera.rotation_z = 0
app.run()