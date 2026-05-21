import * as THREE from 'three';
import { Controls } from '../ui/Controls';
import { Weapon } from './Weapon';

export class Player {
  private camera: THREE.PerspectiveCamera;
  private controls: Controls;
  private weapon: Weapon;
  private moveSpeed = 0.15;
  
  private lookSensitivityX = 0.0012; 
  private lookSensitivityY = 0.0008; 
  
  private rotationY = 0;
  private rotationX = 0;
  private isMoving = false;
  private velocityY = 0;
  private isGrounded = true;
  private gravity = -0.01;
  private jumpForce = 0.2;
  private isAiming = false;

  constructor(camera: THREE.PerspectiveCamera, controls: Controls) {
    this.camera = camera;
    this.controls = controls;
    this.weapon = new Weapon(this.camera);
  }

  public update() {
    this.handleMovement();
    this.handleRotation();
    this.applyGravity();
    this.handleFOV();
    this.weapon.update(this.isMoving, Date.now() * 0.001);
  }

  public setAim(isAiming: boolean) {
    this.isAiming = isAiming;
    this.weapon.setAim(isAiming);
  }

  private handleFOV() {
    const targetFOV = this.isAiming ? 45 : 75;
    if (this.camera.fov !== targetFOV) {
      this.camera.fov = THREE.MathUtils.lerp(this.camera.fov, targetFOV, 0.2);
      this.camera.updateProjectionMatrix();
    }
  }

  public jump() {
    if (this.isGrounded) {
      this.velocityY = this.jumpForce;
      this.isGrounded = false;
    }
  }

  private applyGravity() {
    this.camera.position.y += this.velocityY;
    if (this.camera.position.y > 1.7) {
      this.velocityY += this.gravity;
      this.isGrounded = false;
    } else {
      this.camera.position.y = 1.7;
      this.velocityY = 0;
      this.isGrounded = true;
    }
  }

  public shoot() {
    this.weapon.shoot();
    this.shakeCamera();
  }

  private shakeCamera() {
    const intensity = this.isAiming ? 0.01 : 0.03;
    const shake = () => {
      this.camera.position.x += (Math.random() - 0.5) * intensity;
      this.camera.position.y += (Math.random() - 0.5) * intensity;
    };

    const interval = setInterval(shake, 16);
    setTimeout(() => {
      clearInterval(interval);
    }, 50);
  }

  private handleMovement() {
    let moveX = this.controls.moveData.x;
    let moveY = this.controls.moveData.y;

    if (this.controls.keys['KeyW']) moveY = 1;
    if (this.controls.keys['KeyS']) moveY = -1;
    if (this.controls.keys['KeyA']) moveX = -1;
    if (this.controls.keys['KeyD']) moveX = 1;

    // Slower movement when aiming
    const currentSpeed = this.isAiming ? this.moveSpeed * 0.5 : this.moveSpeed;

    if (Math.abs(moveX) > 0.05 || Math.abs(moveY) > 0.05) {
      this.isMoving = true;
      
      const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion);
      forward.y = 0;
      forward.normalize();

      const right = new THREE.Vector3(1, 0, 0).applyQuaternion(this.camera.quaternion);
      right.y = 0;
      right.normalize();

      this.camera.position.addScaledVector(forward, moveY * currentSpeed);
      this.camera.position.addScaledVector(right, moveX * currentSpeed);
    } else {
      this.isMoving = false;
    }
  }

  private handleRotation() {
    const lookX = this.controls.lookData.x;
    const lookY = this.controls.lookData.y;

    // Slower look sensitivity when aiming
    const sensitivityMultiplier = this.isAiming ? 0.4 : 1.0;

    if (Math.abs(lookX) > 0.001) {
      this.rotationY -= lookX * 40 * this.lookSensitivityX * sensitivityMultiplier;
    }

    if (Math.abs(lookY) > 0.001) {
      this.rotationX += lookY * 40 * this.lookSensitivityY * sensitivityMultiplier;
      this.rotationX = Math.max(-Math.PI / 2.2, Math.min(Math.PI / 2.2, this.rotationX));
    }

    this.camera.quaternion.setFromEuler(new THREE.Euler(this.rotationX, this.rotationY, 0, 'YXZ'));
  }

  public upgradeSpeed() {
    this.moveSpeed *= 1.5;
  }
}
