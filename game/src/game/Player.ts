import * as THREE from 'three';
import { Controls } from '../ui/Controls';
import { Weapon } from './Weapon';

export class Player {
  private camera: THREE.PerspectiveCamera;
  private controls: Controls;
  private weapon: Weapon;
  private moveSpeed = 0.1;
  private lookSensitivity = 0.05;
  private rotationY = 0;
  private rotationX = 0;
  private isMoving = false;

  constructor(camera: THREE.PerspectiveCamera, controls: Controls) {
    this.camera = camera;
    this.controls = controls;
    this.weapon = new Weapon(this.camera);
  }

  public update() {
    this.handleMovement();
    this.handleRotation();
    this.weapon.update(this.isMoving, Date.now() * 0.001);
  }

  public shoot() {
    this.weapon.shoot();
    this.shakeCamera();
  }

  private shakeCamera() {
    const originalPos = this.camera.position.clone();
    const intensity = 0.05;
    
    const shake = () => {
      this.camera.position.x += (Math.random() - 0.5) * intensity;
      this.camera.position.y += (Math.random() - 0.5) * intensity;
    };

    const interval = setInterval(shake, 16);
    setTimeout(() => {
      clearInterval(interval);
      // Reset position relative to camera parent or absolute? 
      // Camera is moved by handleMovement, so we just let it be.
    }, 50);
  }

  private handleMovement() {
    const moveX = this.controls.moveData.x;
    const moveY = this.controls.moveData.y;

    if (moveX !== 0 || moveY !== 0) {
      this.isMoving = true;
      const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion);
      forward.y = 0;
      forward.normalize();

      const right = new THREE.Vector3(1, 0, 0).applyQuaternion(this.camera.quaternion);
      right.y = 0;
      right.normalize();

      this.camera.position.addScaledVector(forward, moveY * this.moveSpeed);
      this.camera.position.addScaledVector(right, moveX * this.moveSpeed);
    } else {
      this.isMoving = false;
    }
  }


  private handleRotation() {
    const lookX = this.controls.lookData.x;
    const lookY = this.controls.lookData.y;

    if (lookX !== 0 || lookY !== 0) {
      this.rotationY -= lookX * this.lookSensitivity;
      this.rotationX += lookY * this.lookSensitivity;

      // Clamp vertical rotation
      this.rotationX = Math.max(-Math.PI / 2.2, Math.min(Math.PI / 2.2, this.rotationX));

      this.camera.quaternion.setFromEuler(new THREE.Euler(this.rotationX, this.rotationY, 0, 'YXZ'));
    }
  }
}
