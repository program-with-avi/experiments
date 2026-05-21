import * as THREE from 'three';

export class Weapon {
  public mesh: THREE.Group;
  private gunBody: THREE.Mesh;
  private muzzleFlash: THREE.PointLight;
  private originalPosition = new THREE.Vector3(0.3, -0.3, -0.5);
  private bobAmount = 0.02;
  private bobSpeed = 10;
  private recoilAmount = 0.1;
  private isShooting = false;

  constructor(camera: THREE.Camera) {
    this.mesh = new THREE.Group();
    
    // Low-poly Gun Body
    const bodyGeo = new THREE.BoxGeometry(0.1, 0.15, 0.4);
    const bodyMat = new THREE.MeshStandardMaterial({ color: 0x222222 });
    this.gunBody = new THREE.Mesh(bodyGeo, bodyMat);
    this.mesh.add(this.gunBody);

    // Barrel
    const barrelGeo = new THREE.BoxGeometry(0.05, 0.05, 0.3);
    const barrelMat = new THREE.MeshStandardMaterial({ color: 0x111111 });
    const barrel = new THREE.Mesh(barrelGeo, barrelMat);
    barrel.position.z = -0.3;
    this.mesh.add(barrel);

    // Muzzle Flash Light
    this.muzzleFlash = new THREE.PointLight(0xffaa00, 0, 2);
    this.muzzleFlash.position.z = -0.5;
    this.mesh.add(this.muzzleFlash);

    this.mesh.position.copy(this.originalPosition);
    camera.add(this.mesh);
  }

  public update(isMoving: boolean, time: number) {
    // Bobbing
    if (isMoving) {
      this.mesh.position.y = this.originalPosition.y + Math.sin(time * this.bobSpeed) * this.bobAmount;
      this.mesh.position.x = this.originalPosition.x + Math.cos(time * this.bobSpeed * 0.5) * this.bobAmount;
    } else {
      this.mesh.position.lerp(this.originalPosition, 0.1);
    }
  }

  public shoot() {
    if (this.isShooting) return;
    this.isShooting = true;

    // Recoil
    this.mesh.position.z += this.recoilAmount;
    this.muzzleFlash.intensity = 2;

    setTimeout(() => {
      this.muzzleFlash.intensity = 0;
      this.mesh.position.z = this.originalPosition.z;
      this.isShooting = false;
    }, 50);
  }
}
