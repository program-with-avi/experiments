import * as THREE from 'three';

export class Weapon {
  public mesh: THREE.Group;
  private gunBody: THREE.Mesh;
  private muzzleFlash: THREE.PointLight;
  private originalPosition = new THREE.Vector3(0.3, -0.3, -0.5);
  private adsPosition = new THREE.Vector3(0, -0.15, -0.4);
  private currentTargetPosition = new THREE.Vector3();
  
  private bobAmount = 0.02;
  private bobSpeed = 10;
  private recoilAmount = 0.1;
  private isShooting = false;
  private isAiming = false;

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

    // Sight (Front Post)
    const sightGeo = new THREE.BoxGeometry(0.01, 0.04, 0.01);
    const sightMat = new THREE.MeshStandardMaterial({ color: 0xff0000 });
    const sight = new THREE.Mesh(sightGeo, sightMat);
    sight.position.set(0, 0.08, -0.4);
    this.mesh.add(sight);

    // Muzzle Flash Light
    this.muzzleFlash = new THREE.PointLight(0xffaa00, 0, 2);
    this.muzzleFlash.position.z = -0.5;
    this.mesh.add(this.muzzleFlash);

    this.currentTargetPosition.copy(this.originalPosition);
    this.mesh.position.copy(this.currentTargetPosition);
    camera.add(this.mesh);
  }

  public setAim(isAiming: boolean) {
    this.isAiming = isAiming;
    this.currentTargetPosition.copy(isAiming ? this.adsPosition : this.originalPosition);
  }

  public update(isMoving: boolean, time: number) {
    // Smooth transition to target position (ADS or Hip)
    this.mesh.position.lerp(this.currentTargetPosition, 0.2);

    // Reduced bobbing when aiming
    const currentBobAmount = this.isAiming ? this.bobAmount * 0.2 : this.bobAmount;
    
    if (isMoving) {
      this.mesh.position.y += Math.sin(time * this.bobSpeed) * currentBobAmount;
      this.mesh.position.x += Math.cos(time * this.bobSpeed * 0.5) * currentBobAmount;
    }
  }

  public shoot() {
    if (this.isShooting) return;
    this.isShooting = true;

    // Recoil (less recoil in ADS)
    const currentRecoil = this.isAiming ? this.recoilAmount * 0.5 : this.recoilAmount;
    this.mesh.position.z += currentRecoil;
    this.muzzleFlash.intensity = 2;

    setTimeout(() => {
      this.muzzleFlash.intensity = 0;
      this.mesh.position.z = this.currentTargetPosition.z;
      this.isShooting = false;
    }, 50);
  }
}
