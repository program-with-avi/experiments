import * as THREE from 'three';

export class Weapon {
  public mesh: THREE.Group;
  private gunBody: THREE.Mesh;
  private muzzleFlash: THREE.PointLight;
  private originalPosition = new THREE.Vector3(0.3, -0.3, -0.5);
  
  // RECALIBRATED ADS POSITION FOR PERFECT CENTER
  private adsPosition = new THREE.Vector3(0, -0.12, -0.4); 
  private currentTargetPosition = new THREE.Vector3();
  
  private bobAmount = 0.005; 
  private bobSpeed = 8;
  private recoilAmount = 0.05;
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
    const barrelGeo = new THREE.BoxGeometry(0.04, 0.04, 0.3);
    const barrelMat = new THREE.MeshStandardMaterial({ color: 0x111111 });
    const barrel = new THREE.Mesh(barrelGeo, barrelMat);
    barrel.position.z = -0.3;
    this.mesh.add(barrel);

    // --- NEW PRECISION SIGHT SYSTEM ---
    const sightGroup = new THREE.Group();
    sightGroup.position.set(0, 0.12, -0.1); // Position on top of gun

    // Circle Frame (Torus)
    const frameGeo = new THREE.TorusGeometry(0.04, 0.004, 8, 32);
    const frameMat = new THREE.MeshStandardMaterial({ color: 0x111111 });
    const frame = new THREE.Mesh(frameGeo, frameMat);
    // Rotate torus to face the camera
    frame.rotation.y = Math.PI;
    sightGroup.add(frame);

    // Red Dot (Small emissive sphere)
    const dotGeo = new THREE.SphereGeometry(0.004, 8, 8);
    const dotMat = new THREE.MeshBasicMaterial({ color: 0xff0000 });
    const dot = new THREE.Mesh(dotGeo, dotMat);
    dot.position.set(0, 0, 0); // Center of the frame
    sightGroup.add(dot);

    this.mesh.add(sightGroup);

    // Muzzle Flash
    this.muzzleFlash = new THREE.PointLight(0xffaa00, 0, 2);
    this.muzzleFlash.position.set(0, 0.05, -0.5);
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
    this.mesh.position.lerp(this.currentTargetPosition, 0.3);

    const currentBobAmount = this.isAiming ? this.bobAmount * 0.1 : this.bobAmount;
    
    if (isMoving) {
      this.mesh.position.y += Math.sin(time * this.bobSpeed) * currentBobAmount;
      this.mesh.position.x += Math.cos(time * this.bobSpeed * 0.5) * (currentBobAmount * 0.5);
    }
  }

  public shoot() {
    if (this.isShooting) return;
    this.isShooting = true;

    const currentRecoil = this.isAiming ? this.recoilAmount * 0.3 : this.recoilAmount;
    this.mesh.position.z += currentRecoil;
    this.muzzleFlash.intensity = 2;

    setTimeout(() => {
      this.muzzleFlash.intensity = 0;
      this.mesh.position.z = this.currentTargetPosition.z;
      this.isShooting = false;
    }, 50);
  }
}
