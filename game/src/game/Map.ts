import * as THREE from 'three';

export class Map {
  private scene: THREE.Scene;

  constructor(scene: THREE.Scene) {
    this.scene = scene;
    this.initArena();
  }

  private initArena() {
    // Ground
    const groundGeo = new THREE.PlaneGeometry(100, 100);
    const groundMat = new THREE.MeshStandardMaterial({ color: 0x333333 });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    this.scene.add(ground);

    // Outer Walls
    this.createWall(0, 5, -50, 100, 10, 1); // North
    this.createWall(0, 5, 50, 100, 10, 1);  // South
    this.createWall(-50, 5, 0, 1, 10, 100); // West
    this.createWall(50, 5, 0, 1, 10, 100);  // East

    // Random Crates (Krunker Style)
    for (let i = 0; i < 20; i++) {
      const size = 1 + Math.random() * 2;
      this.createCrate(
        (Math.random() - 0.5) * 80,
        size / 2,
        (Math.random() - 0.5) * 80,
        size
      );
    }

    // Central Pillar
    this.createWall(0, 10, 0, 5, 20, 5);
  }

  private createWall(x: number, y: number, z: number, w: number, h: number, d: number) {
    const geo = new THREE.BoxGeometry(w, h, d);
    const mat = new THREE.MeshStandardMaterial({ color: 0x555555 });
    const wall = new THREE.Mesh(geo, mat);
    wall.position.set(x, y, z);
    wall.castShadow = true;
    wall.receiveShadow = true;
    this.scene.add(wall);
  }

  private createCrate(x: number, y: number, z: number, size: number) {
    const geo = new THREE.BoxGeometry(size, size, size);
    const mat = new THREE.MeshStandardMaterial({ color: 0x8b4513 }); // Brown crate
    const crate = new THREE.Mesh(geo, mat);
    crate.position.set(x, y, z);
    crate.castShadow = true;
    crate.receiveShadow = true;
    this.scene.add(crate);
  }
}
