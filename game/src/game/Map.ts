import * as THREE from 'three';

export class Map {
  private scene: THREE.Scene;

  constructor(scene: THREE.Scene) {
    this.scene = scene;
    this.initArena();
  }

  private initArena() {
    // Ground - Grid Texture
    const groundGeo = new THREE.PlaneGeometry(200, 200);
    const groundMat = new THREE.MeshStandardMaterial({ 
      color: 0x111111,
      roughness: 0.8,
      metalness: 0.2
    });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    this.scene.add(ground);

    // Floor Grid
    const grid = new THREE.GridHelper(200, 40, 0x00ffff, 0x00ffff);
    (grid.material as THREE.Material).opacity = 0.2;
    (grid.material as THREE.Material).transparent = true;
    grid.position.y = 0.05;
    this.scene.add(grid);

    // Cyber Skybox (using fog and a large sphere)
    this.scene.background = new THREE.Color(0x00050a);
    this.scene.fog = new THREE.FogExp2(0x00050a, 0.015);

    // Neon Walls
    this.createNeonWall(0, 5, -100, 200, 20, 1, 0x00ffff); // North
    this.createNeonWall(0, 5, 100, 200, 20, 1, 0x00ffff);  // South
    this.createNeonWall(-100, 5, 0, 1, 20, 200, 0xff00ff); // West
    this.createNeonWall(100, 5, 0, 1, 20, 200, 0xff00ff);  // East

    // Structured Obstacles (Arena Style)
    for (let i = 0; i < 30; i++) {
      const h = 2 + Math.random() * 6;
      const w = 2 + Math.random() * 4;
      const x = (Math.random() - 0.5) * 160;
      const z = (Math.random() - 0.5) * 160;
      
      // Avoid spawning near center
      if (Math.abs(x) < 10 && Math.abs(z) < 10) continue;

      this.createCyberCrate(x, h/2, z, w, h, w);
    }

    // Central Neon Tower
    this.createNeonWall(0, 15, 0, 8, 30, 8, 0xffff00);
  }

  private createNeonWall(x: number, y: number, z: number, w: number, h: number, d: number, color: number) {
    const geo = new THREE.BoxGeometry(w, h, d);
    const mat = new THREE.MeshStandardMaterial({ 
      color: 0x222222,
      emissive: color,
      emissiveIntensity: 0.5
    });
    const wall = new THREE.Mesh(geo, mat);
    wall.position.set(x, y, z);
    wall.castShadow = true;
    wall.receiveShadow = true;
    this.scene.add(wall);

    // Glow effect (simplified)
    const glowGeo = new THREE.BoxGeometry(w + 0.2, h + 0.2, d + 0.2);
    const glowMat = new THREE.MeshBasicMaterial({ color: color, transparent: true, opacity: 0.1 });
    const glow = new THREE.Mesh(glowGeo, glowMat);
    glow.position.set(x, y, z);
    this.scene.add(glow);
  }

  private createCyberCrate(x: number, y: number, z: number, w: number, h: number, d: number) {
    const geo = new THREE.BoxGeometry(w, h, d);
    const mat = new THREE.MeshStandardMaterial({ 
      color: 0x333333,
      metalness: 0.8,
      roughness: 0.2,
      emissive: 0x444444
    });
    const crate = new THREE.Mesh(geo, mat);
    crate.position.set(x, y, z);
    crate.castShadow = true;
    crate.receiveShadow = true;
    this.scene.add(crate);

    // Edge highlight
    const wireframe = new THREE.WireframeGeometry(geo);
    const line = new THREE.LineSegments(wireframe);
    (line.material as THREE.Material).color = new THREE.Color(0x00ffff);
    (line.material as THREE.Material).transparent = true;
    (line.material as THREE.Material).opacity = 0.3;
    line.position.set(x, y, z);
    this.scene.add(line);
  }
}
