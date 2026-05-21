import * as THREE from 'three';
import { io, Socket } from 'socket.io-client';
import { Controls } from './ui/Controls';
import { Player } from './game/Player';
import { Map } from './game/Map';

class Game {
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private controls: Controls;
  private player: Player;
  private map: Map;
  private targets: THREE.Mesh[] = [];
  private raycaster: THREE.Raycaster;
  private huntCoins: number = 0;
  
  private socket: Socket | null = null;
  private otherPlayers: { [id: string]: THREE.Group } = {};

  constructor() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x00050a); // Matching map background
    this.scene.fog = new THREE.FogExp2(0x00050a, 0.015);

    this.camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    this.camera.position.set(0, 1.7, 0);
    this.scene.add(this.camera);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    document.getElementById('app')?.appendChild(this.renderer.domElement);

    // Initialize map BEFORE controls and player to ensure scene is ready
    this.map = new Map(this.scene);
    this.controls = new Controls();
    this.player = new Player(this.camera, this.controls);
    this.raycaster = new THREE.Raycaster();

    this.controls.onFire = () => this.shoot();
    this.controls.onJump = () => this.player.jump();
    this.controls.onAim = (isAiming) => this.player.setAim(isAiming);
    this.controls.onPurchase = (item) => this.handlePurchase(item);

    this.initLights();
    this.initPracticeTargets();
    this.initMultiplayerUI();

    window.addEventListener('resize', () => this.onWindowResize());
    
    // Start the animation loop
    this.animate();
  }

  private initMultiplayerUI() {
    const btn = document.createElement('button');
    btn.innerText = 'GO ONLINE';
    btn.style.position = 'absolute';
    btn.style.top = '70px';
    btn.style.left = '20px';
    btn.style.padding = '10px 20px';
    btn.style.background = 'green';
    btn.style.color = 'white';
    btn.style.border = '2px solid white';
    btn.style.fontWeight = 'bold';
    btn.style.pointerEvents = 'auto';
    btn.style.zIndex = '100';
    document.body.appendChild(btn);

    btn.onclick = () => {
      this.connectToServer();
      btn.remove();
    };
  }

  private connectToServer() {
    this.socket = io('http://localhost:3000');

    this.socket.on('currentPlayers', (players: any) => {
      Object.keys(players).forEach((id) => {
        if (id !== this.socket?.id) {
          this.addOtherPlayer(players[id]);
        }
      });
    });

    this.socket.on('newPlayer', (playerInfo: any) => {
      this.addOtherPlayer(playerInfo);
    });

    this.socket.on('playerMoved', (playerInfo: any) => {
      if (this.otherPlayers[playerInfo.id]) {
        this.otherPlayers[playerInfo.id].position.copy(playerInfo.position);
        this.otherPlayers[playerInfo.id].rotation.y = playerInfo.rotation.y;
      }
    });

    this.socket.on('playerDisconnected', (id: string) => {
      if (this.otherPlayers[id]) {
        this.scene.remove(this.otherPlayers[id]);
        delete this.otherPlayers[id];
      }
    });
  }

  private addOtherPlayer(playerInfo: any) {
    const group = new THREE.Group();
    
    // Body
    const bodyGeo = new THREE.BoxGeometry(1, 1.5, 0.5);
    const bodyMat = new THREE.MeshStandardMaterial({ color: 0xff0000 });
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    body.position.y = 0.75;
    group.add(body);

    // Head
    const headGeo = new THREE.BoxGeometry(0.5, 0.5, 0.5);
    const headMat = new THREE.MeshStandardMaterial({ color: 0xffcccc });
    const head = new THREE.Mesh(headGeo, headMat);
    head.position.y = 1.75;
    head.userData.isHead = true;
    group.add(head);

    group.position.copy(playerInfo.position);
    this.scene.add(group);
    this.otherPlayers[playerInfo.id] = group;
  }

  private handlePurchase(item: string) {
    if (item === 'speed' && this.huntCoins >= 100) {
      this.huntCoins -= 100;
      this.player.upgradeSpeed();
      this.controls.updateCurrency(this.huntCoins);
      alert('Speed Boost Purchased!');
    } else if (item === 'red_crosshair' && this.huntCoins >= 50) {
      this.huntCoins -= 50;
      this.controls.updateCurrency(this.huntCoins);
      alert('Red Crosshair Purchased!');
    } else {
      alert('Not enough HuntCoins!');
    }
  }

  private shoot() {
    this.player.shoot();
    
    this.raycaster.setFromCamera({ x: 0, y: 0 }, this.camera);
    const intersects = this.raycaster.intersectObjects(this.targets);

    if (intersects.length > 0) {
      const hitTarget = intersects[0].object as THREE.Mesh;
      this.handleHit(hitTarget);
    }
  }

  private handleHit(target: THREE.Mesh) {
    const isHeadshot = target.userData.isHead;
    const points = isHeadshot ? 50 : 10;
    
    this.huntCoins += points;
    this.controls.updateCurrency(this.huntCoins);
    this.controls.showKill(isHeadshot ? `HEADSHOT! [+${points}]` : `Eliminated Target [+${points}]`);

    const parent = target.parent instanceof THREE.Group ? target.parent : target;
    
    const flash = (obj: THREE.Mesh) => {
      const mat = obj.material as THREE.MeshStandardMaterial;
      const oldColor = mat.color.clone();
      mat.color.set(0xffffff);
      setTimeout(() => mat.color.copy(oldColor), 100);
    };

    if (parent instanceof THREE.Group) {
      parent.children.forEach(child => {
        if (child instanceof THREE.Mesh) flash(child);
      });
    } else {
      flash(target);
    }

    setTimeout(() => {
      const newPos = new THREE.Vector3(
        (Math.random() - 0.5) * 80,
        0,
        (Math.random() - 0.5) * 80
      );
      parent.position.copy(newPos);
    }, 150);

    this.showHitMarker();
  }

  private showHitMarker() {
    const marker = document.createElement('div');
    marker.style.position = 'absolute';
    marker.style.top = '50%';
    marker.style.left = '50%';
    marker.style.width = '30px';
    marker.style.height = '30px';
    marker.style.border = '2px solid red';
    marker.style.borderRadius = '50%';
    marker.style.transform = 'translate(-50%, -50%) scale(0.5)';
    marker.style.transition = 'all 0.1s ease-out';
    marker.style.pointerEvents = 'none';
    marker.style.zIndex = '15';
    document.body.appendChild(marker);

    requestAnimationFrame(() => {
      marker.style.transform = 'translate(-50%, -50%) scale(1.5)';
      marker.style.opacity = '0';
    });

    setTimeout(() => marker.remove(), 100);
  }

  private initLights() {
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4); // Dimmer ambient
    this.scene.add(ambientLight);

    const sun = new THREE.DirectionalLight(0xffffff, 0.8);
    sun.position.set(50, 100, 50);
    sun.castShadow = true;
    this.scene.add(sun);
  }

  private initPracticeTargets() {
    for (let i = 0; i < 15; i++) { // More targets
      const group = new THREE.Group();

      // Body
      const bodyGeo = new THREE.BoxGeometry(1, 1.5, 0.5);
      const bodyMat = new THREE.MeshStandardMaterial({ color: 0x0000ff });
      const body = new THREE.Mesh(bodyGeo, bodyMat);
      body.position.y = 0.75;
      group.add(body);

      // Head
      const headGeo = new THREE.BoxGeometry(0.5, 0.5, 0.5);
      const headMat = new THREE.MeshStandardMaterial({ color: 0xffeeee });
      const head = new THREE.Mesh(headGeo, headMat);
      head.position.y = 1.75;
      head.userData.isHead = true;
      group.add(head);

      group.position.set(
        (Math.random() - 0.5) * 160,
        0,
        (Math.random() - 0.5) * 160
      );
      
      this.scene.add(group);
      this.targets.push(body, head);
    }
  }

  private onWindowResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }

  private animate() {
    requestAnimationFrame(() => this.animate());
    
    if (this.player) {
      this.player.update();
    }

    if (this.socket && this.socket.connected) {
      this.socket.emit('playerMovement', {
        position: this.camera.position,
        rotation: { x: 0, y: this.camera.rotation.y, z: 0 }
      });
    }

    this.renderer.render(this.scene, this.camera);
  }
}

new Game();
