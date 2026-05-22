import * as THREE from 'three';
import { io, Socket } from 'socket.io-client';
import { Controls } from './ui/Controls';
import { Player } from './game/Player';
import { Map } from './game/Map';

class Game {
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private controls!: Controls;
  private player!: Player;
  private map!: Map;
  private targets: THREE.Mesh[] = [];
  private raycaster: THREE.Raycaster;
  private huntCoins: number = 0;
  
  private socket: Socket | null = null;
  private otherPlayers: { [id: string]: THREE.Group } = {};

  constructor() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x00050a);
    this.scene.fog = new THREE.FogExp2(0x00050a, 0.015);

    this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    this.camera.position.set(20, 1.7, 20);
    this.scene.add(this.camera);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    document.getElementById('app')?.appendChild(this.renderer.domElement);

    this.raycaster = new THREE.Raycaster();

    window.addEventListener('resize', () => this.onWindowResize());
    
    (window as any).startGame = (mode: string) => this.initGame(mode);
  }

  private initGame(mode: string) {
    document.getElementById('main-menu')!.style.display = 'none';
    document.getElementById('ui-root')!.style.display = 'block';

    this.map = new Map(this.scene);
    this.controls = new Controls();
    this.player = new Player(this.camera, this.controls);

    this.controls.onFire = () => this.shoot();
    this.controls.onJump = () => this.player.jump();
    this.controls.onAim = (isAiming) => this.player.setAim(isAiming);

    this.initLights();

    if (mode === 'practice') {
      this.initPracticeTargets();
    } else {
      this.connectToServer(mode);
    }

    this.animate();
  }

  private connectToServer(mode: string) {
    this.socket = io('http://localhost:3000');

    this.socket.on('connect', () => {
      if (mode === 'friend') {
        const roomId = prompt('Enter Room Code:', '123') || 'default';
        this.socket?.emit('joinRoom', roomId);
      }
    });

    this.socket.on('currentPlayers', (players: any) => {
      Object.keys(players).forEach((id) => {
        if (id !== this.socket?.id) this.addOtherPlayer(players[id]);
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
    const body = new THREE.Mesh(new THREE.BoxGeometry(1, 1.5, 0.5), new THREE.MeshStandardMaterial({ color: 0xff0000 }));
    body.position.y = 0.75;
    group.add(body);
    const head = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.5, 0.5), new THREE.MeshStandardMaterial({ color: 0xffcccc }));
    head.position.y = 1.75;
    head.userData.isHead = true;
    group.add(head);
    group.position.copy(playerInfo.position);
    this.scene.add(group);
    this.otherPlayers[playerInfo.id] = group;
  }

  private shoot() {
    this.player.shoot();
    // FIXED VECTOR2 INSTANTIATION
    this.raycaster.setFromCamera(new THREE.Vector2(0, 0), this.camera);
    const intersects = this.raycaster.intersectObjects(this.targets);
    if (intersects.length > 0) {
      this.handleHit(intersects[0].object as THREE.Mesh);
    }
  }

  private handleHit(target: THREE.Mesh) {
    const isHeadshot = target.userData.isHead;
    const points = isHeadshot ? 50 : 10;
    this.huntCoins += points;
    const display = document.getElementById('currency-display');
    if (display) display.innerText = `HuntCoins: ${this.huntCoins}`;
    
    const feed = document.getElementById('debug-log');
    if (feed) {
      feed.style.display = 'block';
      feed.innerText = isHeadshot ? 'HEADSHOT! +50' : 'TARGET HIT! +10';
      setTimeout(() => feed.style.display = 'none', 1000);
    }

    const parent = target.parent instanceof THREE.Group ? target.parent : target;
    setTimeout(() => {
      parent.position.set((Math.random() - 0.5) * 160, 0, (Math.random() - 0.5) * 160);
    }, 100);
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
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.4));
    const sun = new THREE.DirectionalLight(0xffffff, 0.8);
    sun.position.set(50, 100, 50);
    this.scene.add(sun);
  }

  private initPracticeTargets() {
    for (let i = 0; i < 15; i++) {
      const group = new THREE.Group();
      const body = new THREE.Mesh(new THREE.BoxGeometry(1, 1.5, 0.5), new THREE.MeshStandardMaterial({ color: 0x0000ff }));
      body.position.y = 0.75;
      group.add(body);
      const head = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.5, 0.5), new THREE.MeshStandardMaterial({ color: 0xffeeee }));
      head.position.y = 1.75;
      head.userData.isHead = true;
      group.add(head);
      group.position.set((Math.random() - 0.5) * 160, 0, (Math.random() - 0.5) * 160);
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
    if (this.player) this.player.update(this.map?.collidables);
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
