export class Controls {
  public moveData = { x: 0, y: 0 };
  public lookData = { x: 0, y: 0 };
  public keys: { [key: string]: boolean } = {};
  
  public onJump: (() => void) | null = null;
  public onFire: (() => void) | null = null;
  public onAim: ((isAiming: boolean) => void) | null = null;

  constructor() {
    this.initTouchJoystick('zone-move', 'base-move', 'knob-move', (x, y) => {
      this.moveData.x = x;
      this.moveData.y = y;
    });

    this.initTouchJoystick('zone-look', 'base-look', 'knob-look', (x, y) => {
      this.lookData.x = x;
      this.lookData.y = y;
    });

    this.initButtons();
    this.initKeyboard();
  }

  private initTouchJoystick(zoneId: string, baseId: string, knobId: string, callback: (x: number, y: number) => void) {
    const zone = document.getElementById(zoneId)!;
    const base = document.getElementById(baseId)!;
    const knob = document.getElementById(knobId)!;
    const debug = document.getElementById('debug-log')!;

    let activeTouchId: number | null = null;
    let touchStartX = 0;
    let touchStartY = 0;

    const handleTouch = (e: TouchEvent) => {
      e.preventDefault();
      const touches = e.changedTouches;

      for (let i = 0; i < touches.length; i++) {
        const touch = touches[i];
        
        if (e.type === 'touchstart' && activeTouchId === null) {
          activeTouchId = touch.identifier;
          
          touchStartX = touch.clientX;
          touchStartY = touch.clientY;

          base.style.display = 'block';
          const zoneRect = zone.getBoundingClientRect();
          
          const localX = touchStartX - zoneRect.left;
          const localY = touchStartY - zoneRect.top;
          
          base.style.left = `${localX - 60}px`;
          base.style.top = `${localY - 60}px`;
          
          knob.style.left = '30px';
          knob.style.top = '30px';
          
          debug.innerText = `Touch Start: ${zoneId}`;
        } 
        
        if (touch.identifier === activeTouchId) {
          if (e.type === 'touchmove') {
            let dx = touch.clientX - touchStartX;
            let dy = touch.clientY - touchStartY;
            
            const maxDist = 60;
            const dist = Math.sqrt(dx*dx + dy*dy);
            
            if (dist > maxDist) {
              dx *= maxDist / dist;
              dy *= maxDist / dist;
            }
            
            knob.style.left = `${30 + dx}px`;
            knob.style.top = `${30 + dy}px`;
            
            callback(dx / maxDist, -dy / maxDist);
            debug.innerText = `Moving: ${zoneId} | X: ${(dx/maxDist).toFixed(2)} Y: ${(-dy/maxDist).toFixed(2)}`;
          }
          
          if (e.type === 'touchend' || e.type === 'touchcancel') {
            activeTouchId = null;
            base.style.display = 'none';
            callback(0, 0);
            debug.innerText = `Touch End: ${zoneId}`;
          }
        }
      }
    };

    zone.addEventListener('touchstart', handleTouch, { passive: false });
    zone.addEventListener('touchmove', handleTouch, { passive: false });
    zone.addEventListener('touchend', handleTouch, { passive: false });
    zone.addEventListener('touchcancel', handleTouch, { passive: false });
  }

  private initButtons() {
    const fire = document.getElementById('btn-fire')!;
    const jump = document.getElementById('btn-jump')!;
    const aim = document.getElementById('btn-aim')!;

    const handleFire = (e: Event) => {
      e.preventDefault();
      this.onFire?.();
    };
    fire.addEventListener('touchstart', handleFire);
    fire.addEventListener('mousedown', handleFire);

    const handleJump = (e: Event) => {
      e.preventDefault();
      this.onJump?.();
    };
    jump.addEventListener('touchstart', handleJump);
    jump.addEventListener('mousedown', handleJump);

    let isAiming = false;
    const handleAim = (e: Event) => {
      e.preventDefault();
      isAiming = !isAiming;
      aim.style.background = isAiming ? 'rgba(255,165,0,0.9)' : 'rgba(255,165,0,0.6)';
      this.onAim?.(isAiming);
    };
    aim.addEventListener('touchstart', handleAim);
    aim.addEventListener('mousedown', handleAim);
  }

  private initKeyboard() {
    window.addEventListener('keydown', (e) => {
      this.keys[e.code] = true;
      if (e.code === 'Space') this.onJump?.();
    });
    window.addEventListener('keyup', (e) => this.keys[e.code] = false);

    window.addEventListener('mousemove', (e) => {
      if (document.pointerLockElement) {
        this.lookData.x = -e.movementX * 0.05;
        this.lookData.y = -e.movementY * 0.05;
        setTimeout(() => { this.lookData.x = 0; this.lookData.y = 0; }, 20);
      }
    });

    document.addEventListener('mousedown', () => {
      if (!document.pointerLockElement && window.innerWidth > 1000) {
        document.body.requestPointerLock();
      } else {
        this.onFire?.();
      }
    });
  }

  public updateCurrency(amount: number) {
    const display = document.getElementById('currency-display');
    if (display) display.innerText = `HuntCoins: ${amount}`;
  }
}
