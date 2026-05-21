import nipplejs from 'nipplejs';

export class Controls {
  public moveJoystick: nipplejs.JoystickManager | null = null;
  public lookJoystick: nipplejs.JoystickManager | null = null;
  public moveData = { x: 0, y: 0 };
  public lookData = { x: 0, y: 0 };

  constructor() {
    this.initUI();
    this.initJoysticks();
  }

  private initUI() {
    const ui = document.createElement('div');
    ui.id = 'ui-layer';
    ui.style.position = 'absolute';
    ui.style.top = '0';
    ui.style.left = '0';
    ui.style.width = '100%';
    ui.style.height = '100%';
    ui.style.pointerEvents = 'none';
    ui.style.zIndex = '10';
    document.body.appendChild(ui);

    const leftZone = document.createElement('div');
    leftZone.id = 'joystick-move';
    leftZone.style.position = 'absolute';
    leftZone.style.bottom = '10%';
    leftZone.style.left = '5%';
    leftZone.style.width = '150px';
    leftZone.style.height = '150px';
    leftZone.style.pointerEvents = 'auto';
    ui.appendChild(leftZone);

    const rightZone = document.createElement('div');
    rightZone.id = 'joystick-look';
    rightZone.style.position = 'absolute';
    rightZone.style.bottom = '10%';
    rightZone.style.right = '5%';
    rightZone.style.width = '150px';
    rightZone.style.height = '150px';
    rightZone.style.pointerEvents = 'auto';
    ui.appendChild(rightZone);

    const fireBtn = document.createElement('button');
    fireBtn.id = 'fire-button';
    fireBtn.innerText = 'FIRE';
    fireBtn.style.position = 'absolute';
    fireBtn.style.bottom = '35%';
    fireBtn.style.right = '10%';
    fireBtn.style.width = '80px';
    fireBtn.style.height = '80px';
    fireBtn.style.borderRadius = '50%';
    fireBtn.style.background = 'rgba(255, 0, 0, 0.6)';
    fireBtn.style.color = 'white';
    fireBtn.style.border = '2px solid white';
    fireBtn.style.fontSize = '16px';
    fireBtn.style.fontWeight = 'bold';
    fireBtn.style.pointerEvents = 'auto';
    fireBtn.style.zIndex = '20';
    ui.appendChild(fireBtn);

    fireBtn.addEventListener('touchstart', (e) => {
      e.preventDefault();
      this.onFire?.();
    });

    // Crosshair
    const crosshair = document.createElement('div');
    crosshair.style.position = 'absolute';
    crosshair.style.top = '50%';
    crosshair.style.left = '50%';
    crosshair.style.width = '20px';
    crosshair.style.height = '20px';
    crosshair.style.border = '2px solid white';
    crosshair.style.borderRadius = '50%';
    crosshair.style.transform = 'translate(-50%, -50%)';
    crosshair.style.pointerEvents = 'none';
    ui.appendChild(crosshair);

    const dot = document.createElement('div');
    dot.style.position = 'absolute';
    dot.style.top = '50%';
    dot.style.left = '50%';
    dot.style.width = '4px';
    dot.style.height = '4px';
    dot.style.background = 'white';
    dot.style.borderRadius = '50%';
    dot.style.transform = 'translate(-50%, -50%)';
    ui.appendChild(dot);

    // Currency UI
    const currencyUI = document.createElement('div');
    currencyUI.id = 'currency-display';
    currencyUI.innerText = 'HuntCoins: 0';
    currencyUI.style.position = 'absolute';
    currencyUI.style.top = '20px';
    currencyUI.style.right = '20px';
    currencyUI.style.color = 'yellow';
    currencyUI.style.fontSize = '24px';
    currencyUI.style.fontWeight = 'bold';
    currencyUI.style.textShadow = '2px 2px 0 #000';
    ui.appendChild(currencyUI);

    // Shop Button
    const shopBtn = document.createElement('button');
    shopBtn.innerText = 'SHOP';
    shopBtn.style.position = 'absolute';
    shopBtn.style.top = '20px';
    shopBtn.style.left = '20px';
    shopBtn.style.padding = '10px 20px';
    shopBtn.style.background = 'blue';
    shopBtn.style.color = 'white';
    shopBtn.style.border = '2px solid white';
    shopBtn.style.fontWeight = 'bold';
    shopBtn.style.pointerEvents = 'auto';
    ui.appendChild(shopBtn);

    shopBtn.onclick = () => this.toggleShop();

    // Shop Overlay
    const shopOverlay = document.createElement('div');
    shopOverlay.id = 'shop-overlay';
    shopOverlay.style.position = 'absolute';
    shopOverlay.style.top = '50%';
    shopOverlay.style.left = '50%';
    shopOverlay.style.transform = 'translate(-50%, -50%)';
    shopOverlay.style.width = '80%';
    shopOverlay.style.height = '60%';
    shopOverlay.style.background = 'rgba(0,0,0,0.9)';
    shopOverlay.style.border = '3px solid white';
    shopOverlay.style.display = 'none';
    shopOverlay.style.flexDirection = 'column';
    shopOverlay.style.alignItems = 'center';
    shopOverlay.style.padding = '20px';
    shopOverlay.style.color = 'white';
    shopOverlay.style.zIndex = '100';
    shopOverlay.style.pointerEvents = 'auto';
    ui.appendChild(shopOverlay);

    const shopTitle = document.createElement('h2');
    shopTitle.innerText = 'HUNTMAN SHOP';
    shopOverlay.appendChild(shopTitle);

    const itemList = document.createElement('div');
    itemList.style.display = 'flex';
    itemList.style.gap = '20px';
    itemList.style.marginTop = '20px';
    shopOverlay.appendChild(itemList);

    this.createShopItem(itemList, 'Speed Boost', 100, () => this.onPurchase?.('speed'));
    this.createShopItem(itemList, 'Red Crosshair', 50, () => this.onPurchase?.('red_crosshair'));

    const closeBtn = document.createElement('button');
    closeBtn.innerText = 'CLOSE';
    closeBtn.style.marginTop = 'auto';
    closeBtn.onclick = () => this.toggleShop();
    shopOverlay.appendChild(closeBtn);

    // Kill Feed Container
    const killFeed = document.createElement('div');
    killFeed.id = 'kill-feed';
    killFeed.style.position = 'absolute';
    killFeed.style.top = '20px';
    killFeed.style.right = '200px';
    killFeed.style.width = '250px';
    killFeed.style.display = 'flex';
    killFeed.style.flexDirection = 'column';
    killFeed.style.alignItems = 'flex-end';
    killFeed.style.pointerEvents = 'none';
    ui.appendChild(killFeed);
  }

  public showKill(message: string) {
    const feed = document.getElementById('kill-feed');
    if (!feed) return;

    const killMsg = document.createElement('div');
    killMsg.innerText = message;
    killMsg.style.background = 'rgba(0,0,0,0.5)';
    killMsg.style.color = 'white';
    killMsg.style.padding = '5px 10px';
    killMsg.style.marginBottom = '5px';
    killMsg.style.borderRadius = '5px';
    killMsg.style.fontSize = '14px';
    killMsg.style.borderRight = '4px solid red';
    feed.appendChild(killMsg);

    setTimeout(() => {
      killMsg.style.opacity = '0';
      killMsg.style.transition = 'opacity 0.5s';
      setTimeout(() => killMsg.remove(), 500);
    }, 3000);
  }

  private toggleShop() {
    const shop = document.getElementById('shop-overlay');
    if (shop) {
      shop.style.display = shop.style.display === 'none' ? 'flex' : 'none';
    }
  }

  private createShopItem(parent: HTMLElement, name: string, price: number, action: () => void) {
    const item = document.createElement('div');
    item.style.border = '1px solid white';
    item.style.padding = '10px';
    item.style.textAlign = 'center';

    const itemName = document.createElement('div');
    itemName.innerText = name;
    item.appendChild(itemName);

    const itemPrice = document.createElement('div');
    itemPrice.innerText = `${price} HC`;
    itemPrice.style.color = 'yellow';
    item.appendChild(itemPrice);

    const buyBtn = document.createElement('button');
    buyBtn.innerText = 'BUY';
    buyBtn.onclick = action;
    item.appendChild(buyBtn);

    parent.appendChild(item);
  }

  public onPurchase: ((item: string) => void) | null = null;

  public updateCurrency(amount: number) {
    const display = document.getElementById('currency-display');
    if (display) display.innerText = `HuntCoins: ${amount}`;
  }

  public onFire: (() => void) | null = null;

  private initJoysticks() {
    this.moveJoystick = nipplejs.create({
      zone: document.getElementById('joystick-move')!,
      mode: 'static',
      position: { left: '50%', top: '50%' },
      color: 'white',
    });

    this.moveJoystick.on('move', (evt, data) => {
      this.moveData.x = data.vector.x;
      this.moveData.y = data.vector.y;
    });

    this.moveJoystick.on('end', () => {
      this.moveData.x = 0;
      this.moveData.y = 0;
    });

    this.lookJoystick = nipplejs.create({
      zone: document.getElementById('joystick-look')!,
      mode: 'static',
      position: { left: '50%', top: '50%' },
      color: 'red',
    });

    this.lookJoystick.on('move', (evt, data) => {
      this.lookData.x = data.vector.x;
      this.lookData.y = data.vector.y;
    });

    this.lookJoystick.on('end', () => {
      this.lookData.x = 0;
      this.lookData.y = 0;
    });
  }
}
