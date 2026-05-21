const express = require('express');
const http = require('http');
const { Server } = require('socket.io');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

const players = {};

io.on('connection', (socket) => {
  console.log('A user connected:', socket.id);

  players[socket.id] = {
    id: socket.id,
    position: { x: 0, y: 1.7, z: 0 },
    rotation: { x: 0, y: 0, z: 0 },
  };

  socket.emit('currentPlayers', players);
  socket.broadcast.emit('newPlayer', players[socket.id]);

  socket.on('playerMovement', (movementData) => {
    if (players[socket.id]) {
      players[socket.id].position = movementData.position;
      players[socket.id].rotation = movementData.rotation;
      socket.broadcast.emit('playerMoved', players[socket.id]);
    }
  });

  socket.on('disconnect', () => {
    console.log('User disconnected:', socket.id);
    delete players[socket.id];
    io.emit('playerDisconnected', socket.id);
  });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
